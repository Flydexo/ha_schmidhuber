"""Drive one CarRacing episode with a trained controller and render a 3-panel GIF:

    [ real frame | VAE reconstruction | MDN-RNN one-step-ahead prediction ]

The controller drives from [z; h] (VAE latent + RNN hidden state). We record the 64x64
frames it saw and the actions it took, then:
  - panel 2 = VAE(vae.pt) encode->decode of each real frame  (what V "sees")
  - panel 3 = MDN-RNN(rnn) teacher-forced prediction of frame t from (z_{t-1}, a_{t-1})
              decoded through the same VAE                    (what M "imagines next")

    uv run python controller_triptych_gif.py \
        --controller models/controller-controller-2.pt \
        --vae vae.pt --rnn "rnn-rnn{epochs=20,episodes=1k}.pt" \
        --out evaluations/controller_triptych.gif
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import model
from vae_reconstruct_video import (
    load_rnn,
    load_vae,
    pick_device,
    reconstruct,
    rnn_predict,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_controller(path, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    W = sd["layer.weight"].to(device)  # (3, 288)
    b = sd["layer.bias"].to(device)  # (3,)
    return W, b


def resize64(obs, img_size):
    t = torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).byte().numpy()


@torch.no_grad()
def drive(controller, vae, rnn, cfg, device, max_steps, seed, action_map, img_size):
    """Roll out one controller-driven episode; return (frames uint8 T,H,W,3), (actions T,3), reward."""
    import gymnasium as gym

    W, b = controller
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True,
                   lap_complete_percent=0.95, domain_randomize=False,
                   max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    h = torch.zeros(256, device=device)
    hidden = None
    frames, actions = [], []
    total = 0.0
    for _ in range(max_steps):
        frame = resize64(obs, img_size)
        x = torch.from_numpy(frame).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        _, z, _ = vae.encode(x)  # posterior mean latent (1, 32)
        z = z.squeeze(0)
        a = W @ torch.cat([z, h]) + b  # (3,)
        if action_map == "sigmoid":
            a = torch.cat([torch.tanh(a[:1]), torch.sigmoid(a[1:])])
        else:  # tanh-clamp
            a = torch.cat([torch.tanh(a[:1]), torch.clamp(torch.tanh(a[1:]), 0, 1)])
        a_np = a.cpu().numpy().astype(np.float32)
        frames.append(frame)
        actions.append(a_np)
        obs, r, term, trunc, _ = env.step(a_np)
        total += r
        _, _, _, hidden, out = rnn(z.view(1, 1, -1), a.view(1, 1, -1), hidden)
        h = out.reshape(-1)
        if term or trunc:
            break
    env.close()
    return np.asarray(frames, np.uint8), np.asarray(actions, np.float32), total


def label_panel(img_uint8, text, scale, label_h):
    """Upscale a (H,W,3) frame and stack a captioned bar on top; returns (H*scale+label_h, W*scale, 3)."""
    H, W, _ = img_uint8.shape
    big = np.asarray(Image.fromarray(img_uint8).resize((W * scale, H * scale), Image.NEAREST))
    bar = Image.new("RGB", (W * scale, label_h), (17, 17, 17))
    d = ImageDraw.Draw(bar)
    tw = d.textlength(text)
    d.text(((W * scale - tw) / 2, max(0, (label_h - 11) // 2)), text, fill=(235, 235, 235))
    return np.concatenate([np.asarray(bar), big], axis=0)


def build_triptych(real, recon, pred, scale, label_h, gap, labels):
    """Compose T frames of [real | recon | pred] with captions and black gutters."""
    T = len(real)
    out = []
    sep = None
    for t in range(T):
        cells = [
            label_panel(real[t], labels[0], scale, label_h),
            label_panel(recon[t], labels[1], scale, label_h),
            label_panel(pred[t], labels[2], scale, label_h),
        ]
        if gap:
            if sep is None:
                sep = np.zeros((cells[0].shape[0], gap, 3), np.uint8)
            row = np.concatenate([cells[0], sep, cells[1], sep, cells[2]], axis=1)
        else:
            row = np.concatenate(cells, axis=1)
        out.append(row)
    return np.asarray(out, np.uint8)


def write_gif(path, frames, fps):
    """Pipe RGB frames to ffmpeg with a shared palette for a clean, small GIF."""
    T, H, W, _ = frames.shape
    vf = ("split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-vf", vf, "-loop", "0", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="models/controller-controller-2.pt")
    p.add_argument("--vae", default="vae.pt")
    p.add_argument("--rnn", default="rnn-rnn{epochs=20,episodes=1k}.pt")
    p.add_argument("--out", default="evaluations/controller_triptych.gif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--action-map", choices=["sigmoid", "tanh-clamp", "auto"], default="auto")
    p.add_argument("--sample", action="store_true", help="sample the RNN mixture (else decode its mean)")
    p.add_argument("--temp", type=float, default=1.0, help="MDN sampling temperature")
    p.add_argument("--gif-start", type=int, default=0, help="first recorded frame to include")
    p.add_argument("--gif-frames", type=int, default=250, help="number of frames in the GIF")
    p.add_argument("--stride", type=int, default=2, help="keep every Nth frame in the GIF")
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--label-h", type=int, default=18)
    p.add_argument("--gap", type=int, default=6)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    cfg = None
    print(f"device={device}  controller={args.controller}", file=sys.stderr)

    vae = load_vae(args.vae, device)
    rnn = load_rnn(args.rnn, device, args.temp)
    controller = load_controller(args.controller, device)

    maps = ["sigmoid", "tanh-clamp"] if args.action_map == "auto" else [args.action_map]
    best = None
    for m in maps:
        frames, actions, reward = drive(controller, vae, rnn, cfg, device,
                                        args.max_steps, args.seed, m, args.img_size)
        print(f"action-map={m:10s}  frames={len(frames):4d}  reward={reward:.1f}", file=sys.stderr)
        if best is None or reward > best[2]:
            best = (frames, actions, reward, m)
    frames, actions, reward, chosen = best
    print(f"using action-map={chosen}  reward={reward:.1f}  ({len(frames)} frames)", file=sys.stderr)

    # VAE reconstruction of every real frame, and RNN one-step prediction (aligned to frames[1:]).
    recon = reconstruct(vae, frames, device, args.batch_size)
    _, pred = rnn_predict(vae, rnn, frames, actions, device, args.batch_size, args.sample)

    # Align all three to frames[1:] (the RNN can only predict from t>=1).
    real1, recon1 = frames[1:], recon[1:]
    # window + stride to keep the GIF small
    s, n, st = args.gif_start, args.gif_frames, args.stride
    sl = slice(s, s + n * st, st)
    real1, recon1, pred = real1[sl], recon1[sl], pred[sl]
    print(f"gif frames: {len(real1)}", file=sys.stderr)

    labels = ("real", "VAE recon", "RNN +1 pred")
    grid = build_triptych(real1, recon1, pred, args.scale, args.label_h, args.gap, labels)
    os.makedirs(os.path.dirname(os.path.join(HERE, args.out)) or ".", exist_ok=True)
    write_gif(os.path.join(HERE, args.out), grid, args.fps)
    size_mb = os.path.getsize(os.path.join(HERE, args.out)) / 1e6
    print(f"wrote {args.out}  ({grid.shape[0]} frames, {grid.shape[2]}x{grid.shape[1]}px, "
          f"{size_mb:.2f} MB, reward={reward:.0f}, action-map={chosen})")


if __name__ == "__main__":
    main()
