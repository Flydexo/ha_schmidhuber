"""Encode a dataset episode (or a live-driven session) through a trained VAE and render a
side-by-side video: original frames on the left, VAE autoencoded frames on the right.

From the dataset:
    uv run python vae_reconstruct_video.py \
        --vae models/vae-AR.pt --episode 0 --out recon_ep0.mp4

Drive it yourself (arrow keys steer/gas/brake) for --steps frames, then reconstruct:
    uv run python vae_reconstruct_video.py --vae models/vae-AR.pt --render --out drive.mp4
"""
import argparse
import os
import subprocess
import sys

import lancedb
import numpy as np
import torch
import torch.nn.functional as F

import model

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_device(name):
    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_episode(uri, episode_id, img_size):
    """Return (frames (T,H,W,3) uint8, actions (T,3) float32) for one episode."""
    table = lancedb.connect(uri).open_table("episodes")
    row = table.search().where(f"episode_id = {episode_id}").limit(1).to_arrow()
    if row.num_rows == 0:
        raise SystemExit(f"episode_id {episode_id} not found in {uri!r}")
    obs_list = row.column("observations")[0].values  # FixedSizeListArray (T, img*img*3)
    frames = obs_list.values.to_numpy(zero_copy_only=False).reshape(
        len(obs_list), img_size, img_size, 3
    )
    act_list = row.column("actions")[0].values  # FixedSizeListArray (T, 3)
    actions = act_list.values.to_numpy(zero_copy_only=False).reshape(len(act_list), -1)
    return frames, actions.astype(np.float32)


def collect_interactive(steps, img_size):
    """Open a live CarRacing window, let the user drive, and return
    (frames (T,H,W,3) uint8 resized as the dataset stores, actions (T,3) float32).
    action[t] is the action held while frame[t] was observed (dataset convention).

    Controls: arrow keys = steer / gas / brake, ESC or window-close = stop early.
    """
    import gymnasium as gym
    import pygame

    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # steer, gas, brake

    def register_input():
        stop = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = -1.0
                elif event.key == pygame.K_RIGHT:
                    action[0] = +1.0
                elif event.key == pygame.K_UP:
                    action[1] = +1.0
                elif event.key == pygame.K_DOWN:
                    action[2] = +0.8  # set 1.0 to lock the wheels
                elif event.key == pygame.K_ESCAPE:
                    stop = True
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    action[0] = 0.0
                elif event.key == pygame.K_UP:
                    action[1] = 0.0
                elif event.key == pygame.K_DOWN:
                    action[2] = 0.0
        return stop

    def resize(obs):
        t = torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0)
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).byte().numpy()

    print(
        f"driving for {steps} steps -- arrows = steer/gas/brake, ESC to stop early",
        file=sys.stderr,
    )
    obs, _ = env.reset(seed=0)
    frames, actions = [], []
    stop = False
    while len(frames) < steps and not stop:
        stop = register_input()
        frames.append(resize(obs))
        actions.append(action.copy())  # action held for this frame
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    pygame.quit()
    return np.asarray(frames, dtype=np.uint8), np.asarray(actions, dtype=np.float32)


def load_vae(vae_path, device):
    vae = model.AutoEncoder(cfg=None).to(device)
    state = torch.load(vae_path, map_location=device, weights_only=True)
    # Final .pt files hold clean keys; compiled checkpoints may prefix with _orig_mod.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    vae.load_state_dict(state)
    vae.eval()
    return vae


@torch.no_grad()
def reconstruct(vae, frames, device, batch_size):
    """(T,H,W,3) uint8 -> (T,H,W,3) uint8 VAE reconstructions (decoded from the mean)."""
    recon = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i : i + batch_size]
        x = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).float() / 255.0
        _, mu, _ = vae.encode(x)  # decode the posterior mean (deterministic)
        x_recon = vae.decoder(mu).clamp(0, 1)
        out = (x_recon.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy()
        recon.append(out)
    return np.concatenate(recon, axis=0)


def rnn_cfg(temp):
    """Minimal cfg for model.RNN/MDN (matches conf/config.yaml rnn defaults)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        rnn=SimpleNamespace(
            hidden_size=256, num_mix=5, z_dim=32, action_dim=3, temp=temp
        )
    )


def load_rnn(rnn_path, device, temp):
    rnn = model.RNN(rnn_cfg(temp)).to(device)
    state = torch.load(rnn_path, map_location=device, weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    rnn.load_state_dict(state)
    rnn.eval()
    return rnn


@torch.no_grad()
def rnn_predict(vae, rnn, frames, actions, device, batch_size, sample):
    """Teacher-forced one-step prediction. For each frame t (>=1), the MDN-RNN predicts
    z_t from the *real* previous latent z_{t-1}, action a_{t-1}, and the recurrent state
    built from all earlier steps; that predicted latent is decoded.

    Returns (orig (T-1,H,W,3), pred (T-1,H,W,3)) uint8, aligned to frames[1:].
    """
    # Encode every frame to its posterior-mean latent (the conditioning signal).
    z_all = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i : i + batch_size].copy()
        x = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).float() / 255.0
        _, mu, _ = vae.encode(x)
        z_all.append(mu)
    z_all = torch.cat(z_all, 0)  # (T, 32)

    a = torch.from_numpy(actions).to(device).float()  # (T, 3)
    # input step t = (z_t, a_t) -> predicts z_{t+1}; run the whole episode in one pass.
    z_in, a_in = z_all[:-1], a[:-1]  # (T-1, .)
    pi, mu, sigma, _, _ = rnn(z_in, a_in)  # pi (T-1,K), mu/sigma (T-1,K,32)

    if sample:
        z_pred = rnn.mdn.sample(pi, mu, sigma)  # draw from the mixture (uses --temp)
    else:
        z_pred = (pi.unsqueeze(-1) * mu).sum(1)  # mixture mean E[z_{t+1}]

    # Decode predicted latents (batched to bound memory).
    pred = []
    for i in range(0, len(z_pred), batch_size):
        x_recon = vae.decoder(z_pred[i : i + batch_size]).clamp(0, 1)
        pred.append((x_recon.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy())
    pred = np.concatenate(pred, 0)  # (T-1, H, W, 3)
    return frames[1:], pred


def upscale(frames, factor):
    if factor == 1:
        return frames
    return frames.repeat(factor, axis=1).repeat(factor, axis=2)


def write_video(path, left, right, fps, scale, gap):
    """Pipe raw side-by-side RGB frames to ffmpeg -> mp4."""
    left, right = upscale(left, scale), upscale(right, scale)
    T, H, W, _ = left.shape
    sep = np.zeros((T, H, gap, 3), dtype=np.uint8) if gap else None
    W_total = W * 2 + gap
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W_total}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(T):
        parts = [left[t], sep[t], right[t]] if gap else [left[t], right[t]]
        proc.stdin.write(np.concatenate(parts, axis=1).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vae", required=True, help="path to trained VAE .pt weights")
    p.add_argument("--rnn", default=None, help="path to trained MDN-RNN .pt; right panel "
                   "then shows the RNN's one-step-ahead prediction instead of the VAE recon")
    p.add_argument("--sample", action="store_true", help="sample the RNN mixture instead "
                   "of decoding its mean (uses --temp)")
    p.add_argument("--temp", type=float, default=0.2, help="MDN sampling temperature (--rnn)")
    p.add_argument("--episode", type=int, default=None, help="episode_id to encode")
    p.add_argument("--render", action="store_true", help="drive the car live, then encode")
    p.add_argument("--steps", type=int, default=1000, help="frames to collect when --render")
    p.add_argument("--out", default=None, help="output mp4 (default recon_ep<id>.mp4)")
    p.add_argument("--lancedb-uri", default=os.path.join(HERE, "db"))
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scale", type=int, default=4, help="integer upscale of the 64px frames")
    p.add_argument("--gap", type=int, default=8, help="black separator width (post-scale px)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-frames", type=int, default=None, help="cap frames for a quick preview")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = p.parse_args()

    if not args.render and args.episode is None:
        p.error("provide --episode <id>, or --render to drive the car live")

    device = pick_device(args.device)
    src = "drive" if args.render else f"ep{args.episode}"
    out = args.out or f"recon_{src}.mp4"
    print(f"device={device}  vae={args.vae}  source={src}", file=sys.stderr)

    if args.render:
        frames, actions = collect_interactive(args.steps, args.img_size)
    else:
        frames, actions = load_episode(args.lancedb_uri, args.episode, args.img_size)
    if args.max_frames:
        frames, actions = frames[: args.max_frames], actions[: args.max_frames]
    print(f"loaded {len(frames)} frames", file=sys.stderr)

    vae = load_vae(args.vae, device)
    if args.rnn:
        rnn = load_rnn(args.rnn, device, args.temp)
        left, right = rnn_predict(
            vae, rnn, frames, actions, device, args.batch_size, args.sample
        )
        kind = "RNN one-step prediction"
    else:
        left, right = frames, reconstruct(vae, frames, device, args.batch_size)
        kind = "reconstruction"

    write_video(out, left, right, args.fps, args.scale, args.gap)
    print(f"wrote {out}  ({len(left)} frames, left=original / right={kind})")


if __name__ == "__main__":
    main()
