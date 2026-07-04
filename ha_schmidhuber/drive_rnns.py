"""Drive CarRacing yourself, record the session, then render a labeled grid video
comparing the original frames against the one-step-ahead prediction of every MDN-RNN
in a folder -- all sharing a single VAE for encoding/decoding.

    uv run python drive_rnns.py --vae vae.pt --rnns models --out rnns_drive.mp4

The first cell of the grid is the original ("original"); each remaining cell is one
RNN's teacher-forced next-frame prediction, captioned with the RNN's filename.
Instead of driving, you can replay a dataset episode with --episode <id>.

Controls while driving: arrow keys = steer / gas / brake, ESC or window-close = stop.
"""
import argparse
import glob
import os
import sys

import numpy as np

# Grid rendering is shared with the VAE-comparison script.
from drive_vaes import build_grid, write_video
from vae_reconstruct_video import (
    collect_interactive,
    load_episode,
    load_rnn,
    load_vae,
    pick_device,
    rnn_predict,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def find_rnns(folder):
    """Return [(name, path), ...] for every RNN .pt directly in `folder`, sorted by name.

    Only files whose basename starts with 'rnn' are treated as RNNs, so a mixed
    models/ dir (vae-*.pt, controller-*.pt) can be pointed at directly.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.pt")))
    rnns = [
        (os.path.splitext(os.path.basename(p))[0], p)
        for p in paths
        if os.path.basename(p).lower().startswith("rnn")
    ]
    if not rnns:
        raise SystemExit(f"no rnn-*.pt files found in {folder!r}")
    return rnns


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae", required=True, help="single VAE .pt used to encode/decode")
    p.add_argument("--rnns", required=True, help="folder holding the rnn-*.pt weights")
    p.add_argument("--sample", action="store_true", help="sample each RNN mixture instead "
                   "of decoding its mean (uses --temp)")
    p.add_argument("--temp", type=float, default=0.2, help="MDN sampling temperature")
    p.add_argument("--episode", type=int, default=None,
                   help="replay this dataset episode instead of driving live")
    p.add_argument("--steps", type=int, default=1000, help="frames to collect when driving")
    p.add_argument("--out", default="rnns_drive.mp4", help="output mp4 path")
    p.add_argument("--lancedb-uri", default=os.path.join(HERE, "db"))
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scale", type=int, default=3, help="integer upscale of the 64px cells")
    p.add_argument("--cols", type=int, default=0, help="grid columns (0 = auto ~square)")
    p.add_argument("--label-h", type=int, default=22, help="caption bar height (post-scale px)")
    p.add_argument("--gap", type=int, default=6, help="black separator width between cells")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-frames", type=int, default=None, help="cap frames for a quick preview")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = p.parse_args()

    device = pick_device(args.device)
    rnns = find_rnns(args.rnns)
    print(f"device={device}  vae={args.vae}  found {len(rnns)} rnn(s): "
          + ", ".join(n for n, _ in rnns), file=sys.stderr)

    # Record the session once; every RNN sees the same frames + actions.
    if args.episode is not None:
        frames, actions = load_episode(args.lancedb_uri, args.episode, args.img_size)
    else:
        frames, actions = collect_interactive(args.steps, args.img_size)
    if args.max_frames:
        frames, actions = frames[: args.max_frames], actions[: args.max_frames]
    print(f"recorded {len(frames)} frames", file=sys.stderr)

    vae = load_vae(args.vae, device)

    # rnn_predict returns predictions aligned to frames[1:]; the original cell matches.
    cells, labels = [frames[1:]], ["original"]
    for name, path in rnns:
        print(f"predicting with {name} ...", file=sys.stderr)
        try:
            rnn = load_rnn(path, device, args.temp)
        except RuntimeError as e:
            # Older checkpoints predate the current model.RNN/MDN layout; skip them.
            print(f"  skipping {name}: incompatible checkpoint ({str(e).splitlines()[0]})",
                  file=sys.stderr)
            continue
        _, pred = rnn_predict(
            vae, rnn, frames, actions, device, args.batch_size, args.sample
        )
        cells.append(pred)
        labels.append(name)
        del rnn

    if len(cells) == 1:
        raise SystemExit("no loadable RNNs -- nothing to compare against the original")

    n = len(cells)
    cols = args.cols or max(1, int(np.ceil(np.sqrt(n))))
    grid = build_grid(cells, labels, cols, args.scale, args.label_h, args.gap)
    write_video(args.out, grid, args.fps)
    print(f"wrote {args.out}  ({grid.shape[0]} frames, {n} cells in a "
          f"{(n + cols - 1) // cols}x{cols} grid; left cell=original, "
          f"rest=one-step RNN prediction)")


if __name__ == "__main__":
    main()
