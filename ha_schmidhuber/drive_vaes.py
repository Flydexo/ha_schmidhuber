"""Drive CarRacing yourself, record the session, then render a labeled grid video
comparing the original frames against the reconstruction of every VAE in a folder.

    uv run python drive_vaes.py --vaes models --out vaes_drive.mp4

The first cell of the grid is the original ("original"); each remaining cell is one
VAE's autoencoded reconstruction, captioned with the VAE's filename (without .pt).
Instead of driving, you can replay a dataset episode with --episode <id>.

Controls while driving: arrow keys = steer / gas / brake, ESC or window-close = stop.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Reuse the single-VAE tooling so the two scripts stay in lockstep.
from vae_reconstruct_video import (
    collect_interactive,
    load_episode,
    load_vae,
    pick_device,
    reconstruct,
    upscale,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def find_vaes(folder):
    """Return [(name, path), ...] for every VAE .pt directly in `folder`, sorted by name.

    Only files whose basename starts with 'vae' are treated as VAEs, so a mixed
    models/ dir (rnn-*.pt, controller-*.pt) can be pointed at directly.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.pt")))
    vaes = [
        (os.path.splitext(os.path.basename(p))[0], p)
        for p in paths
        if os.path.basename(p).lower().startswith("vae")
    ]
    if not vaes:
        raise SystemExit(f"no vae-*.pt files found in {folder!r}")
    return vaes


def _font(size):
    """A truetype font if we can find one, else PIL's bitmap default."""
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_strip(text, width, height, font):
    """A (height, width, 3) uint8 caption bar: white text centered on dark grey."""
    img = Image.new("RGB", (width, height), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    # Shrink the text to fit the cell width if the filename is long.
    f = font
    while True:
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        tw, th = r - l, b - t
        if tw <= width - 6 or getattr(f, "size", 8) <= 8:
            break
        f = _font(max(8, f.size - 1))
    draw.text(((width - tw) / 2 - l, (height - th) / 2 - t), text, font=f, fill=(240, 240, 240))
    return np.asarray(img, dtype=np.uint8)


def build_grid(cells, labels, cols, scale, label_h, gap):
    """cells: list of (T,H,W,3) uint8 stacks (already same T,H,W). labels: matching names.

    Returns (T, GH, GW, 3) uint8: each cell upscaled with a caption bar on top, tiled
    row-major into a `cols`-wide grid, black `gap` between cells, empty slots black.
    """
    n = len(cells)
    rows = (n + cols - 1) // cols
    cells = [upscale(c, scale) for c in cells]
    T, H, W, _ = cells[0].shape
    font = _font(max(10, W // 8))
    caps = [label_strip(name, W, label_h, font) for name in labels]  # (label_h, W, 3) each

    cell_h, cell_w = label_h + H, W
    blank_cell = np.zeros((T, cell_h, cell_w, 3), dtype=np.uint8)

    def cell_video(idx):
        if idx >= n:
            return blank_cell
        cap = np.broadcast_to(caps[idx], (T, label_h, W, 3))
        return np.concatenate([cap, cells[idx]], axis=1)  # (T, cell_h, W, 3)

    hgap = np.zeros((T, cell_h, gap, 3), dtype=np.uint8) if gap else None

    row_videos = []
    for r in range(rows):
        parts = []
        for c in range(cols):
            parts.append(cell_video(r * cols + c))
            if gap and c < cols - 1:
                parts.append(hgap)
        row_videos.append(np.concatenate(parts, axis=2))  # (T, cell_h, row_w, 3)

    row_w = row_videos[0].shape[2]
    vgap = np.zeros((T, gap, row_w, 3), dtype=np.uint8) if gap else None
    grid_parts = []
    for r, rv in enumerate(row_videos):
        grid_parts.append(rv)
        if gap and r < rows - 1:
            grid_parts.append(vgap)
    return np.concatenate(grid_parts, axis=1)  # (T, GH, GW, 3)


def write_video(path, grid, fps):
    """Pipe raw RGB grid frames to ffmpeg -> mp4."""
    T, H, W, _ = grid.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(T):
        proc.stdin.write(np.ascontiguousarray(grid[t]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vaes", required=True, help="folder holding the vae-*.pt weights")
    p.add_argument("--episode", type=int, default=None,
                   help="replay this dataset episode instead of driving live")
    p.add_argument("--steps", type=int, default=1000, help="frames to collect when driving")
    p.add_argument("--out", default="vaes_drive.mp4", help="output mp4 path")
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
    vaes = find_vaes(args.vaes)
    print(f"device={device}  found {len(vaes)} vae(s): "
          + ", ".join(n for n, _ in vaes), file=sys.stderr)

    # Record the session once; every VAE sees the same frames.
    if args.episode is not None:
        frames, _ = load_episode(args.lancedb_uri, args.episode, args.img_size)
    else:
        frames, _ = collect_interactive(args.steps, args.img_size)
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"recorded {len(frames)} frames", file=sys.stderr)

    cells = [frames]                 # first cell = original
    labels = ["original"]
    for name, path in vaes:
        print(f"encoding through {name} ...", file=sys.stderr)
        try:
            vae = load_vae(path, device)
        except RuntimeError as e:
            # Older checkpoints predate the current model.AutoEncoder layout; skip them.
            print(f"  skipping {name}: incompatible checkpoint ({str(e).splitlines()[0]})",
                  file=sys.stderr)
            continue
        cells.append(reconstruct(vae, frames, device, args.batch_size))
        labels.append(name)
        del vae

    if len(cells) == 1:
        raise SystemExit("no loadable VAEs -- nothing to compare against the original")

    n = len(cells)
    cols = args.cols or max(1, int(np.ceil(np.sqrt(n))))
    grid = build_grid(cells, labels, cols, args.scale, args.label_h, args.gap)
    write_video(args.out, grid, args.fps)
    print(f"wrote {args.out}  ({grid.shape[0]} frames, {n} cells in a "
          f"{(n + cols - 1) // cols}x{cols} grid)")


if __name__ == "__main__":
    main()
