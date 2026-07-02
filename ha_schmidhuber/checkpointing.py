"""Training-state checkpoints, shared by train_vae.py and train_rnn.py.

Follows the same convention already used for the controller (models/controller-<run>/
checkpoint-<gen>.pt): one file per checkpoint under a per-run directory. Each file holds
full training state (model + optimizer + scheduler + step/epoch), not just weights, so a
run can resume after a preemption instead of restarting from scratch. Old checkpoints are
pruned to bound disk usage.
"""
import glob
import os

import torch


def checkpoint_every(total_steps, pct=0.01):
    """Step interval corresponding to `pct` of total_steps (at least 1 step)."""
    return max(1, round(total_steps * pct))


def _sorted_checkpoints(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint-*.pt"))
    return sorted(ckpts, key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))


def save_checkpoint(ckpt_dir, step, keep_last=5, **state):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"checkpoint-{step:07d}.pt")
    torch.save({"step": step, **state}, path)
    for old in _sorted_checkpoints(ckpt_dir)[:-keep_last]:
        os.remove(old)
    return path


def latest_checkpoint(ckpt_dir):
    ckpts = _sorted_checkpoints(ckpt_dir)
    return ckpts[-1] if ckpts else None


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)
