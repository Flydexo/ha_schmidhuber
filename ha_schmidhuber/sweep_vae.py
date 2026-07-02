"""Grid sweep over VAE beta / free_bits / linear_beta, via Hydra multirun.

train_vae.py uses the @hydra.main decorator, so Hydra's own multirun sweeper (-m)
already does the grid launch, config composition, and per-job isolation -- no manual
itertools/subprocess loop needed. This script is just a documented, one-command entry
point for the specific ablation grid; for anything else just call `train_vae.py -m`
directly with whatever overrides you want swept (comma-separated = swept).

Each job's run_name is built from its own swept values via OmegaConf interpolation
(${training.vae.beta} etc. resolve to that job's value), so every grid point gets a
distinct trackio run automatically.

Usage:
    uv run python sweep_vae.py                        # full grid, sequential, real training
    uv run python sweep_vae.py profile=smoke           # smoke-test the sweep itself
    uv run python sweep_vae.py training.vae.lr=0.0003  # extra overrides forwarded to every job

Runs sequentially by default (Hydra's basic launcher). For concurrent runs, install
hydra-joblib-launcher and add `hydra/launcher=joblib hydra.launcher.n_jobs=N`.
"""
import subprocess
import sys

GRID = [
    "training.vae.beta=0.5,1.0,4.0",
    "training.vae.free_bits=true,false",
    "training.vae.linear_beta=true,false",
]
# Quoted so Hydra's override grammar treats the braces as a literal string, not its own
# (unrelated) dict-value syntax.
RUN_NAME_OVERRIDE = (
    'trackio.run_name="vae-sweep{beta=${training.vae.beta},'
    'free_bits=${training.vae.free_bits},linear_beta=${training.vae.linear_beta}}"'
)


def main():
    cmd = [sys.executable, "train_vae.py", "-m", *GRID, RUN_NAME_OVERRIDE, *sys.argv[1:]]
    print(f"$ {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
