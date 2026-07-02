"""Ablation over the free-bits KL floor: free_bits_nats = 16 vs 32.

Both jobs run with free_bits=true (the floor only bites when it's on); the only thing swept
is how many nats of total KL are protected from being penalised toward zero. The point is to
see the effect on posterior collapse -- watch the `posterior/*` metrics (dead_dims, mean_sigma,
active_dims) that train_vae.py logs each epoch.

Like sweep_vae.py this is a thin wrapper over Hydra multirun (-m): comma-separated values are
swept, and each grid point gets a distinct trackio run via run_name interpolation.

Usage:
    uv run python ablate_free_bits.py                 # real training, 16 vs 32, sequential
    uv run python ablate_free_bits.py profile=smoke    # smoke-test the ablation plumbing
    uv run python ablate_free_bits.py training.vae.beta=1.0   # extra overrides forwarded to both
"""
import subprocess
import sys

GRID = [
    "training.vae.free_bits=true",
    "training.vae.free_bits_nats=16,32",
]
# Quoted so Hydra's override grammar treats the braces as a literal string.
RUN_NAME_OVERRIDE = 'trackio.run_name="vae-fb{nats=${training.vae.free_bits_nats}}"'


def main():
    cmd = [sys.executable, "train_vae.py", "-m", *GRID, RUN_NAME_OVERRIDE, *sys.argv[1:]]
    print(f"$ {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
