import subprocess
import sys

GRID = [
    "training.vae.beta=0.5,1.0,2.0",
    "training.vae.free_bits=true,false",
    'training.vae.beta_schedule="constant","tanh","sigmoid"',
    "training.vae.lambda=0.125,0.25,0.5,1,2",
]
# Quoted so Hydra's override grammar treats the braces as a literal string, not its own
# (unrelated) dict-value syntax.
RUN_NAME_OVERRIDE = (
    'trackio.run_name="beta=${training.vae.beta},'
    'free_bits=${training.vae.free_bits},beta_schedule=${training.vae.beta_schedule},lambda=${training.vae.lambda}"'
)


def main():
    cmd = [
        sys.executable,
        "train_vae.py",
        "-m",
        *GRID,
        RUN_NAME_OVERRIDE,
        *sys.argv[1:],
    ]
    print(f"$ {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
