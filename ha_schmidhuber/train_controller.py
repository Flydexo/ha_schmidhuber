"""Parallel CMA-ES training of the World Models controller (Ha & Schmidhuber).

The VAE and MDN-RNN are frozen; CMA-ES evolves the single linear controller
    a_t = W [z_t ; h_t] + b
evaluated in the real CarRacing environment.

Speed comes from two levels of parallelism:
  * env-level: gymnasium AsyncVectorEnv steps the whole CMA population's
    environments in parallel worker processes (Box2D physics is CPU-bound).
  * batch-level: the shared VAE / MDN-RNN and the per-candidate controller
    run as a single batched forward over the population on the GPU (mps).

Run as a module (the __main__ guard is required for the 'spawn' start method
used on macOS):
    uv run python train_controller.py --generations 300 --avg 16
"""
import os
import time
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import cma
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from hydra import initialize_config_dir, compose
from dotenv import load_dotenv
from tqdm import tqdm
import trackio

import model

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cfg():
    load_dotenv(os.path.join(HERE, ".env"))
    with initialize_config_dir(version_base=None, config_dir=os.path.join(HERE, "conf")):
        overrides = []
        url = os.environ.get("TRACKIO_WRITE_URL")
        if url:
            overrides.append(f'trackio.write_url="{url}"')
        return compose(config_name="config", overrides=overrides)


class Controller(nn.Module):
    """Kept identical to the notebook so saved state_dicts are interchangeable."""

    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(
            cfg.controller.state_dim + cfg.controller.hidden_dim,
            cfg.controller.action_dim,
        )

    def forward(self, z, h):
        return self.layer(torch.cat((z, h), dim=-1))


def make_env(max_steps, render_mode="rgb_array"):
    # Top-level + primitive args so functools.partial(make_env, ...) is picklable
    # for AsyncVectorEnv's 'spawn' workers.
    return gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
        max_episode_steps=max_steps,
    )


def preprocess(obs_np, cfg):
    # (N, 96, 96, 3) uint8 -> (N, 3, 64, 64) float in [0,1] on device
    obs = torch.from_numpy(obs_np).permute(0, 3, 1, 2).to(cfg.device).float()
    obs = F.interpolate(
        obs, size=(cfg.dataset.img_size, cfg.dataset.img_size),
        mode="bilinear", align_corners=False,
    )
    return obs / 255


def _sync(device):
    # Make GPU work actually complete so timings aren't misattributed. No-op on CPU.
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def _episode(W, b, envs, vae, rnn, cfg, max_steps, profile=False):
    """One batched episode over all N lanes; returns per-lane cumulative reward."""
    dev = cfg.device
    t = {"reset": 0.0, "forward": 0.0, "step": 0.0, "rnn": 0.0, "prep": 0.0}
    N = W.shape[0]

    t0 = time.perf_counter()
    obs, _ = envs.reset()
    obs = preprocess(obs, cfg)
    _sync(dev)
    t["reset"] += time.perf_counter() - t0

    hidden = None                                                    # LSTM (h, c), all lanes
    h = torch.zeros(N, cfg.controller.hidden_dim, device=cfg.device)  # controller input h_t
    total = np.zeros(N, dtype=np.float64)
    active = np.ones(N, dtype=bool)
    steps = 0

    for _ in range(max_steps):
        t0 = time.perf_counter()
        _, z, _ = vae.encode(obs)                                   # (N, 32) use mu: deterministic latent, no sampling noise
        x = torch.cat([z, h], dim=-1).unsqueeze(-1)                 # (N, 288, 1)
        a = torch.bmm(W, x).squeeze(-1) + b                         # (N, 3) per-candidate linear
        # tanh bounds steering to [-1,1]; sigmoid bounds gas/brake to [0,1]
        a = torch.cat([torch.tanh(a[:, :1]), torch.sigmoid(a[:, 1:])], dim=-1)
        a_np = a.cpu().numpy().astype(np.float32)
        _sync(dev)
        t["forward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        obs_np, reward, terminated, truncated, _ = envs.step(a_np)
        t["step"] += time.perf_counter() - t0
        total += reward * active                                    # freeze reward after a lane finishes

        t0 = time.perf_counter()
        _, _, _, hidden, out = rnn(z.unsqueeze(1), a.unsqueeze(1), hidden)  # out: (N, 1, 256)
        h = out.squeeze(1)
        _sync(dev)
        t["rnn"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        obs = preprocess(obs_np, cfg)
        _sync(dev)
        t["prep"] += time.perf_counter() - t0

        steps += 1
        active = active & ~(terminated | truncated)
        if not active.any():
            break

    if profile:
        tot = sum(t.values()) or 1e-9
        parts = " ".join(f"{k}={v:.2f}s({100*v/tot:.0f}%|{1000*v/steps:.1f}ms/st)"
                         for k, v in t.items())
        print(f"[profile] N={N} steps={steps} total={tot:.2f}s  {parts}")

    return total


@torch.no_grad()
def evaluate(solutions, envs, vae, rnn, cfg, max_steps, avg, profile=False):
    """Fitness per candidate, averaged over `avg` rollouts.

    mu (deterministic latent) removes VAE sampling noise; averaging over rollouts
    removes the CarRacing random-track noise, which is the dominant source. Paper uses 16.
    """
    N = len(solutions)
    in_dim = cfg.controller.state_dim + cfg.controller.hidden_dim
    out_dim = cfg.controller.action_dim
    # Layout matches parameters_to_vector -> [weight (out,in) row-major, bias (out,)]
    params = torch.tensor(np.array(solutions), dtype=torch.float32, device=cfg.device)
    W = params[:, :out_dim * in_dim].view(N, out_dim, in_dim)       # (N, 3, 288)
    b = params[:, out_dim * in_dim:]                                # (N, 3)

    fit = np.zeros(N, dtype=np.float64)
    for _ in range(avg):
        fit += _episode(W, b, envs, vae, rnn, cfg, max_steps, profile)
    return fit / avg


@torch.no_grad()
def dream_evaluate(solutions, rnn, reward_head, cfg, max_steps, avg):
    """Fitness by rolling the controller *inside* the MDN-RNN (the paper's "dream").

    No real env: z_{t+1} is sampled from the RNN's mixture and reward is predicted by
    reward_head(h_t). Pure batched tensor ops, so a full-population rollout is ~ms on GPU.

    NOTE: the CarRacing MDN-RNN only predicts next-z, not reward, so this is meaningful
    only with a *trained* reward head (models/reward-<run>.pt). With an untrained head it
    still runs (and is fast) but optimises noise -- it's here for experimentation, not for
    reproducing the paper score, which is obtained in the real env.
    """
    N = len(solutions)
    in_dim = cfg.controller.state_dim + cfg.controller.hidden_dim
    out_dim = cfg.controller.action_dim
    params = torch.tensor(np.array(solutions), dtype=torch.float32, device=cfg.device)
    W = params[:, :out_dim * in_dim].view(N, out_dim, in_dim)       # (N, 3, 288)
    b = params[:, out_dim * in_dim:]                                # (N, 3)

    fit = torch.zeros(N, device=cfg.device)
    for _ in range(avg):
        z = torch.randn(N, cfg.rnn.z_dim, device=cfg.device)        # seed from the VAE prior N(0, I)
        h = torch.zeros(N, cfg.controller.hidden_dim, device=cfg.device)
        hidden = None
        for _ in range(max_steps):
            x = torch.cat([z, h], dim=-1).unsqueeze(-1)             # (N, 288, 1)
            a = torch.bmm(W, x).squeeze(-1) + b                     # (N, 3)
            a = torch.cat([torch.tanh(a[:, :1]), torch.sigmoid(a[:, 1:])], dim=-1)
            fit += reward_head(h).squeeze(-1)                       # predicted reward for this step
            pi, mu, sigma, hidden, out = rnn(z.unsqueeze(1), a.unsqueeze(1), hidden)
            z = rnn.mdn.sample(pi.squeeze(1), mu.squeeze(1), sigma.squeeze(1))  # dreamed next latent
            h = out.squeeze(1)
    return (fit / avg).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--popsize", type=int, default=None, help="CMA population (default: cma's own)")
    p.add_argument("--sigma", type=float, default=0.3, help="CMA initial step size")
    p.add_argument("--max-steps", type=int, default=1000, help="steps per rollout (paper: 1000)")
    p.add_argument("--avg", type=int, default=16, help="rollouts averaged per candidate (paper: 16)")
    p.add_argument("--render", action="store_true",
                   help="show one env in a window (forces SyncVectorEnv, slower)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch intra-op threads; keep low so parallelism comes from env processes")
    p.add_argument("--profile", action="store_true",
                   help="print per-generation timing breakdown (reset/forward/step/rnn/prep)")
    p.add_argument("--dream", action="store_true",
                   help="train inside the MDN-RNN dream (latent rollouts, no real env; "
                        "needs models/reward-<run>.pt to be meaningful)")
    args = p.parse_args()

    # On a many-core box, torch defaults to one thread per core. For our tiny
    # per-step batch that is pure overhead and steals cores from the env workers.
    torch.set_num_threads(args.torch_threads)

    cfg = load_cfg()
    run = cfg.trackio.run_name
    device = cfg.device

    vae = model.AutoEncoder(cfg).to(device)
    vae.load_state_dict(torch.load(os.path.join(HERE, f"models/vae-{run}.pt"), weights_only=True, map_location=torch.device(cfg.device)))
    vae.eval()
    rnn = model.RNN(cfg).to(device)
    rnn.load_state_dict(torch.load(os.path.join(HERE, f"models/rnn-{run}.pt"), weights_only=True, map_location=torch.device(cfg.device)))
    rnn.eval()

    reward_head = None
    if args.dream:
        reward_head = nn.Linear(cfg.rnn.hidden_size, 1).to(device)
        rpath = os.path.join(HERE, f"models/reward-{run}.pt")
        if os.path.exists(rpath):
            reward_head.load_state_dict(torch.load(rpath, weights_only=True, map_location=device))
            print(f"dream: loaded reward head from {rpath}")
        else:
            print(f"dream: WARNING no {rpath} -- using an UNTRAINED reward head; "
                  "fitness is not meaningful (real-env training is what reproduces the paper)")
        reward_head.eval()

    x0 = parameters_to_vector(Controller(cfg).parameters()).detach().numpy()
    opts = {"seed": args.seed}
    if args.popsize is not None:
        opts["popsize"] = args.popsize
    es = cma.CMAEvolutionStrategy(x0, args.sigma, opts)

    N = es.popsize
    max_steps = args.max_steps
    envs = None
    if args.dream:
        pass                                                # no real env in dream mode
    elif args.render:
        # One human-rendered lane; Sync runs in-process so the window works on macOS.
        fns = [partial(make_env, max_steps, "human")] + \
              [partial(make_env, max_steps) for _ in range(N - 1)]
        envs = gym.vector.SyncVectorEnv(fns)
    else:
        envs = gym.vector.AsyncVectorEnv([partial(make_env, max_steps) for _ in range(N)])
    print(f"population={N}  max_steps={max_steps}  avg={args.avg}  "
          f"render={args.render}  dream={args.dream}  device={device}")

    use_trackio = True
    try:
        trackio.init(
            name=f"controller-{run}",
            project=cfg.trackio.project,
            server_url=cfg.trackio.write_url,
            config={
                "generations": args.generations,
                "popsize": N,
                "sigma": args.sigma,
                "seed": args.seed,
                "max_steps": max_steps,
                "avg": args.avg,
            },
        )
    except Exception as e:
        use_trackio = False
        print(f"trackio disabled: {e}")

    best_ever = -np.inf
    pbar = tqdm(range(args.generations), desc="CMA")
    try:
        for gen in pbar:
            if es.stop():
                print("CMA stop:", es.stop())
                break
            solutions = es.ask()
            if args.dream:
                fitnesses = dream_evaluate(solutions, rnn, reward_head, cfg, max_steps, args.avg)
            else:
                fitnesses = evaluate(solutions, envs, vae, rnn, cfg, max_steps, args.avg, args.profile)
            es.tell(solutions, [-f for f in fitnesses])             # CMA minimizes -> negate

            mean, gen_best = float(fitnesses.mean()), float(fitnesses.max())
            if gen_best > best_ever:
                best_ever = gen_best
                ctrl = Controller(cfg)
                vector_to_parameters(
                    torch.tensor(es.result.xbest, dtype=torch.float32), ctrl.parameters()
                )
                torch.save(ctrl.state_dict(), os.path.join(HERE, f"models/controller-{run}.pt"))

            pbar.set_postfix(mean=f"{mean:.1f}", gen_best=f"{gen_best:.1f}", best=f"{best_ever:.1f}")
            if use_trackio:
                trackio.log({
                    "reward/mean": mean,
                    "reward/gen_best": gen_best,
                    "reward/best_ever": best_ever,
                    "reward/min": float(fitnesses.min()),
                    "cma/sigma": float(es.sigma),
                })
    finally:
        if envs is not None:
            envs.close()
        if use_trackio:
            trackio.finish()

    print(f"done. best mean-reward {best_ever:.1f} -> models/controller-{run}.pt")


if __name__ == "__main__":
    main()
