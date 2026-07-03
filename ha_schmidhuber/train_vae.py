import multiprocessing
import os

import hydra
import lancedb
import numpy as np
import torch
import torch.nn.functional as F
import trackio
from dotenv import load_dotenv
from huggingface_hub.hf_api import WebhookWatchedItem
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory
from tqdm.auto import tqdm

import model
from checkpointing import (
    checkpoint_every,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# A latent dim whose mean posterior sigma exceeds this is ~indistinguishable from the
# N(0,1) prior -- the encoder has stopped encoding information into it (posterior collapse).
DEAD_SIGMA = 0.9


class ShuffledFrameDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg, episode_ids, buffer_frames=20_000, seed=0):
        super().__init__()
        self.lancedb_uri = cfg.dataset.lancedb_uri
        self.img_size = cfg.dataset.img_size
        self.max_episode_steps = cfg.dataset.max_episode_steps
        self.episode_ids = np.asarray(episode_ids)
        self.buffer_frames = buffer_frames
        self.seed = seed
        self.device = cfg.device
        # Estimate: episodes rarely terminate before the step limit, so this is exact
        # for almost every episode. Only used for len(loader) / progress / schedules.
        self.est_frames = len(self.episode_ids) * cfg.dataset.max_episode_steps

    def __len__(self):
        return self.est_frames

    def _episode_frames(self, table, ep_id):
        row = table.search().where(f"episode_id = {ep_id}").limit(1).to_arrow()
        # Decode straight from the arrow buffers -- .as_py() materialises every pixel as a
        # Python object (~2.4 s per 12 MB episode, measured); the buffer view is ~1 ms.
        obs_list = row.column("observations")[
            0
        ].values  # FixedSizeListArray (T, img*img*3)
        frames = obs_list.values.to_numpy(zero_copy_only=False).reshape(
            len(obs_list), self.img_size, self.img_size, 3
        )
        # Truncate to max_episode_steps even if the stored episode is longer (e.g. the
        # smoke profile re-reads a full-length dataset but wants short episodes).
        return frames[: self.max_episode_steps]

    def __iter__(self):
        # _pass survives across epochs inside persistent workers -> fresh order every epoch
        self._pass = getattr(self, "_pass", -1) + 1
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng((self.seed, self._pass, wid))
        ids = self.episode_ids[wid::nw].copy()  # disjoint episodes per worker
        rng.shuffle(ids)
        table = lancedb.connect(self.lancedb_uri).open_table("episodes")

        buffer, it, exhausted = [], iter(ids), False
        while True:
            while not exhausted and len(buffer) < self.buffer_frames:
                ep = next(it, None)
                if ep is None:
                    exhausted = True
                else:
                    buffer.extend(self._episode_frames(table, int(ep)))
            if not buffer:
                return
            j = rng.integers(len(buffer))
            buffer[j], buffer[-1] = buffer[-1], buffer[j]  # swap-pop a random frame
            frame = buffer.pop()
            # copy() detaches the 12 KB frame from its 12 MB parent episode array
            yield (
                torch.from_numpy(frame.copy()).to(self.device).permute(2, 0, 1).float()
                / 255.0
            )


def beta_schedule(step, total_steps, target_beta, schedule):
    """Constant target_beta, or a linear 0 -> target_beta ramp over the whole run."""
    if schedule == "linear":
        return target_beta * min(1.0, step / total_steps)
    elif schedule == "tanh":
        return target_beta * torch.tanh(2 * step / total_steps)
    elif schedule == "sigmoid":
        return target_beta * torch.sigmoid(12 * step / total_steps - 6)
    return target_beta


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv(os.path.join(HERE, ".env"))
    url = os.environ.get("TRACKIO_WRITE_URL")
    if url:
        cfg.trackio.write_url = url

    run = cfg.trackio.run_name
    device = cfg.device
    torch.manual_seed(0)

    vae = model.AutoEncoder(cfg).to(device)
    vae = torch.compile(vae)

    # Split by *episode*, not by frame -- neighbouring frames are near-duplicates, so a
    # frame-level split would leak train data into val.
    n_episodes = (
        lancedb.connect(cfg.dataset.lancedb_uri).open_table("episodes").count_rows()
    )
    if cfg.dataset.get("max_episodes"):
        n_episodes = min(n_episodes, cfg.dataset.max_episodes)
    perm = np.random.default_rng(0).permutation(n_episodes)
    n_val = max(1, int(0.1 * n_episodes))
    val_ids, train_ids = perm[:n_val], perm[n_val:]

    train_ds = ShuffledFrameDataset(cfg, train_ids, seed=0)
    val_ds = ShuffledFrameDataset(cfg, val_ids, seed=1)
    print(f"{n_episodes} episodes -- {len(train_ids)} train / {len(val_ids)} val")

    # IterableDataset: no shuffle arg (order comes from the shuffle buffer), workers get
    # disjoint episode shards. batch_size is now *frames* per step, not episodes.
    batch_size = cfg.training.vae.batch_size
    num_workers = cfg.num_workers
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=True if device == "cuda" else False,
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)

    optimizer = torch.optim.AdamW(vae.parameters(), cfg.training.vae.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.training.vae.nb_epochs
    )

    total_steps = cfg.training.vae.nb_epochs * len(train_loader)
    fb = 32 * cfg.training.vae.get("lambfa")  # KL floor in nats when free_bits is on
    ckpt_interval = checkpoint_every(total_steps, pct=0.01)
    ckpt_dir = os.path.join(HERE, "models", f"vae-{run}")

    step, start_epoch = 0, 0
    if cfg.training.vae.get("resume"):
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path:
            state = load_checkpoint(ckpt_path, map_location=device)
            vae.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, start_epoch = state["step"], state["epoch"]
            print(f"resumed from {ckpt_path} (step {step}, epoch {start_epoch})")

    trackio.init(
        name=run,
        project=cfg.trackio.project + "-vae",
        server_url=cfg.trackio.write_url,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow" if cfg.training.vae.get("resume") else "never",
        webhook_url=os.environ.get("TRACKIO_WEBHOOK_URL"),
    )

    n_dead = 32
    for epoch in tqdm(range(start_epoch, cfg.training.vae.nb_epochs), desc="Epochs"):
        vae.train()
        for x in train_loader:
            # (B, 3, H, W) float32 in [0,1], frames from ~dozens of episodes
            x_recon, kl = vae(x)
            recon = F.mse_loss(x_recon, x, reduction="sum") / x.shape[0]
            b = beta_schedule(
                step, total_steps, cfg.training.vae.beta, cfg.training.vae.beta_schedule
            )
            kl_term = (
                torch.clamp(b * kl, min=fb) if cfg.training.vae.free_bits else b * kl
            )
            loss = recon + kl_term

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            trackio.log(
                {
                    "training-loss": loss.item(),
                    "training-recon": recon.item(),
                    "training-kl": kl.item(),
                }
            )
            step += 1

            if step % ckpt_interval == 0:
                path = save_checkpoint(
                    ckpt_dir,
                    step,
                    keep_last=5,
                    model=vae.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                )
                print(f"checkpoint -> {path}")

        vae.eval()
        # Posterior-collapse diagnostics: accumulate per-dim posterior stats over the val set.
        z_dim = vae.dense.mu.out_features
        sig_sum = torch.zeros(z_dim, device=device)
        mu_sum = torch.zeros(z_dim, device=device)
        mu_sq_sum = torch.zeros(z_dim, device=device)
        kl_dim_sum = torch.zeros(z_dim, device=device)
        n_frames = 0
        with torch.no_grad():
            for x in val_loader:
                x = x
                z, mu, log_sigma = vae.encode(x)
                x_recon = vae.decoder(z)
                recon = 0.5 * F.mse_loss(x_recon, x, reduction="sum") / x.shape[0]
                kl = model.AutoEncoder.kl_divergence(mu, log_sigma)
                # constant beta at eval (no ramp) -- val measures the final objective
                kl_term = (
                    torch.clamp(cfg.training.vae.beta * kl, min=fb)
                    if cfg.training.vae.free_bits
                    else cfg.training.vae.beta * kl
                )
                loss = recon + kl_term
                trackio.log(
                    {
                        "val-loss": loss.item(),
                        "val-recon": recon.item(),
                        "val-kl": kl.item(),
                    }
                )

                sigma = log_sigma.exp()
                # standard per-dim Gaussian KL(N(mu,sigma^2)||N(0,1)) -- a collapse measure
                # independent of the (non-standard) training objective above.
                kl_dim = -log_sigma - 0.5 + 0.5 * mu.pow(2) + 0.5 * sigma.pow(2)
                sig_sum += sigma.sum(0)
                mu_sum += mu.sum(0)
                mu_sq_sum += mu.pow(2).sum(0)
                kl_dim_sum += kl_dim.sum(0)
                n_frames += x.shape[0]

        if n_frames:
            sigma_pd = sig_sum / n_frames  # mean posterior sigma per dim
            kl_pd = kl_dim_sum / n_frames  # mean standard KL per dim
            mu_var_pd = mu_sq_sum / n_frames - (mu_sum / n_frames).pow(
                2
            )  # activity (Var_x E[z|x])
            dead = sigma_pd > DEAD_SIGMA  # posterior ~ prior => collapsed
            n_dead = int(dead.sum())
            trackio.log(
                {
                    "posterior/mean_sigma": sigma_pd.mean().item(),
                    "posterior/min_sigma": sigma_pd.min().item(),
                    "posterior/max_sigma": sigma_pd.max().item(),
                    "posterior/dead_dims": n_dead,
                    "posterior/active_dims": z_dim - n_dead,
                    "posterior/frac_dead": n_dead / z_dim,
                    "posterior/mean_kl_per_dim": kl_pd.mean().item(),
                    "posterior/mean_mu_var": mu_var_pd.mean().item(),
                }
            )
            print(
                f"[epoch {epoch}] posterior: {n_dead}/{z_dim} dead dims (sigma>{DEAD_SIGMA}), "
                f"mean sigma {sigma_pd.mean():.3f}, mean KL/dim {kl_pd.mean():.3f}"
            )
            print(
                "  sigma/dim:",
                np.array2string(
                    sigma_pd.cpu().numpy(),
                    precision=3,
                    suppress_small=True,
                    max_line_width=100,
                ),
            )
        scheduler.step()

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    final_path = os.path.join(HERE, "models", f"vae-{run}.pt")
    torch.save(vae.state_dict(), final_path)
    # artifact = trackio.Artifact(
    #     name=f"vae-{run}",
    #     type="model",
    #     description="Variational Auto Encoder",
    #     metadata=OmegaConf.to_container(cfg),
    # )
    # artifact.add_file(final_path)
    # trackio.log_artifact(artifact)
    trackio.alert(
        title="VAE trained!",
        text=f"Dead dims: {n_dead}",
        level=trackio.AlertLevel.INFO,
    )
    trackio.finish()
    return


if __name__ == "__main__":
    main()
