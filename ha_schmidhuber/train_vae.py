"""Train the VAE (Ha & Schmidhuber World Models).

Script version of train-vae.ipynb (the notebook is kept for interactive exploration --
this is the headless path for real runs and CI/remote machines). Same code runs a full
training run or a laptop smoke test; only the Hydra profile differs.

Run:
    uv run python train_vae.py                          # full training run
    uv run python train_vae.py profile=smoke             # fast local smoke test
    uv run python train_vae.py training.vae.beta=4        # override any config value
    uv run python train_vae.py training.vae.resume=true   # resume from latest checkpoint
    uv run python train_vae.py -m training.vae.beta=0.5,1,4   # Hydra multirun sweep
"""
import multiprocessing
import os

import hydra
import lancedb
import numpy as np
import torch
import torch.nn.functional as F
import trackio
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import model
from checkpointing import checkpoint_every, latest_checkpoint, load_checkpoint, save_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))


class ShuffledFrameDataset(torch.utils.data.IterableDataset):
    """Streams episodes from LanceDB and yields single frames in shuffled order.

    Frames within an episode are highly correlated, so batching whole episodes gives
    poor gradients. Loading every frame in RAM is impossible (10k x 1000 frames ~ 123 GB)
    and reading one ~12 MB episode row per frame would be a 1000x read amplification.
    A shuffle buffer gets both: each episode is read once per pass, its frames are mixed
    into a pool of `buffer_frames` frames spanning ~dozens of episodes, and frames are
    drawn from the pool at random (~250 MB RAM at the default size).
    """

    def __init__(self, cfg, episode_ids, buffer_frames=20_000, seed=0):
        super().__init__()
        self.lancedb_uri = cfg.dataset.lancedb_uri
        self.img_size = cfg.dataset.img_size
        self.max_episode_steps = cfg.dataset.max_episode_steps
        self.episode_ids = np.asarray(episode_ids)
        self.buffer_frames = buffer_frames
        self.seed = seed
        # Estimate: episodes rarely terminate before the step limit, so this is exact
        # for almost every episode. Only used for len(loader) / progress / schedules.
        self.est_frames = len(self.episode_ids) * cfg.dataset.max_episode_steps

    def __len__(self):
        return self.est_frames

    def _episode_frames(self, table, ep_id):
        row = table.search().where(f'episode_id = {ep_id}').limit(1).to_arrow()
        # Decode straight from the arrow buffers -- .as_py() materialises every pixel as a
        # Python object (~2.4 s per 12 MB episode, measured); the buffer view is ~1 ms.
        obs_list = row.column('observations')[0].values   # FixedSizeListArray (T, img*img*3)
        frames = obs_list.values.to_numpy(zero_copy_only=False).reshape(len(obs_list), self.img_size, self.img_size, 3)
        # Truncate to max_episode_steps even if the stored episode is longer (e.g. the
        # smoke profile re-reads a full-length dataset but wants short episodes).
        return frames[:self.max_episode_steps]

    def __iter__(self):
        # _pass survives across epochs inside persistent workers -> fresh order every epoch
        self._pass = getattr(self, "_pass", -1) + 1
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng((self.seed, self._pass, wid))
        ids = self.episode_ids[wid::nw].copy()   # disjoint episodes per worker
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
            buffer[j], buffer[-1] = buffer[-1], buffer[j]   # swap-pop a random frame
            frame = buffer.pop()
            # copy() detaches the 12 KB frame from its 12 MB parent episode array
            yield torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0


def beta_schedule(step, total_steps, target_beta, linear):
    """Constant target_beta, or a linear 0 -> target_beta ramp over the whole run."""
    if linear:
        return target_beta * min(1.0, step / total_steps)
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

    # Split by *episode*, not by frame -- neighbouring frames are near-duplicates, so a
    # frame-level split would leak train data into val.
    n_episodes = lancedb.connect(cfg.dataset.lancedb_uri).open_table("episodes").count_rows()
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
    num_workers = min(6, multiprocessing.cpu_count() - 1)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)

    optimizer = torch.optim.AdamW(vae.parameters(), cfg.training.vae.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.vae.nb_epochs)

    total_steps = cfg.training.vae.nb_epochs * len(train_loader)
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
        name=run, project=cfg.trackio.project, server_url=cfg.trackio.write_url,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    for epoch in tqdm(range(start_epoch, cfg.training.vae.nb_epochs), desc="Epochs"):
        vae.train()
        for x in train_loader:
            x = x.to(device)  # (B, 3, H, W) float32 in [0,1], frames from ~dozens of episodes
            x_recon, kl = vae(x)
            recon = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
            b = beta_schedule(step, total_steps, cfg.training.vae.beta, cfg.training.vae.linear_beta)
            kl_term = torch.clamp(b * kl, min=32) if cfg.training.vae.free_bits else b * kl
            loss = recon + kl_term

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            trackio.log({"training-loss": loss.item(), "training-recon": recon.item(), "training-kl": kl.item()})
            step += 1

            if step % ckpt_interval == 0:
                path = save_checkpoint(
                    ckpt_dir, step, keep_last=5,
                    model=vae.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epoch,
                )
                print(f"checkpoint -> {path}")

        vae.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_recon, kl = vae(x)
                recon = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
                # constant beta at eval (no ramp) -- val measures the final objective
                kl_term = torch.clamp(cfg.training.vae.beta * kl, min=32) if cfg.training.vae.free_bits else cfg.training.vae.beta * kl
                loss = recon + kl_term
                trackio.log({"val-loss": loss.item(), "val-recon": recon.item(), "val-kl": kl.item()})
        scheduler.step()

    trackio.finish()

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    final_path = os.path.join(HERE, "models", f"vae-{run}.pt")
    torch.save(vae.state_dict(), final_path)
    print(f"done -> {final_path}")


if __name__ == "__main__":
    main()
