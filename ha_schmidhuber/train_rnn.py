"""Train the MDN-RNN (Ha & Schmidhuber World Models).

Script version of train-rnn.ipynb (the notebook is kept for interactive exploration --
diagnostics like the sigma check and image display live there; this is the headless
path for real runs). Loads a frozen VAE trained by train_vae.py.

Run:
    uv run python train_rnn.py                          # full training run
    uv run python train_rnn.py profile=smoke             # fast local smoke test
    uv run python train_rnn.py training.rnn.resume=true  # resume from latest checkpoint
    uv run python train_rnn.py -m training.rnn.lr=1e-3,3e-4   # Hydra multirun sweep
"""
import os

import hydra
import lancedb
import numpy as np
import torch
import trackio
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset, random_split
from tqdm.auto import tqdm

import model
from checkpointing import checkpoint_every, latest_checkpoint, load_checkpoint, save_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))


class FullEpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.lancedb_uri = cfg.dataset.lancedb_uri
        self.img_size = cfg.dataset.img_size
        self.max_episode_steps = cfg.dataset.max_episode_steps
        self._table = None
        self._len = lancedb.connect(cfg.dataset.lancedb_uri).open_table("episodes").count_rows()

    def _get_table(self):
        if self._table is None:
            self._table = lancedb.connect(self.lancedb_uri).open_table("episodes")
        return self._table

    def __getitem__(self, idx):
        episode = self._get_table().search().where(f'episode_id = {idx}').limit(1).to_arrow()
        # Decode straight from the arrow buffers -- .as_py() materialises every pixel as a
        # Python object (~2.4 s per 12 MB episode, measured); the buffer view is ~1 ms.
        obs_list = episode.column('observations')[0].values   # FixedSizeListArray (T, img*img*3)
        act_list = episode.column('actions')[0].values        # FixedSizeListArray (T, 3)
        frames_np = obs_list.values.to_numpy(zero_copy_only=False).reshape(len(obs_list), self.img_size, self.img_size, 3)
        actions_np = act_list.values.to_numpy(zero_copy_only=False).reshape(len(act_list), -1)
        # Truncate to max_episode_steps even if the stored episode is longer (e.g. the
        # smoke profile re-reads a full-length dataset but wants short episodes).
        # .copy() detaches from the read-only arrow buffer (torch needs writable memory).
        frames_np = frames_np[:self.max_episode_steps].copy()
        actions_np = actions_np[:self.max_episode_steps].astype(np.float32)
        # Keep obs uint8 HWC here: float32 CHW would 4x the worker->main IPC, the pinned
        # prefetch buffers, and the PCIe transfer. The GPU does the float conversion.
        return torch.from_numpy(frames_np), torch.from_numpy(actions_np)

    def __len__(self):
        return self._len


def collate_episode_list(batch):
    """Identity collate: keep episodes as a plain list, not stacked (variable length)."""
    return batch


class ZDataset(torch.utils.data.Dataset):
    """Pre-computed VAE mu encodings -- lives entirely in CPU RAM (~38 MB for 1k episodes)."""
    def __init__(self, z_list, a_list):
        self.z = z_list
        self.a = a_list

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.a[idx]


def collate_padded(batch):
    """Pad variable-length episodes; return a boolean mask for valid (non-padded) timesteps."""
    z_list, a_list = zip(*batch)
    T_list = [z.shape[0] - 1 for z in z_list]   # number of valid input/target pairs
    T_max = max(T_list)
    B, z_dim, a_dim = len(batch), z_list[0].shape[-1], a_list[0].shape[-1]

    z_in = torch.zeros(B, T_max, z_dim)
    a_in = torch.zeros(B, T_max, a_dim)
    z_tgt = torch.zeros(B, T_max, z_dim)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, (z, a, T) in enumerate(zip(z_list, a_list, T_list)):
        z_in[i, :T] = z[:-1]
        a_in[i, :T] = a[:T]
        z_tgt[i, :T] = z[1:]
        mask[i, :T] = True

    return z_in, a_in, z_tgt, mask


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
    vae.load_state_dict(torch.load(os.path.join(HERE, cfg.rnn.vae_path), map_location=device, weights_only=True))
    vae.eval()

    rnn = model.RNN(cfg).to(device)

    ds = FullEpisodicDataset(cfg)
    if cfg.dataset.get("max_episodes"):
        ds = Subset(ds, range(min(len(ds), cfg.dataset.max_episodes)))

    # Pre-compute VAE mu for all episodes (once, before training) -- eliminates repeated
    # VAE inference across every epoch. All 1k episodes fit in ~38 MB of RAM.
    #
    # Reading/decoding episodes from lancedb is CPU/IO-bound while VAE encode is GPU-bound,
    # so the two need to run at very different granularities to keep the GPU fed:
    #   - batch_size=episodes_per_batch episodes are pulled per DataLoader item (instead of
    #     1) so several episodes' worth of frames are concatenated into a single big
    #     vae.encode() call -- on a big GPU box, a batch_size=1 loader leaves the GPU idle
    #     between tiny single-episode forward passes (visible as near-0% GPU utilization).
    #   - num_workers is capped at a small, fixed count, not one per core: each persistent
    #     worker buffers prefetch_factor batches of episodes_per_batch full episodes, so
    #     scaling workers with core count on a big box (32+ cores) would multiply that
    #     buffering many times over for no real gain past a handful of workers feeding the
    #     single GPU consumer.
    num_workers_pre = min(8, max(1, (os.cpu_count() or 1) - 1))
    episodes_per_batch = cfg.training.rnn.precompute_batch_episodes
    pre_kwargs = dict(
        batch_size=episodes_per_batch, collate_fn=collate_episode_list,
        num_workers=num_workers_pre, persistent_workers=num_workers_pre > 0,
        # 2 is plenty now that reads are query-bound (~0.1 s/episode); each buffered batch
        # is episodes_per_batch x ~12 MB of uint8, so deeper prefetch just burns RAM.
        prefetch_factor=2 if num_workers_pre > 0 else None,
        multiprocessing_context='spawn' if num_workers_pre > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    z_cache, a_cache = [], []
    with torch.no_grad():
        for episodes in tqdm(DataLoader(ds, **pre_kwargs), desc="Pre-computing z"):
            lens = [min(obs.shape[0], acts.shape[0]) for obs, acts in episodes]
            obs_cat = torch.cat([obs[:T] for (obs, _), T in zip(episodes, lens)]).to(device, non_blocking=True)
            # uint8 HWC -> float CHW in [0,1] on the GPU (cheap there; 4x less PCIe traffic)
            obs_cat = obs_cat.permute(0, 3, 1, 2).float().div_(255)
            z_cat, _, _ = vae.encode(obs_cat)     # one forward pass for the whole batch of episodes
            z_cat = z_cat.cpu()
            offset = 0
            for (_, acts), T in zip(episodes, lens):
                z_cache.append(z_cat[offset:offset + T])
                a_cache.append(acts[:T])
                offset += T

    z_ds = ZDataset(z_cache, a_cache)
    train_z_ds, val_z_ds = random_split(z_ds, [0.9, 0.1])
    print(f"Pre-computed {len(z_cache)} episodes -- {len(train_z_ds)} train / {len(val_z_ds)} val")

    batch_size = 32
    z_kwargs = dict(batch_size=batch_size, collate_fn=collate_padded, num_workers=0)
    train_z_loader = DataLoader(train_z_ds, shuffle=True, **z_kwargs)
    val_z_loader = DataLoader(val_z_ds, shuffle=False, **z_kwargs)

    optimizer = torch.optim.AdamW(rnn.parameters(), cfg.training.rnn.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.rnn.nb_epochs)

    total_steps = cfg.training.rnn.nb_epochs * len(train_z_loader)
    ckpt_interval = checkpoint_every(total_steps, pct=0.01)
    ckpt_dir = os.path.join(HERE, "models", f"rnn-{run}")

    step, start_epoch = 0, 0
    if cfg.training.rnn.get("resume"):
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path:
            state = load_checkpoint(ckpt_path, map_location=device)
            rnn.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, start_epoch = state["step"], state["epoch"]
            print(f"resumed from {ckpt_path} (step {step}, epoch {start_epoch})")

    trackio.init(
        name=f"rnn-{run}", project=cfg.trackio.project, server_url=cfg.trackio.write_url,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    for epoch in range(start_epoch, cfg.training.rnn.nb_epochs):
        rnn.train()
        for z_in, a_in, z_tgt, mask in train_z_loader:
            z_in = z_in.to(device)    # (B, T, 32)
            a_in = a_in.to(device)    # (B, T, 3)
            z_tgt = z_tgt.to(device)  # (B, T, 32)
            mask = mask.to(device)    # (B, T) bool

            pi, mu, sigma, _, rnn_out = rnn(z_in, a_in)
            loss = model.MDN.loss(pi, mu, sigma, z_tgt, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
            optimizer.step()
            trackio.log({"rnn-training-loss": loss.item()})
            step += 1

            if step % ckpt_interval == 0:
                path = save_checkpoint(
                    ckpt_dir, step, keep_last=5,
                    model=rnn.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epoch,
                )
                print(f"checkpoint -> {path}")

        rnn.eval()
        with torch.no_grad():
            for z_in, a_in, z_tgt, mask in val_z_loader:
                z_in = z_in.to(device)
                a_in = a_in.to(device)
                z_tgt = z_tgt.to(device)
                mask = mask.to(device)
                pi, mu, sigma, _, _ = rnn(z_in, a_in)
                loss = model.MDN.loss(pi, mu, sigma, z_tgt, mask)
                trackio.log({"rnn-val-loss": loss.item()})

        scheduler.step()

    trackio.finish()

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    final_path = os.path.join(HERE, "models", f"rnn-{run}.pt")
    torch.save(rnn.state_dict(), final_path)
    print(f"done -> {final_path}")


if __name__ == "__main__":
    main()
