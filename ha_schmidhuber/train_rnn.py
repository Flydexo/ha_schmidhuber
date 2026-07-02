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
import multiprocessing
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
        frames_np = np.array(episode.column('observations').combine_chunks()[0].as_py(), dtype=np.uint8).reshape(-1, self.img_size, self.img_size, 3)
        actions_np = np.array(episode.column('actions').combine_chunks()[0].as_py(), dtype=np.float32)
        # Truncate to max_episode_steps even if the stored episode is longer (e.g. the
        # smoke profile re-reads a full-length dataset but wants short episodes).
        frames_np = frames_np[:self.max_episode_steps]
        actions_np = actions_np[:self.max_episode_steps]
        obs = torch.from_numpy(frames_np).clone().permute(0, 3, 1, 2).to(torch.float32) / 255
        acts = torch.from_numpy(actions_np).clone()
        return obs, acts

    def __len__(self):
        return self._len


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
    vae.load_state_dict(torch.load(os.path.join(HERE, "models", f"vae-{run}.pt"), map_location=device, weights_only=True))
    vae.eval()

    rnn = model.RNN(cfg).to(device)

    ds = FullEpisodicDataset(cfg)
    if cfg.dataset.get("max_episodes"):
        ds = Subset(ds, range(min(len(ds), cfg.dataset.max_episodes)))

    # Pre-compute VAE mu for all episodes (once, before training) -- eliminates repeated
    # VAE inference across every epoch. All 1k episodes fit in ~38 MB of RAM.
    num_workers_pre = min(6, multiprocessing.cpu_count() - 1)
    pre_kwargs = dict(
        batch_size=1, collate_fn=lambda b: b[0],
        num_workers=num_workers_pre, persistent_workers=num_workers_pre > 0,
        prefetch_factor=4 if num_workers_pre > 0 else None,
        multiprocessing_context='fork' if num_workers_pre > 0 else None,
    )

    z_cache, a_cache = [], []
    with torch.no_grad():
        for obs, acts in tqdm(DataLoader(ds, **pre_kwargs), desc="Pre-computing z"):
            T = min(obs.shape[0], acts.shape[0])
            z, _, _ = vae.encode(obs[:T].to(device))
            z_cache.append(z.cpu())
            a_cache.append(acts[:T].cpu())

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
