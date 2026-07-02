import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
from dotenv import load_dotenv
import trackio
import numpy as np
import gymnasium as gym
import torch
import lancedb
from tqdm.auto import tqdm
import pyarrow as pa
import torch.nn.functional as F

with initialize(version_base=None, config_path="conf"):
    load_dotenv()
    write_url = os.environ["TRACKIO_WRITE_URL"]
    cfg = compose(config_name="config", overrides=[f'trackio.write_url="{write_url}"'])
    db = lancedb.connect(cfg.dataset.lancedb_uri)

    IMG_SIZE = cfg.dataset.img_size           # 64
    MAX_STEPS = cfg.dataset.max_episode_steps  # 300

    schema = pa.schema([
        pa.field("episode_id", pa.uint32()),
        pa.field("observations", pa.list_(pa.list_(pa.uint8(), IMG_SIZE * IMG_SIZE * 3))),
        pa.field("actions", pa.list_(pa.list_(pa.float32(), 3)))
    ])

    tbl = db.create_table("episodes", pa.Table.from_batches([], schema=schema), schema=schema, mode="overwrite")


NUM_ENVS = os.cpu_count()
envs = gym.make_vec(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True,
    num_envs=NUM_ENVS,
    max_episode_steps=MAX_STEPS,
    vectorization_mode="async"
)

def batch_resize(obs_np):
    """(N, H, W, 3) uint8 → (N, IMG_SIZE, IMG_SIZE, 3) uint8 via a single vectorised call."""
    t = torch.from_numpy(obs_np).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).byte().numpy()


def flush_to_db(buffer, tbl, schema, start_id):
    """Write a list of (obs_array, act_array) episodes to LanceDB in one batch."""
    n = len(buffer)
    ids = pa.array(range(start_id, start_id + n), type=pa.uint32())
    all_obs, all_acts, offsets = [], [], [0]
    for obs_arr, act_arr in buffer:
        all_obs.append(obs_arr.reshape(-1))
        all_acts.append(act_arr.reshape(-1))
        offsets.append(offsets[-1] + len(obs_arr))
    flat_obs = pa.FixedSizeListArray.from_arrays(
        pa.array(np.concatenate(all_obs), type=pa.uint8()), IMG_SIZE * IMG_SIZE * 3
    )
    flat_act = pa.FixedSizeListArray.from_arrays(
        pa.array(np.concatenate(all_acts), type=pa.float32()), 3
    )
    tbl.add(pa.Table.from_arrays(
        [ids, pa.ListArray.from_arrays(offsets, flat_obs), pa.ListArray.from_arrays(offsets, flat_act)],
        schema=schema,
    ))


# ── Random policy ──────────────────────────────────────────────────────────────
# i.i.d. action_space.sample() at 50 Hz averages to zero net steering with gas 0.5
# and brake 0.5 — the car crawls near the start line and the dataset never sees
# corners at speed. Instead: hold each random action for 1..ACTION_REPEAT frames
# and only rarely touch the brake.
ACTION_REPEAT = 8
BRAKE_PROB = 0.1

def sample_action(rng):
    return np.array([
        rng.uniform(-1.0, 1.0),                                       # steer
        rng.uniform(0.0, 1.0),                                        # gas
        rng.uniform(0.0, 1.0) if rng.random() < BRAKE_PROB else 0.0,  # brake
    ], dtype=np.float32)


WRITE_BATCH = 10   # flush every N completed episodes (~12 MB/episode at 1000 steps)

rng = np.random.default_rng(0)
current_obs = [None] * NUM_ENVS
env_obs  = [[] for _ in range(NUM_ENVS)]
env_acts = [[] for _ in range(NUM_ENVS)]
held_action = np.stack([sample_action(rng) for _ in range(NUM_ENVS)])
hold_left = rng.integers(1, ACTION_REPEAT + 1, size=NUM_ENVS)
# Gymnasium 1.x vector envs default to NEXT_STEP autoreset: when done[i] is True the
# returned obs is the *final* frame, and the following step() ignores the action and
# returns the reset obs. skip[i] marks that phantom reset step so nothing is recorded.
skip = np.zeros(NUM_ENVS, dtype=bool)
write_buffer = []
total_collected = 0
next_ep_id = 0

raw_obs, _ = envs.reset()
resized = batch_resize(raw_obs)
for i in range(NUM_ENVS):
    current_obs[i] = resized[i]

pbar = tqdm(total=cfg.dataset.episodes)
while total_collected < cfg.dataset.episodes:
    for i in range(NUM_ENVS):
        if not skip[i]:
            env_obs[i].append(current_obs[i])
        if hold_left[i] == 0:
            held_action[i] = sample_action(rng)
            hold_left[i] = rng.integers(1, ACTION_REPEAT + 1)
        hold_left[i] -= 1

    raw_obs, _, term, trunc, _ = envs.step(held_action)
    resized = batch_resize(raw_obs)
    done = term | trunc

    for i in range(NUM_ENVS):
        if skip[i]:
            current_obs[i] = resized[i]   # true initial obs of the new episode
            skip[i] = False
            continue

        env_acts[i].append(held_action[i].copy())   # copy: held_action mutates in place
        current_obs[i] = resized[i]

        if done[i]:
            write_buffer.append((
                np.array(env_obs[i], dtype=np.uint8),    # (T, 64, 64, 3)
                np.array(env_acts[i], dtype=np.float32)  # (T, 3)
            ))
            env_obs[i], env_acts[i] = [], []
            skip[i] = True     # next step() call resets env i — record nothing for it
            hold_left[i] = 0   # new episode starts with a freshly sampled action
            total_collected += 1
            pbar.update(1)

            if len(write_buffer) >= WRITE_BATCH:
                flush_to_db(write_buffer, tbl, schema, next_ep_id)
                next_ep_id += len(write_buffer)
                write_buffer = []

            if total_collected >= cfg.dataset.episodes:
                break

if write_buffer:
    flush_to_db(write_buffer, tbl, schema, next_ep_id)

pbar.close()
envs.close()