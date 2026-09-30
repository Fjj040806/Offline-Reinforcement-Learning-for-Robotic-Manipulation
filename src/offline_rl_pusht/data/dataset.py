from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from offline_rl_pusht.utils.normalizer import Standardizer


REQUIRED_COLUMNS = [
    "observation.state",
    "action",
    "episode_index",
    "frame_index",
    "next.reward",
    "next.done",
    "next.success",
]


@dataclass
class SplitIndices:
    train_episodes: np.ndarray
    val_episodes: np.ndarray
    test_episodes: np.ndarray


def split_episode_ids(
    episode_ids: Iterable[int],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> SplitIndices:
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0.")

    episodes = np.asarray(sorted(set(int(x) for x in episode_ids)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)

    n = len(episodes)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))

    return SplitIndices(
        episodes[:n_train],
        episodes[n_train:n_train + n_val],
        episodes[n_train + n_val:],
    )


def load_pusht_table(
    repo_id: str = "lerobot/pusht",
    revision: str | None = "v1.3",
):
    ds = load_dataset(repo_id, split="train", revision=revision)
    missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
    if missing:
        raise KeyError(
            f"Dataset is missing expected columns: {missing}. "
            f"Available columns: {ds.column_names}"
        )
    return ds.select_columns(REQUIRED_COLUMNS)


def dataset_to_numpy(ds) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray(ds["observation.state"], dtype=np.float32),
        "action": np.asarray(ds["action"], dtype=np.float32),
        "episode_index": np.asarray(ds["episode_index"], dtype=np.int64),
        "frame_index": np.asarray(ds["frame_index"], dtype=np.int64),
        "reward": np.asarray(ds["next.reward"], dtype=np.float32),
        "done": np.asarray(ds["next.done"], dtype=bool),
        "success": np.asarray(ds["next.success"], dtype=bool),
    }


def mask_for_episodes(episode_index: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    return np.isin(episode_index, episode_ids)


class BCDataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        state_normalizer: Standardizer,
        action_normalizer: Standardizer,
    ):
        if len(states) != len(actions):
            raise ValueError("states and actions must have equal length.")

        self.states = torch.from_numpy(
            state_normalizer.transform_np(states).astype(np.float32)
        )
        self.actions = torch.from_numpy(
            action_normalizer.transform_np(actions).astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]


def prepare_bc_splits(
    arrays: dict[str, np.ndarray],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
):
    splits = split_episode_ids(
        arrays["episode_index"],
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    masks = {
        "train": mask_for_episodes(arrays["episode_index"], splits.train_episodes),
        "val": mask_for_episodes(arrays["episode_index"], splits.val_episodes),
        "test": mask_for_episodes(arrays["episode_index"], splits.test_episodes),
    }

    train_states = arrays["state"][masks["train"]]
    train_actions = arrays["action"][masks["train"]]

    state_normalizer = Standardizer.fit(train_states)
    action_normalizer = Standardizer.fit(train_actions)

    datasets = {}
    for name, mask in masks.items():
        datasets[name] = BCDataset(
            arrays["state"][mask],
            arrays["action"][mask],
            state_normalizer,
            action_normalizer,
        )

    return datasets, splits, masks, state_normalizer, action_normalizer
