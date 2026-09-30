import numpy as np
from offline_rl_pusht.data.dataset import split_episode_ids, prepare_bc_splits


def test_episode_split_no_overlap():
    ids = np.repeat(np.arange(20), 5)
    split = split_episode_ids(ids, seed=123)
    train = set(split.train_episodes.tolist())
    val = set(split.val_episodes.tolist())
    test = set(split.test_episodes.tolist())

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(20))


def test_prepare_bc_splits():
    episode_index = np.repeat(np.arange(10), 10)
    n = len(episode_index)
    rng = np.random.default_rng(0)
    arrays = {
        "state": rng.normal(size=(n, 2)).astype(np.float32),
        "action": rng.normal(size=(n, 2)).astype(np.float32),
        "episode_index": episode_index,
        "frame_index": np.tile(np.arange(10), 10),
        "reward": np.zeros(n, dtype=np.float32),
        "done": np.zeros(n, dtype=bool),
        "success": np.zeros(n, dtype=bool),
    }

    datasets, _, _, state_norm, action_norm = prepare_bc_splits(arrays, seed=1)
    assert len(datasets["train"]) + len(datasets["val"]) + len(datasets["test"]) == n
    assert state_norm.mean.shape == (2,)
    assert action_norm.mean.shape == (2,)
