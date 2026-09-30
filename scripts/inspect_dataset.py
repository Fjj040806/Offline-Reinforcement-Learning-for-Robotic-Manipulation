from collections import Counter
import numpy as np

from offline_rl_pusht.data.dataset import (
    load_pusht_table, dataset_to_numpy, prepare_bc_splits
)


def main():
    ds = load_pusht_table()
    arr = dataset_to_numpy(ds)

    episode_counts = Counter(arr["episode_index"].tolist())
    lengths = np.asarray(list(episode_counts.values()))
    datasets, splits, _, _, _ = prepare_bc_splits(arr)

    print("=== PushT V0.1 Dataset Inspection ===")
    print(f"Frames: {len(ds):,}")
    print(f"Episodes: {len(episode_counts):,}")
    print(f"State shape: {arr['state'].shape}")
    print(f"Action shape: {arr['action'].shape}")
    print(
        "Episode length: "
        f"min={lengths.min()}, mean={lengths.mean():.1f}, "
        f"median={np.median(lengths):.1f}, max={lengths.max()}"
    )
    print(f"Reward mean={arr['reward'].mean():.4f}, max={arr['reward'].max():.4f}")
    print(f"Frames flagged success: {arr['success'].sum():,}")
    print()
    print("Episode-level split:")
    print(f"  train: {len(splits.train_episodes):,} episodes | {len(datasets['train']):,} frames")
    print(f"  val:   {len(splits.val_episodes):,} episodes | {len(datasets['val']):,} frames")
    print(f"  test:  {len(splits.test_episodes):,} episodes | {len(datasets['test']):,} frames")


if __name__ == "__main__":
    main()
