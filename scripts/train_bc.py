from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from offline_rl_pusht.data.dataset import load_pusht_table, dataset_to_numpy, prepare_bc_splits
from offline_rl_pusht.models.actor import MLPActor
from offline_rl_pusht.training.bc import train_one_epoch, evaluate_loss
from offline_rl_pusht.utils.config import load_config
from offline_rl_pusht.utils.seed import set_seed


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bc.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["training"]["device"])
    print(f"Device: {device}")

    ds = load_pusht_table(
        repo_id=cfg["dataset"]["repo_id"],
        revision=cfg["dataset"].get("revision"),
    )
    arrays = dataset_to_numpy(ds)

    datasets, splits, _, state_norm, action_norm = prepare_bc_splits(
        arrays,
        train_fraction=float(cfg["dataset"]["train_fraction"]),
        val_fraction=float(cfg["dataset"]["val_fraction"]),
        test_fraction=float(cfg["dataset"]["test_fraction"]),
        seed=int(cfg["seed"]),
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
    )

    model = MLPActor(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        output_dim=int(cfg["model"]["output_dim"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    checkpoint_path = Path(cfg["output"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "version": "0.1.0",
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg["model"],
                    "state_normalizer": state_norm.state_dict(),
                    "action_normalizer": action_norm.state_dict(),
                    "split_episodes": {
                        "train": splits.train_episodes.tolist(),
                        "val": splits.val_episodes.tolist(),
                        "test": splits.test_episodes.tolist(),
                    },
                    "best_val_loss": best_val,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:03d} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | best={best_val:.6f}"
        )

    history_path = Path(cfg["output"]["history"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
