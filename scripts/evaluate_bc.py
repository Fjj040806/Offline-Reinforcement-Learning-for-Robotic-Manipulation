from __future__ import annotations

import argparse
import numpy as np
import torch

from offline_rl_pusht.data.dataset import load_pusht_table, dataset_to_numpy, mask_for_episodes
from offline_rl_pusht.models.actor import MLPActor
from offline_rl_pusht.evaluation.metrics import regression_metrics
from offline_rl_pusht.utils.config import load_config
from offline_rl_pusht.utils.normalizer import Standardizer


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/bc_state.pt")
    parser.add_argument("--config", default="configs/bc.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    mc = checkpoint["model_config"]
    model = MLPActor(
        input_dim=int(mc["input_dim"]),
        hidden_dims=list(mc["hidden_dims"]),
        output_dim=int(mc["output_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state_norm = Standardizer.from_state_dict(checkpoint["state_normalizer"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_normalizer"])

    ds = load_pusht_table(
        repo_id=cfg["dataset"]["repo_id"],
        revision=cfg["dataset"].get("revision"),
    )
    arrays = dataset_to_numpy(ds)

    episode_ids = np.asarray(checkpoint["split_episodes"][args.split], dtype=np.int64)
    mask = mask_for_episodes(arrays["episode_index"], episode_ids)
    states = arrays["state"][mask]
    actions = arrays["action"][mask]

    norm_states = state_norm.transform_np(states)
    norm_actions = action_norm.transform_np(actions)

    pred_norm = model(torch.from_numpy(norm_states).float().to(device)).cpu().numpy()
    pred_actions = action_norm.inverse_np(pred_norm)

    nm = regression_metrics(pred_norm, norm_actions)
    om = regression_metrics(pred_actions, actions)

    print(f"=== {args.split.upper()} BC Evaluation ===")
    print(f"Examples:       {len(states):,}")
    print(f"Normalized MSE: {nm['mse']:.6f}")
    print(f"Action MSE:     {om['mse']:.6f}")
    print(f"Action RMSE:    {om['rmse']:.6f}")
    print(f"Action MAE:     {om['mae']:.6f}")


if __name__ == "__main__":
    main()
