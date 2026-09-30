from __future__ import annotations

import argparse
import numpy as np
import torch
import gymnasium as gym
import gym_pusht  # noqa: F401

from offline_rl_pusht.models.actor import MLPActor
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

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

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")

    returns, max_rewards, successes = [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward, max_reward, success = 0.0, 0.0, False

        for _ in range(args.max_steps):
            state_2d = np.asarray(obs[:2], dtype=np.float32)[None, :]
            state_n = state_norm.transform_np(state_2d)
            pred_n = model(torch.from_numpy(state_n).float().to(device)).cpu().numpy()
            action = action_norm.inverse_np(pred_n)[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            total_reward += float(reward)
            max_reward = max(max_reward, float(reward))
            success = success or bool(info.get("is_success", False))

            if terminated or truncated:
                break

        returns.append(total_reward)
        max_rewards.append(max_reward)
        successes.append(success)

    env.close()

    print("=== State-only BC Simulator Rollout ===")
    print("Deliberately partially observed baseline.")
    print(f"Episodes:        {args.episodes}")
    print(f"Mean return:     {np.mean(returns):.4f}")
    print(f"Mean max reward: {np.mean(max_rewards):.4f}")
    print(f"Success rate:    {np.mean(successes) * 100:.2f}%")


if __name__ == "__main__":
    main()
