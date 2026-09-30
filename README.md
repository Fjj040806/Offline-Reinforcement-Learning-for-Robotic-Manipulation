# Offline Reinforcement Learning for Robotic Manipulation — PushT

> **Project V0.1** — data pipeline + exploratory analysis + state-only Behavior Cloning baseline.

This repository studies offline policy learning on the **PushT** robotic manipulation benchmark using the Hugging Face `lerobot/pusht` dataset.

The long-term project compares **Behavior Cloning (BC)** with **Implicit Q-Learning (IQL)** and later adds visual state representations. V0.1 intentionally starts with a small, auditable baseline so every later improvement has a clean comparison point.

## Why PushT?

PushT is a continuous-control manipulation task. A circular end-effector must push a T-shaped block into a target region.

The Hugging Face dataset contains:
- 2-D robot/end-effector position (`observation.state`)
- 2-D continuous action (`action`)
- rewards, done flags, success flags
- episode/frame indices
- RGB observations stored as video

**Important limitation:** the dataset's 2-D `observation.state` does **not** contain the T-block pose. Therefore the V0.1 state-only BC policy is deliberately **partially observed**. This is useful as a baseline, not as the final solution.

## V0.1 goals

- [x] Reproducible project structure
- [x] Load `lerobot/pusht` from Hugging Face
- [x] Split **by episode** to avoid trajectory leakage
- [x] Compute train-only normalization statistics
- [x] Dataset EDA utilities
- [x] Train a state-only BC policy
- [x] Evaluate validation/test MSE / MAE
- [x] Optional PushT simulator rollout for the BC baseline
- [x] Unit tests that do not require downloading the dataset
- [ ] Visual encoder
- [ ] IQL actor / critic / value networks
- [ ] Offline RL experiments
- [ ] Dataset-quality / expectile ablations

## Roadmap

```text
V0.1  Hugging Face data -> EDA -> state-only BC -> simulator baseline
V0.2  pixel/visual representation -> image-conditioned BC
V0.3  IQL from scratch -> BC vs IQL
V0.4  dataset-quality + dataset-size + expectile ablations
V1.0  polished report, GIFs, reproducible benchmark
```

## Repository layout

```text
offline-rl-pusht-v0.1/
├── configs/
│   └── bc.yaml
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── scripts/
│   ├── inspect_dataset.py
│   ├── train_bc.py
│   ├── evaluate_bc.py
│   └── rollout_bc.py
├── src/offline_rl_pusht/
│   ├── data/dataset.py
│   ├── models/actor.py
│   ├── training/bc.py
│   ├── evaluation/metrics.py
│   └── utils/
└── tests/
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -e ".[dev]"
```

## 1. Inspect the dataset

```bash
python scripts/inspect_dataset.py
```

The script prints dataset size, episode count, episode-length statistics, reward statistics and split sizes.

## 2. Run EDA

```bash
jupyter lab notebooks/01_dataset_exploration.ipynb
```

The notebook visualizes episode lengths, rewards, actions, end-effector trajectories, and episode outcomes.

## 3. Train Behavior Cloning

```bash
python scripts/train_bc.py --config configs/bc.yaml
```

Policy:

```text
2-D agent position
      ↓
MLP(2 -> 128 -> 128 -> 2)
      ↓
2-D target action
```

Objective:

\[
\mathcal{L}_{BC}
=
\frac{1}{N}\sum_i\|\pi_\theta(s_i)-a_i\|_2^2
\]

Outputs:
- `checkpoints/bc_state.pt`
- `results/bc_history.json`

## 4. Evaluate offline

```bash
python scripts/evaluate_bc.py \
  --checkpoint checkpoints/bc_state.pt \
  --config configs/bc.yaml
```

Metrics:
- normalized action MSE
- original-scale action MSE
- original-scale action RMSE
- original-scale action MAE

## 5. Optional simulator rollout

```bash
python scripts/rollout_bc.py \
  --checkpoint checkpoints/bc_state.pt \
  --episodes 20
```

The simulator exposes full state:

```text
[agent_x, agent_y, block_x, block_y, block_angle]
```

V0.1 intentionally feeds only `[agent_x, agent_y]` to the policy so that it matches the offline dataset baseline. Weak rollout performance is therefore an informative baseline rather than a bug.

## Experimental discipline

### Split by episode, not frame

Frames from the same trajectory are highly correlated. Frame-level random splitting leaks temporal neighbors across train and validation sets.

V0.1 uses episode-level splits:

```text
80% train / 10% validation / 10% test
```

### Normalize from training episodes only

State and action statistics are fit only on the training split and reused for validation/test.

### Reproducibility

Python, NumPy and PyTorch are seeded.

## Research questions for later versions

1. Can IQL outperform pure BC on PushT?
2. How much does visual information improve over the 2-D partially observed baseline?
3. How does demonstration quality affect BC versus IQL?
4. How sensitive is IQL to the expectile parameter?
5. How does performance scale with offline dataset size?

## Suggested V0.2

```text
96 x 96 RGB -> CNN encoder -> latent
                              +
                       2-D agent position
                              ↓
                         policy MLP
                              ↓
                         2-D action
```

This directly addresses partial observability because the image contains the block and target geometry.

## License

MIT.
