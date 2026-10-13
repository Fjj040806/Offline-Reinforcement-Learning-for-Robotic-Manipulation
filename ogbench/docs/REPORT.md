# Goal-conditioned Offline RL for Multi-step Robotic Manipulation

## Outcome and scope

Completed 18 core runs and 4500 closed-loop evaluation episodes. These are reduced-budget CPU experiments, not a replication of published benchmark performance.

## Question

Can offline goal-conditioned value learning and hierarchical decision making improve manipulation over goal-conditioned behavioral cloning? Does this depend on task complexity, data coverage or subgoal interval? These are hypotheses, not assumed outcomes.

## Data and environment

Official OGBench Cube Single and Cube Double. Each has 1M training transitions and 100k validation transitions. Three training seeds per algorithm. Observations have 28/37 dimensions and include privileged exact object state. Five normalized actions specify relative end-effector xyz/yaw and gripper opening; the environment supplies IK/PD. Four original NPZ files were re-downloaded and matched the SHA256 checks recorded in Phase 1.

## Methods and experimental protocol

Pinned upstream GCBC, GCIQL (DDPG+BC extraction, alpha=1) and HIQL (subgoal interval=10, default low/high temperatures). Networks reduced to two hidden layers of 128, batch 256, 20,000 updates. Final fixed-budget checkpoints; no success-driven checkpoint choice. Training uses offline trajectories; simulator interaction is evaluation only. Unnormalized observations follow upstream conventions. Training loss samples and validation loss samples use official GCDataset/HGCDataset goal relabeling.

GCBC minimizes action negative log likelihood. GCIQL uses expectile value regression, goal-conditioned temporal-difference critic learning, and its configured policy extraction objective. HIQL learns a goal representation and high/low policy hierarchy using advantage-weighted objectives. Upstream algorithms are prior work; the project contribution is the harness, controlled experiment records, analysis and interactive explorer.

Core resets 10000–10049 for each of five fixed evaluation goals, shared across algorithms. Goal success and timeouts come from the environment. This tests held-out resets of fixed goals, not unseen-goal generalization. All core comparisons below use seed-level means and sample SD, not an independent-episode significance test.

## Core results

| Environment | Method | Mean success | Seed SD (pp) | Success / episodes |
|---|---|---:|---:|---:|
| cube-double-play-v0 | GCBC | 0.00% | 0.00 | 0 / 750 |
| cube-double-play-v0 | GCIQL | 0.00% | 0.00 | 0 / 750 |
| cube-double-play-v0 | HIQL | 0.27% | 0.23 | 2 / 750 |
| cube-single-play-v0 | GCBC | 0.27% | 0.46 | 2 / 750 |
| cube-single-play-v0 | GCIQL | 1.47% | 1.22 | 11 / 750 |
| cube-single-play-v0 | HIQL | 6.93% | 1.51 | 52 / 750 |

## Controlled ablations

Cube Double data fractions select complete trajectories with subset seed 42. GCIQL and HIQL use the same subset IDs; reduced subsets do not cut trajectory boundaries. HIQL intervals 5/10/25 are compared. Evaluated with distinct development resets 20000–20009, 10 episodes per goal, 3 training seeds. Full-data interval-10 checkpoints are reused; this table must not be pooled with core test episodes.

| Method | Data | Subgoal interval | Mean success | Seed SD (pp) | Success / episodes |
|---|---:|---:|---:|---:|---:|
| GCIQL | 25% | N/A | 0.67% | 1.15 | 1 / 150 |
| GCIQL | 50% | N/A | 0.00% | 0.00 | 0 / 150 |
| GCIQL | 100% | N/A | 0.00% | 0.00 | 0 / 150 |
| HIQL | 25% | 10 | 0.00% | 0.00 | 0 / 150 |
| HIQL | 50% | 10 | 0.00% | 0.00 | 0 / 150 |
| HIQL | 100% | 5 | 0.00% | 0.00 | 0 / 150 |
| HIQL | 100% | 10 | 0.00% | 0.00 | 0 / 150 |
| HIQL | 100% | 25 | 0.00% | 0.00 | 0 / 150 |

## Interpretation and limits

Success rates are low under this training budget. The evidence does not justify a claim that hierarchical RL is superior, that the policies are converged, or that they generalize to real robots. Small seed differences and infrequent successes should not be ranked as robust improvements. Shared hidden widths do not equal identical total parameter counts; this is not a parameter-matched isolation of hierarchy. Logged times include JIT compilation and evaluation, with concurrent CPU scheduling, and should not be used as an algorithmic speed ranking. Dataset size and interval effects at this budget may also reflect optimization rather than the intrinsic quality of an algorithm. Single/Double differ in both object count and task structure, so task differences cannot be assigned solely to horizon.

Action prediction loss can decrease without successful closed-loop behavior. The earlier PyTorch GCBC pilot achieved 0/15 developer successes; this new study switches to pinned official algorithms and a different network, and is not a paired performance improvement claim over that pilot. The browser displays sampled positions from real evaluation runs, not a live simulator; arm meshes and joint angles are not shown.

## Reproducibility and validation

Configuration files, per-episode records, checkpoints, source versions and data checksums are committed. Verify artifacts with the provided unittest suite. Browser checks cover playback, scrubbing, all environment/algorithm/seed/task selectors, canvas camera controls and downloads on desktop/mobile. Full-budget benchmarking and real hardware validation remain outside this V1 software release.

## References

Park et al., OGBench: Benchmarking Offline Goal-Conditioned RL. https://seohong.me/projects/ogbench/

Pinned source commit a977253ce5b88434f4e4f531c50047cc95993a51. Original MIT license in vendor/OGBENCH-LICENSE.
