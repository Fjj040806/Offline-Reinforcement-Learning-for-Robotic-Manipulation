# Mathematical objectives and implementation map

These are existing OGBench reference algorithms, not newly proposed methods. Exact executable definitions are in `vendor/impls/agents` and `vendor/impls/utils/datasets.py`.

## Goal-conditioned behavioral cloning

Given state $s$, action $a$ and a sampled future goal $g$ from the same trajectory:

$$L_{BC}(\theta)=-\mathbb E_{(s,a,g)}\log\pi_\theta(a\mid s,g).$$

The fixed-variance Gaussian policy makes this proportional to action MSE up to an additive constant. A low prediction loss does not guarantee closed-loop goal achievement.

## Goal-conditioned implicit Q-learning

Let $\bar Q=\min(\bar Q_1,\bar Q_2)$ denote the target ensemble. Value learning uses asymmetric squared error:

$$L_V=\mathbb E\left[\ell_\tau(\bar Q(s,a,g)-V(s,g))\right],\qquad
\ell_\tau(u)=\begin{cases}\tau u^2&u\geq0\\(1-\tau)u^2&u<0.\end{cases}$$

The critic target is $y=r_g+\gamma m_gV(s',g)$, with the goal-conditioned reward and bootstrap mask supplied by the reference sampler:

$$L_Q=\mathbb E\left[(Q_1(s,a,g)-y)^2+(Q_2(s,a,g)-y)^2\right].$$

This release uses the upstream **DDPG+BC** policy extraction branch, not its optional AWR branch. With $a_\theta$ the clipped policy mean, its actor objective is:

$$L_\pi=-\frac{\mathbb E[\min_iQ_i(s,a_\theta,g)]}
 {\operatorname{stopgrad}(\mathbb E[|\min_iQ_i(s,a_\theta,g)|]+10^{-6})}
 -\alpha\mathbb E\log\pi_\theta(a\mid s,g).$$

The experiments use $\alpha=1$, $\tau=0.9$ and $\gamma=0.99$. Target networks are updated by Polyak averaging. Goal relabeling uses the reference implementation; trajectory endings and task success are distinct concepts.

## Hierarchical implicit Q-learning

HIQL uses a learned goal representation, a high-level subgoal policy and a low-level action policy. Subgoal training targets come from a future state at the configured temporal interval, clipped to trajectory boundaries. High and low actor learning use advantage-weighted log likelihood; weights are clipped as defined in the pinned source.

The core experiment uses interval 10; ablations use 5 and 25. A control step is 0.05 seconds, so these correspond nominally to 0.25 / 0.5 / 1.25 seconds in the collected trajectory. This is a training-target interval, not a statement that the deployed policy holds each subgoal unchanged for that duration.

HIQL and GCIQL have different numbers of components; equal hidden widths do not constitute a parameter-matched ablation of hierarchical structure.

## Evaluation versus training rewards

The offline goal sampler constructs rewards for value learning. Evaluation reports the simulator's physical task success flag. We do not treat a dataset trajectory ending as a successful task, and we do not infer success from low training loss.
