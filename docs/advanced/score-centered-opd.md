---
title: Score-Centered On-Policy Distillation
description: Derivation of the pure-distillation loss used by `--use-opd --use-score-centering`, with a stale sampler refreshed every K steps.
---

This page derives the loss that Miles optimizes for score-centered on-policy
distillation (SC-OPD). The loss is **pure distillation**: the only learning signal is
the reverse KL to a fixed teacher; no task reward enters the advantage. The boxed math
grader is used for evaluation and is logged on training rollouts for monitoring only.
The page fixes the notation once, derives score centering for a generic per-token
advantage, derives the distillation term, states the loss and its expected update, and
maps every symbol to the code and the launch flags.

## 1. Notation

| Symbol | Meaning |
|---|---|
| $x$ | prompt, $x \sim \mathcal{D}$ |
| $y = (y_1, \dots, y_T)$ | response, sampled autoregressively from the **sampler** |
| $h_t = (x, y_{<t})$ | prefix at position $t$ |
| $v$ | a candidate vocabulary token at position $t$ |
| $q(\cdot \mid h_t)$ | **sampler** next-token distribution (the inference engine; may be stale or quantized) |
| $p_\theta(\cdot \mid h_t)$ | **trainer** next-token distribution (the weights being updated) |
| $T(\cdot \mid h_t)$ | **teacher** next-token distribution (fixed) |
| $s_v(h_t) = \nabla_\theta \log p_\theta(v \mid h_t)$ | score of token $v$ at prefix $h_t$ |
| $s_t = s_{y_t}(h_t)$ | score of the token actually sampled |
| $\bar{s}(h_t) = \sum_v q(v \mid h_t)\, s_v(h_t)$ | expected score **under the sampler** at $h_t$ |
| $A_t$ | a generic per-token advantage (Sections 2–3); in this recipe $A_t = -\beta\, d_t$ |
| $d_v(h_t) = \log p_\theta(v \mid h_t) - \log T(v \mid h_t)$ | student–teacher log-ratio of token $v$ |
| $d_t = d_{y_t}(h_t)$ | log-ratio at the sampled token |
| $\beta$ | distillation coefficient (`--opd-kl-coef`) |
| $\mathrm{sg}[\cdot]$ | stop-gradient |
| $K$ | sampler refresh interval in rollout steps (`--update-weights-interval`) |

Two identities are used repeatedly. For any distribution $p_\theta$,

$$
\mathbb{E}_{v \sim p_\theta}[s_v] = \sum_v p_\theta(v)\, \nabla_\theta \log p_\theta(v) = \nabla_\theta \sum_v p_\theta(v) = 0,
\tag{1}
$$

and for any random variables $U, W$ under a distribution $\mu$,

$$
\mathbb{E}_\mu[U W] = \mathbb{E}_\mu[U]\, \mathbb{E}_\mu[W] + \mathrm{Cov}_\mu(U, W).
\tag{2}
$$

The sampler is off-policy whenever $q \neq p_\theta$. In Miles this happens through
train–inference mismatch (kernels, quantization) and, deliberately, through
**staleness**: with `--update-weights-interval K` the sampler keeps the weights from
the last sync for $K$ rollout steps, so $q$ lags $p_\theta$ by up to $K$ optimizer
steps.

## 2. Policy gradient under an off-policy sampler

Score centering is a property of the policy-gradient estimator for *any* per-token
advantage, so this section derives it generically; Section 4 specializes the
advantage to the distillation term.

### 2.1 The update, indexed by prefix

The sequence log-probability factorizes, so its score is a sum of token scores:

$$
\nabla_\theta \log p_\theta(y \mid x) = \sum_{t=1}^{T} s_t .
$$

A policy-gradient update with per-token advantages $A_t$ is a sum over positions:

$$
g = \mathbb{E}_{x}\, \mathbb{E}_{y \sim q}\Big[ \sum_t A_t\, s_t \Big]
  = \mathbb{E}_{x} \sum_t \mathbb{E}_{y \sim q}\big[ A_t\, s_t \big].
\tag{3}
$$

To analyze one summand, integrate out everything after position $t$. Define the
advantage of choosing $v$ next, averaged over the continuation, and its prefix mean:

$$
Q_t(h_t, v) = \mathbb{E}\big[A_t \mid h_t, y_t = v\big],
\qquad
V_t(h_t) = \mathbb{E}_{v \sim q(\cdot \mid h_t)}\big[Q_t(h_t, v)\big].
\tag{4}
$$

Since $s_t$ depends only on $(h_t, y_t)$,

$$
\mathbb{E}_{y \sim q}[A_t\, s_t]
= \mathbb{E}_{h_t \sim q}\; \mathbb{E}_{v \sim q(\cdot \mid h_t)}\big[ Q_t(h_t, v)\, s_v(h_t) \big].
\tag{5}
$$

### 2.2 Drift and signal

Apply (2) to the inner expectation of (5) at a fixed prefix:

$$
\mathbb{E}_{v \sim q}\big[Q_t\, s_v\big]
= \underbrace{V_t(h_t)\, \bar{s}(h_t)}_{\text{drift}}
\;+\;
\underbrace{\mathrm{Cov}_{v \sim q}\big(Q_t(h_t, \cdot),\, s_\cdot(h_t)\big)}_{\text{signal}}.
\tag{6}
$$

The **signal** is the only term that depends on which token leads to which
advantage. The **drift** depends on the advantages only through their prefix mean
$V_t$; its direction $\bar{s}(h_t)$ is the negative gradient of the cross-entropy
$\mathrm{CE}\big(q(\cdot \mid h_t), p_\theta(\cdot \mid h_t)\big)$, i.e. it distills the
trainer toward the sampler. On-policy ($q = p_\theta$) the drift vanishes by (1). Off-policy
it is nonzero, and because the sampler is periodically re-synced to the trainer, the
error compounds instead of converging.

## 3. Score centering

### 3.1 Definition and the two properties that make it work

Replace each score by its sampler-centered version:

$$
\tilde{s}_v(h_t) = s_v(h_t) - \bar{s}(h_t).
\tag{7}
$$

Then at every prefix

$$
\mathbb{E}_{v \sim q}\big[Q_t\, \tilde{s}_v\big]
= V_t \,\underbrace{\big(\bar{s} - \bar{s}\big)}_{=0}
+ \mathrm{Cov}_q\big(Q_t, s_v - \bar{s}\big)
= \mathrm{Cov}_q\big(Q_t, s_v\big).
\tag{8}
$$

Two facts carry the result: $\mathbb{E}_q[\tilde{s}_v] = 0$ because *both* expectations
are under the same $q$ (it does not matter that $q \neq p_\theta$), and the
covariance is unchanged because $\bar{s}(h_t)$ is constant in $v$. The drift is
cancelled exactly, the signal is untouched, and the only difference from the on-policy
update is that the covariance is measured under $q$ instead of $p_\theta$. When
$q = p_\theta$ the correction is identically zero.

Unlike importance sampling, the correction is additive, deterministic given the
prefix, and computable exactly, so there is nothing to clip.

### 3.2 Top-$k$ tail model

$\bar{s}(h_t)$ is a sum over the full vocabulary, but the sampler only logs its top-$k$
log-probabilities. Let $H$ be the sampler's top-$k$ head and $\mathcal{T}$ the tail.
Model the sampler's tail by the trainer's, rescaled to the sampler's tail mass:

$$
\hat{q}_v =
\begin{cases}
q_v & v \in H \\[2pt]
\rho\, p_v & v \in \mathcal{T}
\end{cases},
\qquad
\rho = \frac{1 - \sum_{v \in H} q_v}{1 - \sum_{v \in H} p_v}.
\tag{9}
$$

Using (1), the tail sum equals minus the head sum, so the expected score needs the
head only:

$$
\mathbb{E}_{\hat{q}}[s_v]
= \sum_{v \in H} q_v s_v + \rho \Big(\underbrace{\mathbb{E}_{p}[s_v]}_{=0} - \sum_{v \in H} p_v s_v\Big)
= \sum_{v \in H} (q_v - \rho\, p_v)\, s_v .
\tag{10}
$$

With $k = 128$ (`--score-centering-top-k 128`, the paper's default) the head covers
$> 99.9\%$ of the sampler mass and matches full-vocabulary centering.

### 3.3 Composition with importance weights

If an importance-sampling method assigns per-token weights $w_v = f(p_v / q_v)$
(e.g. TIS: $f(r) = \min(r, c)$; MIS: $f(r) = r\,\mathbf{1}[r_{\min} \le r \le r_{\max}]$),
score centering subtracts the expectation of the *weighted* score. Under the tail
model $p_v / \hat{q}_v = 1/\rho$ is constant on the tail, so the head-only form
survives with $\rho$ replaced by

$$
\alpha = \rho\, f(1/\rho),
\qquad
\mathbb{E}_{\hat{q}}[\hat{w}_v s_v] = \sum_{v \in H} (q_v w_v - \alpha\, p_v)\, s_v .
\tag{11}
$$

Vanilla score centering is $f \equiv 1$, giving $\alpha = \rho$.

### 3.4 Loss form

Score centering is implemented as a scalar loss whose gradient is the centered,
weighted score. Per token, with any per-token advantage $A_t$,

$$
\mathcal{L}^{\mathrm{SC}}_t
= - A_t \Big(
\mathrm{sg}[w_{y_t}]\, \log p_\theta(y_t \mid h_t)
\;-\;
\sum_{v \in H_t} \mathrm{sg}\big[q_v w_v - \alpha\, p_v\big]\, \log p_\theta(v \mid h_t)
\Big),
\tag{12}
$$

so that $-\nabla_\theta \mathcal{L}^{\mathrm{SC}}_t = A_t\,(w_{y_t} s_{y_t} - \mathbb{E}_{\hat{q}}[\hat{w} s])$.
All sampler-side quantities are detached; only the trainer's log-probabilities carry
gradient.

## 4. The distillation term

### 4.1 Reverse KL and its on-policy gradient

At a prefix, the reverse KL from student to teacher is
$\mathrm{KL}\big(p_\theta \,\|\, T\big)(h_t) = \mathbb{E}_{v \sim p_\theta}[d_v]$. With $T$ fixed,
$\nabla_\theta d_v = s_v$, so

$$
\nabla_\theta \mathrm{KL}(p_\theta \| T)
= \sum_v \nabla_\theta p_\theta(v)\, d_v + \sum_v p_\theta(v)\, \nabla_\theta d_v
= \mathbb{E}_{p_\theta}[d_v s_v] + \underbrace{\mathbb{E}_{p_\theta}[s_v]}_{=0}
= \mathrm{Cov}_{p_\theta}(d, s),
\tag{13}
$$

where the last step uses (2) and (1) again: $\mathbb{E}_{p_\theta}[d]\,\mathbb{E}_{p_\theta}[s] = 0$.
**The teacher signal is entirely a covariance.** The prefix-mean $\bar{d}$ contributes
nothing on-policy.

On-policy distillation implements (13) by treating $-\beta\, d_t$ as a per-token
reward on the student's own samples: $\mathbb{E}_{y \sim p_\theta}[-\beta\, d_t\, s_t]$ is
exactly $-\beta$ times (13) summed over positions.

### 4.2 Off-policy: sampling from a stale $q$

Now the rollouts come from $q$ (stale by up to $K$ steps), and the per-token reward
is still $-\beta d_t$ with $d_t$ evaluated by the current trainer and the teacher.
Decompose with (2):

$$
\mathbb{E}_{v \sim q}\big[d_v s_v\big]
= \underbrace{\bar{d}_q\, \bar{s}}_{\text{drift}}
+ \underbrace{\mathrm{Cov}_q(d, s)}_{\text{signal}},
\qquad
\bar{d}_q = \mathbb{E}_q[d_v] = \mathrm{KL}(q \| T) - \mathrm{KL}(q \| p_\theta).
\tag{14}
$$

The drift is again in the direction of the sampler, $\bar{s}$, and it is not a teacher
signal: at a fresh sync ($p_\theta = q$) its weight is $\mathrm{KL}(q \| T) > 0$ and it
pushes the trainer *away* from its own stale copy in proportion to how far the teacher
is. With periodic re-syncs this compounds exactly like the drift in Section 2.
Score centering removes it and keeps $\mathrm{Cov}_q(d, s)$.

### 4.3 What the centered distillation term optimizes

$\mathrm{Cov}_q(d, s)$ is the gradient of a clean objective. Since $\nabla_\theta d_v = s_v$
and $\mathbb{E}_q[d_v - \bar{d}_q] = 0$,

$$
\nabla_\theta\, \tfrac{1}{2}\, \mathrm{Var}_{v \sim q}(d_v)
= \mathbb{E}_q\big[(d_v - \bar{d}_q)\, \nabla_\theta d_v\big]
= \mathrm{Cov}_q(d, s).
\tag{15}
$$

So score-centered distillation from $q$ is gradient descent on the $q$-weighted
variance of the student–teacher log-ratio at each prefix. Writing $z_v$ for the
student logits, $d_v - \bar{d}_q = (z_v - \log T_v) - \mathbb{E}_q[z - \log T]$: the
log-partition cancels, the objective is a convex quadratic in the logits, and its
minimum is $\log p_\theta(v \mid h_t) = \log T(v \mid h_t) + \text{const}$ on the support
of $q$, i.e. $p_\theta = T$ at every prefix the sampler visits. Compare (13): on-policy the
update is $\mathrm{Cov}_{p_\theta}(d, s) = \nabla \mathrm{KL}(p_\theta \| T)$; off-policy with
score centering it is $\mathrm{Cov}_q(d, s) = \nabla \tfrac12 \mathrm{Var}_q(d)$. Same pair,
same fixed point; only the covariance measure differs.

## 5. The distillation loss

### 5.1 Per-token advantage

$$
A_t = -\beta\, d_t,
\qquad
d_t = \log p_{\theta_{\mathrm{old}}}(y_t \mid h_t) - \log T(y_t \mid h_t).
\tag{16}
$$

There is no task-reward term. The student log-probability in $d_t$ is the *scoring*
log-probability recorded when the batch was prepared and is detached; it is a fixed
training input, not a function of the weights being updated. The boxed grader's
score on training rollouts is logged as a metric only and never enters $A_t$.

### 5.2 Loss

Substituting (16) into (12):

$$
\boxed{\;
\mathcal{L}_t
= \beta\, d_t
\Big(
\mathrm{sg}[w_{y_t}]\, \log p_\theta(y_t \mid h_t)
-
\sum_{v \in H_t} \mathrm{sg}\big[q_v w_v - \alpha\, p_v\big]\, \log p_\theta(v \mid h_t)
\Big)
\;}
\tag{17}
$$

with $H_t$ the sampler's top-$128$ head at $h_t$, $\rho$ from (9), $\alpha$ from (11)
($\alpha = \rho$ without importance weights), and $w \equiv 1$ unless TIS/MIS is enabled.
The batch loss is the sample-mean reduction of $\mathcal{L}_t$ over response tokens.

### 5.3 Expected update

By (8) with $A_t = -\beta d_t$, and (14)–(15):

$$
-\nabla_\theta \mathbb{E}[\mathcal{L}]
\;=\; -\beta\; \mathbb{E}_x \sum_t \mathbb{E}_{h_t \sim q}\Big[
\mathrm{Cov}_{v \sim q}\big(d_v, s_v\big)
\Big]
\;=\; -\beta\; \mathbb{E}_x \sum_t \mathbb{E}_{h_t \sim q}\Big[ \nabla_\theta\, \tfrac12 \mathrm{Var}_{v \sim q}(d_v) \Big].
\tag{18}
$$

The drift $\beta\, \bar{d}_q\, \bar{s}$ is cancelled at every prefix. What remains is
descent on the $q$-weighted log-ratio variance, whose only stationary point is the
teacher on the sampler's support.

### 5.4 Where $K$ enters

$K$ does not appear in the loss. It sets how far $q$ lags $p_\theta$: $K = 1$ is standard
on-policy distillation and the correction in (17) is numerically negligible; larger $K$
lets each rollout batch be generated from an older policy (cheaper, more reuse of the
inference engine's state), and the correction is what keeps (18) drift-free as
$q$ and $p_\theta$ separate. What $K$ *does* change is coverage — the prefixes $h_t$ are
$q$'s, not $p_\theta$'s — which no per-token correction can fix; that is the trade-off the
refresh interval controls.

## 6. Mapping to the code

| Symbol / step | Code | Flag |
|---|---|---|
| teacher $\log T(y_t \mid h_t)$ | `miles/rollout/on_policy_distillation.py::reward_func` → sglang teacher `/generate` with `return_logprob`; stored as `sample.teacher_log_probs` | `--use-opd --opd-type sglang --rm-url … --opd-log-prob-top-k 0` |
| $A_t = -\beta d_t$, (16) | `miles/backends/training_utils/loss_hub/opd.py::apply_opd_kl_to_advantages` (task reward is zero, so the estimator's advantage is zero and only the OPD term remains) | `--opd-kl-coef β` |
| sampler head $H_t$, $q_v$ | rollout returns `rollout_top_logprob_ids`, `rollout_top_logprobs` | `--use-score-centering --score-centering-top-k 128` |
| $\rho$, $\alpha$, (9)–(11) and loss (12) | `miles/backends/training_utils/loss_hub/score_centering.py::score_centering_loss` | `--use-tis` + `tis_mode` selects $f$ |
| per-batch assembly of (17) | `miles/backends/training_utils/loss_hub/losses.py::_compute_score_centering_pg_loss` (called from `policy_loss_function`) | — |
| $K$ | `train.py::_should_update_weights` | `--update-weights-interval K` |
| boxed grader, monitoring only | `reward_func` → `miles/rollout/rm_hub::rule_based_rm`, returned as the logged rollout reward while the advantage input stays zero | `--opd-monitor-rm-type math` |

Eval datasets (AIME 2024/2025, MATH-500) tag their samples with a per-dataset
`rm_type`; `reward_func` grades those with the built-in reward and never calls the
teacher, so evaluation is unaffected by the distillation setup.

### Serving the frozen sampler and the teacher

The sampler $q$ is a model served by the rollout engines and declared through
`--sglang-config`. A model whose `model_path` differs from `--hf-checkpoint` (or that
sets `update_weights: false`) never receives weight updates, so `q` stays frozen for
the whole run and the initial weight sync is skipped. Two layouts are supported:

| sampler | `--sglang-config` models | teacher log-probs | flags |
|---|---|---|---|
| teacher (`q = T`) | `default` = teacher, frozen; `eval` | the rollout engine's own log-probs | `--rollout-frozen-sampler --opd-teacher-from-rollout-logprobs` |
| initial student (`q = p_0`) | `default` = student, frozen; `teacher` = teacher, frozen; `eval` | scored by the in-job `teacher` router | `--rollout-frozen-sampler --opd-teacher-model teacher` |

`--rollout-frozen-sampler` tells Miles that no training engine will ever receive a
weight update, so the weight-version checks (which otherwise require training data
to come from a synced engine) are skipped and the initial weight broadcast is a no-op.

The eval fleet (`--eval-num-gpus`, `--eval-hf-dir`) evaluates the *student* from HF
snapshots the trainer exports, so evaluation is independent of which model the
training engines serve. `scripts/run_qwen3_1_7b_sc_opd.py` wires all of this;
`--sampler student --no-freeze-sampler --refresh-interval K` is the stale on-policy
variant where the sampler is re-synced every $K$ rollouts.
