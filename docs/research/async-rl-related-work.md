# Related Work: Async / Elastic RL Systems for LLMs

> **Scope.** Literature survey assembled while scoping the *Elastic Fully-Async RL for
> Long-Context Reasoning* idea (length-trajectory–driven reallocation of GPUs between
> rollout generation and training). Organized by theme, with each entry's problem,
> mechanism, headline result, and relevance to our idea.
>
> **Sourcing caveat.** Compiled from paper abstracts and search-level summaries. Full
> PDFs could not be fetched in this environment (network policy returned HTTP 403 on
> arXiv/HuggingFace), so claims are accurate at the level of stated problem, mechanism,
> and headline numbers — but not verified against full-text methodology. Treat numbers
> as reported-by-authors. Re-verify against the PDFs before citing in a paper.

---

## 1. Asynchronous / disaggregated RL systems (the base architecture)

These decouple generation from training so neither blocks the other. This is the
foundation our work builds on (our codebase's `train_async.py` + persistent rollout
worker sits in this family).

### AReaL — *A Large-Scale Asynchronous RL System for Language Reasoning* (2505.24298)
- **Problem.** Synchronous RL alternates generate↔train; the model update must wait for
  the *longest* output in the batch → GPU underutilization.
- **Mechanism.** Fully decouples streaming generation from training; rollout workers
  generate continuously. **Balances rollout vs training workers to bound staleness**, and
  uses a **staleness-enhanced / decoupled PPO** objective to stay stable under stale data.
- **Result.** Up to **2.57×** over the best synchronous systems at matched GPUs, with equal
  or better final quality.
- **Relevance.** Closest conceptual baseline for the async base + staleness control. Its
  worker balancing is *quasi-static*; it does not track length growth over the run.

### AReaL-Hex — *Asynchronous RL over Heterogeneous GPUs* (2511.00796)
- **Problem.** Rollout (HBM-IO-bound) and training (compute-bound) have different hardware
  sweet spots; homogeneous clusters are wasteful.
- **Mechanism.** Heterogeneity-aware scheduler; formulates allocation as **constrained
  optimization with explicit staleness bounds** — phase 1 MILP for parallel strategy +
  workload assignment, phase 2 graph partitioning for device allocation.
- **Result.** Throughput/cost gains on 1.5B–14B math-reasoning RL.
- **Relevance.** Shows allocation-as-optimization with staleness constraints — but solved
  **once**, not adapted to a drifting length trajectory.

### LlamaRL — *Distributed Asynchronous RL Framework* (Meta, 2505.24034)
- **Mechanism.** Single-controller, native-PyTorch, fully distributed async; colocated
  model offloading, async off-policy training, DDMA (distributed direct memory access) for
  weight sync. Includes a formal proof that async yields strict speed-up.
- **Result.** Up to **10.7×** vs DeepSpeed-Chat-like systems on a 405B policy; advantage
  grows with scale.
- **Relevance.** Demonstrates async + offload at extreme scale; allocation is static.

### AsyncFlow — *Asynchronous Streaming RL Framework* (2507.01663)
- **Mechanism.** Distributed data storage/transfer module for fully streamed data
  management + fine-grained scheduling; producer–consumer workflow that **defers parameter
  updates within a staleness threshold**; engine-agnostic, service-oriented interfaces.
- **Result.** Avg **1.59×** over SOTA baseline.
- **Relevance.** Streaming data plane + staleness-bounded updates; no length-driven
  resource reallocation.

### Laminar — *Scalable Asynchronous RL Post-Training* (EuroSys'26, 2510.12633)
- **Problem.** Extreme long-tail skew in trajectory generation → GPU underutilization at
  scale.
- **Mechanism.** **Trajectory-level asynchrony** (generate/consume each trajectory
  independently). Two parts: (1) a **distributed parameter service** of relay workers for
  async fine-grained weight sync (rollouts pull latest weights without stalling the actor),
  and (2) a **dynamic repack** that consolidates long-tail trajectories onto a few
  dedicated rollouts. Decoupling also isolates failures.
- **Result.** Up to **5.48×** throughput on a 1024-GPU cluster; faster convergence.
- **Relevance.** Strong long-tail + weight-sync design; the repack idea is complementary to
  our straggler handling.

---

## 2. Elastic / dynamic resource allocation between train & inference

The closest competitors to our idea. Each lets GPUs flex between roles — but on different
triggers and granularities than "length trajectory over the run."

### StreamRL — *Scalable, Heterogeneous, Elastic RL with Disaggregated Stream Generation* (2504.15930)
- **Problem.** Two bubble types in disaggregated RL: **pipeline bubbles** (stage
  dependencies) and **skewness bubbles** (long-tail output lengths).
- **Mechanism.** **Stream Generation Service** returns finished samples to the trainer in a
  stream (overlap). Uses an **output-length ranker** + **skewness-aware scheduling** to give
  long-tail samples more resources and adjust batch size. Disaggregation enables flexible /
  heterogeneous / cross-datacenter allocation.
- **Result.** Disaggregated streaming **+15%**, async **+25%**.
- **Relevance.** **Most direct prior art** on elastic disaggregation + length skew. Key
  differences vs our idea: it models *intra-step* length skew and (largely profiled)
  allocation, not the **monotonic cross-run growth of mean length** nor **parallelism-shape**
  changes at long context.

### DynaTrain — *Fast Online Parallelism Switching for Elastic LLM Training* (2605.18815)
- **Problem.** Optimal parallelism layout shifts due to resource fluctuation, **RLHF phase
  shifts**, and cluster elasticity; static execution can't follow without checkpoint-restart.
- **Mechanism.** **Sub-second online reconfiguration** across arbitrary multi-dim
  parallelism via a **Virtual Parameter Space** abstraction (any config = deterministic
  mapping; transitions = geometric intersections), a deadlock-free rank-local state-routing
  layer, and an **Elastic Device Manager** that overlaps new-world construction with
  ongoing training to mask topology-change cost. Motivated by progressively releasing GPUs
  from inference → training.
- **Relevance.** **The mechanism we'd want** for cheap online shape/count changes. It
  provides *how to switch*; our contribution is *when/why to switch* (length trajectory) and
  *whether it pays* in the long-context regime.

### ROSE — *Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL* (2605.06534)
- **Problem.** Agentic RL rollouts are long-tail/multi-turn; static provisioning wastes
  GPUs (over) or causes contention (under). Production **serving clusters have idle
  headroom**.
- **Mechanism.** Opportunistically repurpose underutilized **serving** GPUs for rollout
  while preserving serving SLOs. **Cross-cluster weight-transfer engine** (async
  fault-tolerant propagation, shard-aware routing across heterogeneous parallelism,
  **sparsity-aware compression** exploiting ~95% RL weight-delta sparsity → <20s transfer on
  20 Gbps Ethernet). **Elastic rollout scheduler** with turn-wise concurrency-aware routing
  and cache-affinity (KV-prefix) placement.
- **Relevance.** Elasticity sourced from *spare serving capacity* (a different supply than
  our intra-job swing nodes); the sparsity-aware weight transfer is reusable.

### RLBoost — *Harvesting Preemptible Resources for Cost-Efficient RL* (NSDI'26, 2510.19225)
- **Mechanism.** Reserved on-demand **training** cluster + opportunistic offload of
  **rollout** to **preemptible/spot** instances. **Adaptive rollout offload with partial
  response seeding**, **pull-based weight transfer** for fast join, **token-level response
  collection** to minimize preemption loss. Keeps **synchronous** RL semantics.
- **Result.** Up to **1.97×** throughput, **28–49%** better cost efficiency (8B–32B).
- **Relevance.** Reinforces the asymmetry we rely on (rollout scales out on independent
  instances; training needs tightly-coupled GPUs). Elasticity driven by spot availability,
  not length.

### SeamlessFlow — *Trainer–Agent Isolation via Tag Scheduling* (2508.11553)
- **Mechanism.** Isolates trainer from agent rollout; **tag-based scheduling** for
  bubble-free pipelines.
- **Relevance.** Another scheduling-layer approach to the bubble problem.

### Adaptive Placement & Parallelism for RLHF (2312.11819)
- Earlier work on adaptively choosing placement/parallelism for RLHF stages — predecessor
  framing for "the right layout differs per stage."

---

## 3. Long-tail generation, partial rollout, length-aware scheduling

Directly relevant to the **long-context (4k→100k) regime**, where per-sample latency and
intra-batch length variance dominate. Our analysis argued these may matter *more* than
node-count allocation.

### APRIL — *Active Partial Rollouts in RL* (2509.18521)
- **Mechanism.** **Over-provision** rollout requests, **stop once the target count is
  reached**, and **recycle incomplete responses** for continuation in later steps — taming
  long-tail straggler stalls.
- **Relevance.** The canonical partial-rollout pattern; our codebase already has
  `--partial-rollout` + off-policy masking. Core mitigation for the staleness/straggler side.

### SortedRL — *Online Length-Aware Scheduling* (2603.23414)
- **Mechanism.** Dynamically **batches samples with similar predicted output lengths** to
  cut idle/bubble time; sorts rollouts by predicted length.
- **Relevance.** Length-aware batching to kill skewness bubbles within a step.

### Truncated PPO (2506.15050)
- **Mechanism.** Truncate long rollouts to pipeline downstream work without waiting for full
  generation (with corresponding objective handling).
- **Relevance.** Trades completeness for throughput on the tail — relevant when single
  generations reach 100k.

### Adaptive Drafter — *Taming the Long-Tail with speculative decoding* (2511.16665)
- **Mechanism.** Speculative decoding / drafter adapted to accelerate the long-tail of
  reasoning RL generation.
- **Relevance.** Attacks the binding constraint (generation latency) directly — often higher
  ROI than reallocation.

### Seer — *Online Context Learning for Fast Synchronous RL* (2511.14617)
- **Mechanism.** Online context learning to speed up *synchronous* RL generation.

### Heddle — *Distributed Orchestration for Agentic RL Rollout* (2603.28101)
- **Mechanism.** Orchestration system for complex agentic rollout workloads.

---

## 4. Staleness-aware objectives / off-policy correction (the algorithm side)

If we reallocate / run async aggressively, data becomes stale; these keep learning stable.

### Asynchronous RLHF (ICLR'25, 2410.18252)
- **Idea.** Run generators (behavior policy, older weights) and trainers (target policy)
  concurrently → off-policy. Shows off-policy async RLHF can be **faster and as effective**
  with appropriate handling.
- **Relevance.** Foundational justification that bounded off-policy training is viable.

### A-3PO — *Staleness-aware Proximal Policy Approximation* (2512.06547)
- **Problem.** Decoupled-PPO's proximal policy needs an **extra forward pass per step**
  (~10s for LLMs).
- **Mechanism.** Approximate the proximal/trust-region anchor by **interpolating behavior
  and target policies in log-prob space** (fresher data weighted more) — near-zero compute
  (~0.0012s).
- **Result.** **1.8×** speedup at comparable quality.
- **Relevance.** Cheap staleness correction; pairs well with aggressive async/reallocation.

### VESPO — *Variational Sequence-level Soft Policy Optimization* (2602.10693)
- **Mechanism.** Variance-reduced variational objective with a **closed-form reshaping
  kernel on sequence-level importance weights** for off-policy training.

### Staleness-Constrained Rollout Coordination (2601.12784)
- **Mechanism.** Coordinates rollouts under explicit staleness constraints for efficient
  async post-training.

### GAC — *Gradient Alignment Control* (2603.01501)
- **Mechanism.** Stabilizes async RL by controlling gradient alignment between stale and
  fresh data.

### Periodic Asynchrony (2511.18871)
- **Mechanism.** An **on-policy** approach that recovers much of async's speed while
  limiting off-policyness — a point on the sync↔async spectrum.

---

## 5. Foundational RLHF frameworks (context)

- **HybridFlow / veRL** (2409.19256) — flexible hybrid-controller RLHF dataflow; widely used
  base.
- **ROLL / RollArt** (2512.22560) — scaling **agentic** RL training on disaggregated
  infrastructure.
- **ARL-Tangram** (2603.13019) — resource efficiency for agentic RL.
- (OpenRLHF, NeMo-Aligner — commonly cited colocated/disaggregated baselines.)

---

## 6. Synthesis: where our idea sits

**What is already solved (and would be our baselines):**
- Async/disaggregated generation↔training with staleness bounds — AReaL, LlamaRL, AsyncFlow,
  Laminar.
- Elastic GPU flex between roles — StreamRL (profiled/skew-aware), DynaTrain (online
  parallelism switching mechanism), ROSE (spare serving capacity), RLBoost (spot capacity).
- Long-tail / partial rollout / length-aware batching — APRIL, SortedRL, Truncated PPO,
  Laminar repack.
- Cheap staleness correction — Asynchronous RLHF, A-3PO, VESPO, GAC.

**The gap our work targets (the extreme long-context regime, 4k→100k):**
1. **Length-trajectory–driven reconfiguration.** Prior elasticity reacts to *intra-step*
   phase/skew (StreamRL, DynaTrain) or to *external* supply (ROSE, RLBoost). None keys the
   policy on the **monotonic cross-run growth of mean context length** that defines reasoning
   RL.
2. **Parallelism-shape co-adaptation at feasibility thresholds.** At ~100k, the optimal
   *shape* changes (context parallelism on, larger inference KV budget), not just the node
   *count*. DynaTrain supplies a switching mechanism; the *trigger/policy* tied to length is
   open.
3. **A characterization of *when elasticity pays*.** Our "wall-clock concentration" argument
   (most clock is spent in the long-context regime, so a static-split-for-late + partial
   rollout may capture most of the gain, and the bottleneck relocates to staleness/stragglers)
   is, to our knowledge, **not** framed by any of these systems. This cost-model +
   characterization is the most defensible contribution; the mechanism is second.

**Reusable components from the literature:** DynaTrain's online switching, ROSE's
sparsity-aware weight transfer, Laminar's relay-based weight service and repack, APRIL's
recycle, A-3PO's near-free staleness correction.

---

## Citations

| Theme | Paper | Link |
|---|---|---|
| Async base | AReaL | https://arxiv.org/abs/2505.24298 |
| Async base | AReaL-Hex | https://arxiv.org/abs/2511.00796 |
| Async base | LlamaRL | https://arxiv.org/abs/2505.24034 |
| Async base | AsyncFlow | https://arxiv.org/abs/2507.01663 |
| Async base | Laminar | https://arxiv.org/abs/2510.12633 |
| Elastic alloc | StreamRL | https://arxiv.org/abs/2504.15930 |
| Elastic alloc | DynaTrain | https://arxiv.org/abs/2605.18815 |
| Elastic alloc | ROSE | https://arxiv.org/abs/2605.06534 |
| Elastic alloc | RLBoost | https://arxiv.org/abs/2510.19225 |
| Elastic alloc | SeamlessFlow | https://arxiv.org/abs/2508.11553 |
| Elastic alloc | Adaptive Placement/Parallelism RLHF | https://arxiv.org/abs/2312.11819 |
| Long-tail | APRIL | https://arxiv.org/abs/2509.18521 |
| Long-tail | SortedRL | https://arxiv.org/abs/2603.23414 |
| Long-tail | Truncated PPO | https://arxiv.org/abs/2506.15050 |
| Long-tail | Adaptive Drafter | https://arxiv.org/abs/2511.16665 |
| Long-tail | Seer | https://arxiv.org/abs/2511.14617 |
| Long-tail | Heddle | https://arxiv.org/abs/2603.28101 |
| Staleness | Asynchronous RLHF | https://arxiv.org/abs/2410.18252 |
| Staleness | A-3PO | https://arxiv.org/abs/2512.06547 |
| Staleness | VESPO | https://arxiv.org/abs/2602.10693 |
| Staleness | Staleness-Constrained Rollout Coordination | https://arxiv.org/abs/2601.12784 |
| Staleness | GAC | https://arxiv.org/abs/2603.01501 |
| Staleness | Periodic Asynchrony | https://arxiv.org/abs/2511.18871 |
| Framework | HybridFlow/veRL | https://arxiv.org/abs/2409.19256 |
| Framework | ROLL/RollArt | https://arxiv.org/abs/2512.22560 |
| Framework | ARL-Tangram | https://arxiv.org/abs/2603.13019 |
| Analysis | SemiAnalysis — Mind the Gap | https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching |
</content>
</invoke>
