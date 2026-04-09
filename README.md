# Diffusion Models for BlackBox Optimization

## DiffusionOptimizerV2 — Algorithm Description

### Overview

`DiffusionOptimizerV2` is an iterative black-box optimizer that maintains an **elite buffer** of the best-evaluated points and trains a **conditional denoising diffusion model** to generate new candidate solutions conditioned on high quality. The ask/tell loop alternates between generating candidates (`ask`) and incorporating their evaluations (`tell`).

The optimizer supports three sample-generation components per batch: **exploitation** (reverse diffusion), **local search** (mutation-based), and **exploration** (random/structured). Their fractions are controlled by cosine annealing schedules.

---

### Step 0: Initialization

When the optimizer is created for a problem with dimension *d* and bounds [lower, upper]:

1. **[Elite buffer](#elite-buffer-management) sizing**:
   - `elite_min = max(2, elite_min_per_dim × d)`
   - `elite_size = max(elite_min, elite_max_per_dim × d)`

2. **[Noise predictor network](#noise-predictor-architectures)** is created based on `noise_pred_arch` (`concat_mlp`, `film_resnet`, or `adaln_resnet`).

3. **[EMA copy](#ema-model)** of the noise predictor is created if `ema_decay > 0`. Used for inference in `ask()`.

4. **[DDPM scheduler](#diffusion-scheduler)** is configured for training (noise schedule, timesteps, prediction type, clipping).

5. **[Inference scheduler](#inference-scheduler-ddpm--ddim)** type is resolved from `inference.type` (`"ddpm"` or `"ddim"`).

6. **[Exploration engine](#exploration-strategies)**: if `exploration.type == "sobol"`, a scrambled Sobol sequence generator is initialized.

7. **Start state**: `exploration_stage = True`, empty elite buffers, `best_y = -∞`.

---

### Step 1: `ask(n)` — Generate candidate solutions

#### Regime A: Exploration stage (`exploration_stage == True`)

Active until `len(x_data) >= _min_data_count` (clamped to `elite_size`). Returns `n` samples from the configured [exploration strategy](#exploration-strategies) without using the diffusion model.

#### Regime B: Exploitation stage

Once exploration ends, `ask(n)` produces a **three-component mixture**:

1. **Compute fractions** for the three components:
   - **Exploration** (`explore_frac`): [cosine-annealed](#exploration-annealing) from configured value down to 0.01.
   - **[Local search](#local-search)** (`local_search.frac`): if enabled, [cosine-grows](#local-search) from `frac` toward `frac_max`.
   - **Exploitation**: the remainder of the batch.

2. **[Exploitation samples](#reverse-diffusion-inference)** (`n_exploit`):
   - Create an [inference scheduler](#inference-scheduler-ddpm--ddim) (DDPM or DDIM).
   - Sample initial noise x_T ~ N(0, I) in [preconditioned space](#preconditioning).
   - Compute [conditioning target](#conditioning-strategies) y_cond.
   - Run reverse diffusion for `num_timesteps` steps.
   - [Unprecondition](#preconditioning), denormalize, clip to bounds.

3. **[Local search samples](#local-search)** (`n_local`): mutation-based candidates around anchor points from the elite buffer.

4. **Exploration samples** (`n_explore`): samples from the configured [exploration strategy](#exploration-strategies).

5. Concatenate, shuffle, return.

---

### Step 2: `tell(x, y)` — Incorporate evaluations and train

#### 2a. Data ingestion

- Objective values are **negated** internally (minimization → maximization).
- Each valid point is appended to the elite buffer.
- Global best (`best_x`, `best_y`) is updated.

#### 2b. Exploration → exploitation transition

If `exploration_stage` is still True and `len(x_data) < _min_data_count`: return without training. Otherwise, set `exploration_stage = False` permanently.

`_min_data_count` is clamped to `elite_size` so the optimizer cannot get stuck in exploration.

#### 2c. [Elite buffer management](#elite-buffer-management) (`_update_elite`)

Score and evict lowest-scoring points if buffer exceeds capacity. Supports fixed-size and [adaptive shrinking](#adaptive-elite). After eviction, the [global best is pinned](#global-best-pinning) — guaranteed to remain in the buffer.

#### 2d. Update y-normalization

Recompute `y_mean` and `y_std` from the elite buffer only.

#### 2e. Update [preconditioning](#preconditioning) transform

Recompute the whitening/standardization transform from the current elite buffer.

#### 2f. Optional periodic reinitialization

If `reinit_interval > 0` and due: reset model weights and optimizer, then train for `reinit_epochs`.

#### 2g. Prepare training tensors

1. Normalize x → [-1, 1], then [precondition](#preconditioning) → training space.
2. Build conditioning labels using [y_norm_type](#y-normalization):
   - `"rank"`: percentile ranks in [0, 1]. Optional [augmentation](#conditioning-augmentation) adds noise to top-ranked labels.
   - `"mean_std"`: z-score normalization.
3. Compute [rank-weighted sampling](#rank-weighted-sampling) probabilities.

#### 2h. Train networks (`_train_networks`)

- Mini-batch SGD with [rank-weighted sampling](#rank-weighted-sampling).
- Optional data augmentation via `x_noise_std`.
- Forward: add noise → predict noise → [Min-SNR weighted](#min-snr-loss-weighting) MSE loss.
- Backward: AdamW + gradient clipping + [EMA update](#ema-model).
- [Adaptive early stopping](#adaptive-early-stopping): patience and threshold scale with optimization progress.

---

### Summary of data flow

```
tell(x_raw, y_raw)
  │
  ├─ Negate y (min→max), append to buffer, update global best
  ├─ Check exploration→exploitation transition
  ├─ _update_elite() ─── score & evict ──→ buffer ≤ capacity
  │    └─ _pin_global_best() ─── ensure best point remains
  ├─ _update_normalization() ──→ y_mean, y_std
  ├─ _update_precondition() ──→ _pc_transform, _pc_inverse
  ├─ x_pc = precondition(normalize(x))
  ├─ y_cond = rank(y) ∈ [0,1]  (+ optional augmentation)
  ├─ weights = exp(-rank / (rank_temp × n))
  └─ _train_networks(x_pc, y_cond, weights)
       ├─ Forward: add noise → predict noise → Min-SNR MSE
       ├─ Backward: AdamW + grad clip + EMA update
       └─ Adaptive early stopping

ask(n)
  │
  ├─ Compute explore / local / exploit fractions (cosine-annealed)
  ├─ Exploitation (reverse diffusion):
  │    ├─ x_T ~ N(0, I)
  │    ├─ y_cond = conditioning_strategy(n_exploit)
  │    ├─ Reverse diffusion T→0  [DDPM or DDIM scheduler]
  │    ├─ unprecondition → denormalize → clip
  │    └─ replace NaN/Inf with random
  ├─ Local search (mutations around anchors):
  │    ├─ select top anchors by elite_score
  │    ├─ covariant Gaussian mutations (σ decayed by schedule)
  │    └─ clip to bounds
  ├─ Exploration (uniform / sobol / LHS)
  └─ Shuffle & return
```

---

## Detailed Component Reference

### Noise Predictor Architectures

Set via `noise_pred_arch`:

- **`concat_mlp`**: Plain MLP, concatenates [x_t, time_emb, y_cond] at input. Conditioning only enters at the first layer and can be diluted through depth.
- **`film_resnet`** *(recommended)*: MLP with residual blocks + FiLM conditioning. Time+y conditioning is injected at every layer via learned scale+shift; LayerNorm stabilizes online training.
- **`adaln_resnet`**: DiT-style Adaptive LayerNorm residual blocks. Zero-initialized gates make the network start near identity (predicts ~zero noise), matching the epsilon-prediction inductive bias.

### EMA Model

When `ema_decay > 0`, an exponential moving average of the noise predictor weights is maintained. After each training step:

```
θ_ema = ema_decay × θ_ema + (1 - ema_decay) × θ_train
```

The EMA network is used for inference in `ask()`. This smooths out online-training instability. Empirically, EMA helps at dim=2 but can hurt at dim≥5 if the decay is too slow (elite buffer shifts faster than EMA adapts). `ema_decay = -1` disables EMA entirely.

### Diffusion Scheduler

Training always uses `DDPMScheduler` from the `diffusers` library, configured with `num_train_timesteps`, `beta_schedule` (e.g., `squaredcos_cap_v2`), `prediction_type` (`epsilon`), and optional `clip_sample`/`clip_sample_range`.

### Inference Scheduler (DDPM / DDIM)

Set via `inference.type`:

- **`"ddpm"`** *(default)*: Standard DDPM reverse process. Each step adds fresh random noise with variance determined by the beta schedule. Equivalent to η=1 in DDIM formulation.

- **`"ddim"`**: DDIM reverse process with controllable stochasticity via `inference.eta`:
  - `eta = 0.0`: Fully deterministic — same x_T always produces same x_0. Maximum precision but minimum diversity.
  - `eta = 0.5` *(recommended)*: Moderate stochasticity. Provides the optimal exploration-exploitation balance — enough diversity for multi-modal search, less noise accumulation than full DDPM.
  - `eta = 1.0`: Equivalent to DDPM.

  DDIM uses the same trained noise predictor as DDPM — only the inference process changes. The `DDIMScheduler` from `diffusers` is used, accepting the same config as DDPM.

### Exploration Strategies

Set via `exploration.type`. Used during exploration stage and for the exploration fraction of each batch:

- **`uniform`**: i.i.d. uniform samples in [lower, upper].
- **`sobol`**: Scrambled Sobol low-discrepancy sequence. Much more uniform coverage than pure random; successive calls continue the same global sequence.
- **`lhs`**: Latin Hypercube Sampling per call. Stratified marginal coverage but no cross-call continuation.

### Exploration Annealing

When `explore_anneal = True`, the exploration fraction is cosine-annealed from `explore_frac` down to 0.01 over the optimization budget. This shifts the batch composition from exploration-heavy to exploitation-heavy as the model improves.

### Preconditioning

Set via `precondition.type`. Applied to box-normalized [-1, 1] inputs before training and inference:

- **`none`**: No transform.
- **`standardize`**: Per-dimension scaling to unit variance.
- **`whiten`**: Full covariance whitening via eigendecomposition. Falls back to `standardize` if n ≤ d.
- **`whiten_shrink`** *(recommended)*: Ledoit-Wolf shrinkage covariance → whitening. Stable even when n is not much larger than d.
- **`whiten_ema`**: EMA-smoothed shrinkage covariance → whitening.

Preconditioning changes what the diffusion model sees, how diversity is scored (KNN distances are computed in preconditioned space), and requires dynamic `_pc_clip_range` adjustment for the scheduler.

### Elite Buffer Management

The optimizer maintains a fixed-capacity (or adaptively-sized) buffer of the best points.

**Elite scoring strategies** (`elite_filter.type`):

- **`quality`**: Score = y-value. Most exploitative.
- **`quality_knn`**: Score = normalized_y + `diversity_weight` × normalized_knn_distance. Distances computed in preconditioned space. Preserves spatial diversity for multi-modal problems.
- **`crowding`**: NSGA-II style tier + crowding distance scoring.
- **`grid`**: MAP-Elites-like grid archive. Best point per cell gets priority.

### Global Best Pinning

After every elite eviction, the global best point is guaranteed to remain in the buffer. If the best point was evicted (e.g., by a diversity-based filter), it is appended back. This ensures the model always trains on the best-known solution.

### Adaptive Elite

When `elite_adaptive.enabled = True`, the buffer capacity shrinks linearly from `elite_max` to `elite_min` over the optimization budget (LSHADE-style). A soft quality floor gradually rises, preventing the buffer from retaining very poor points.

### Conditioning Strategies

Set via `conditioning.type`. Determines the y_cond values used during reverse diffusion in `ask()`:

- **`optimistic`**: y_cond = 1.0 + `optimism`, with optional Gaussian `spread` across the batch.
- **`percentile_annealing`**: y_cond = annealing percentile of elite buffer (from `p_low` to `p_high`), with optional `spread`. Always within training distribution — no extrapolation.
- **`diverse_batch`** *(recommended)*: Each sample gets a different y_cond, linearly spaced from `low_percentile` to 1.0 + `high_offset`. Creates an exploration-exploitation spectrum within each batch.
- **`combined`**: Annealed percentile sets upper bound, diverse batch provides within-batch spread.

### Conditioning Augmentation

When `conditioning.aug = True` and `y_norm_type = "rank"`, training labels for top-20% points (rank ≥ 0.8) are augmented with small positive uniform noise [0, alpha]. This extends the training distribution above 1.0 so inference targets (which may request y_cond > 1.0) fall within the trained range. Alpha is derived from the conditioning parameters (e.g., `optimism` or `high_offset`).

**Note**: Empirically, aug=False with DDIM η=0.5 performs better than aug=True, as augmentation blurs the conditioning signal that diverse_batch relies on.

### Rank-Weighted Sampling

Training mini-batches are sampled with rank-based weights:

```
weight(rank) = exp(-rank / (rank_temperature × n))
```

where n is the current buffer size. `rank_temperature` controls focus: 0.3 = sharper focus on top points, 0.5 = moderate, 0.8 = nearly uniform. The temperature scales with buffer size so the effective "active fraction" remains constant regardless of buffer capacity.

### Y-Normalization

Set via `y_norm_type`:

- **`rank`** *(recommended)*: Map y-values to percentile ranks in [0, 1] (worst → 0, best → 1). Stable across iterations, avoids extrapolation issues.
- **`mean_std`**: Z-score normalization using elite buffer statistics.

### Min-SNR Loss Weighting

When `snr_loss_weighting = True`, the per-timestep diffusion loss is weighted by `min(SNR(t), γ) / SNR(t)`. This reduces dominance of easy high-SNR timesteps (small t, low noise) and focuses training on informative medium-noise timesteps.

### Adaptive Early Stopping

Training uses early stopping with parameters that adapt to optimization progress:

- **Patience**: 5 → 20 (increases with progress — late-phase training gets more time for fine-tuning)
- **Relative delta**: 2% → 0.5% (tightens with progress — demands smaller improvements to continue)
- **Warmup**: at least 2 full epochs before early stopping activates

This ensures: early in optimization (volatile data), training stops quickly to avoid overfitting stale data. Late in optimization (stable buffer), training runs longer for precise model refinement.

### Local Search

When `local_search.enabled = True`, a fraction of the batch is generated via mutation of anchor points from the elite buffer, bypassing the diffusion model entirely:

1. **Anchor selection**: Top `n_anchors` points by elite score.
2. **Mutation**: Gaussian perturbations scaled by σ. If `use_covariance = True` and preconditioning is active, mutations follow the learned correlation structure via the inverse preconditioning matrix.
3. **Sigma schedule** (`sigma_decay`):
   - `"quadratic"` *(recommended)*: σ = σ_init × (1 − progress)². Smooth, deterministic, monotonically decreasing.
   - `"one_fifth"`: Adaptive 1/5 success rule. Increases σ if >20% of mutations improve, decreases otherwise.
   - `"linear"`: σ = σ_init × max(1 − progress, 0.01).
4. **Fraction annealing**: When `anneal = True`, the local search fraction cosine-grows from `frac` to `frac_max` over the budget, increasing local refinement as the model matures.

Local search is especially effective on rugged/fractal landscapes (e.g., Weierstrass, Katsuura) where the diffusion model cannot learn useful structure. The mutation-based approach operates independently of model quality.
