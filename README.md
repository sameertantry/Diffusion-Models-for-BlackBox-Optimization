# Diffusion Models for BlackBox Optimization

## DiffusionOptimizerV2 (step-by-step algorithm)

### Overview

`DiffusionOptimizerV2` is an iterative black-box optimizer that maintains an **elite buffer** of the best-evaluated points and trains a **conditional denoising diffusion model** (DDPM) to generate new candidate solutions conditioned on high quality. The ask-tell loop alternates between generating candidates (`ask`) and incorporating their evaluations (`tell`).

---

### Step 0: Initialization

When the optimizer is created for a problem with dimension \( d \) and bounds \([\text{lower}, \text{upper}]\):

1. **Elite buffer sizing**:
   - `elite_min = max(2, elite_min_per_dim × d)`
   - `elite_size = max(elite_min, elite_max_per_dim × d)`
   - Example: defaults `elite_min_per_dim=5`, `elite_max_per_dim=100`, \(d=5\) → `elite_min=25`, `elite_size=500`.

2. **Noise predictor network** is created based on `noise_pred_arch`:
   - **`concat_mlp`**: Plain MLP that concatenates \([x_t, \text{time\_emb}, y_\text{cond}]\) at the input. Conditioning only enters at the first layer and can be diluted through depth. Good baseline and can be OK in very low dimension; usually weaker for complex conditioning.
   - **`film_resnet`**: MLP with residual blocks + FiLM conditioning. Time+y conditioning is injected at *every* layer via learned scale+shift; LayerNorm stabilizes online training. This prevents conditioning dilution and is the recommended default.
   - **`adaln_resnet`**: DiT-style Adaptive LayerNorm residual blocks. Conditioning controls normalization parameters and a learned gate. Gates are zero-initialized so the network starts near identity (predicts ~0 noise), which matches the epsilon-prediction inductive bias.

3. **EMA copy** of the noise predictor is created if `ema_decay > 0`. This EMA network is used for inference in `ask()` because it is typically more stable than the raw online-updated network.

4. **DDPM scheduler** (`DDPMScheduler`) is configured with:
   - number of diffusion timesteps (`num_train_timesteps = num_timesteps`)
   - beta schedule (`beta_schedule`, `beta_start`, `beta_end`)
   - prediction type (`prediction_type`)
   - optional internal clipping (`clip_sample`, `clip_sample_range`)

5. **Exploration engine**:
   - If `exploration.type == "sobol"`, initialize a scrambled Sobol sequence generator so that initial sampling is low-discrepancy (space-filling across repeated calls).

6. Start state:
   - `exploration_stage = True` (pure initial exploration before the model is used)
   - empty elite buffers `x_data`, `y_data`
   - global best `best_x`, `best_y`

---

### Step 1: `ask(n)` — Generate candidate solutions

There are two regimes.

#### Regime A: Exploration stage (`exploration_stage == True`)

Active until enough data has been collected (`len(x_data) >= _min_data_count`). In this stage `ask(n)` returns `n` samples from `_sample_initial(n)` and **does not use the diffusion model**.

The exploration strategy is controlled by `exploration.type`:

- **`uniform`**:
  - i.i.d. uniform samples in \([\text{lower}, \text{upper}]\).
  - Pros: simplest and cheap.
  - Cons: in higher \(d\), coverage is noisy and can leave large holes.

- **`sobol`**:
  - scrambled Sobol low-discrepancy sequence.
  - Pros: much more uniform coverage than pure random for the same number of points; sequential calls continue the same global sequence.
  - Good when you want deterministic space-filling structure in early data.

- **`lhs`**:
  - Latin Hypercube Sampling (LHS) per call.
  - Pros: stratified marginal coverage in each dimension for each batch.
  - Cons: successive calls are independent (no global low-discrepancy “continuation” like Sobol).

**Motivation for initial exploration**: before training, the diffusion model has no data. The initial data must cover the space well enough that the elite buffer is not biased to a random local region.

#### Regime B: Exploitation stage (`exploration_stage == False`)

Once exploration ends, `ask(n)` produces a **mixture** of exploitation (model-guided) and exploration (random) samples.

1. **Choose exploration fraction**:
   - If `explore_anneal=True`, compute `current_frac` via cosine annealing from the configured `explore_frac` down to 0.01 over the budget (using `num_evals / budget`).
   - Otherwise `current_frac = explore_frac`.
   - Split: `n_explore = int(n * current_frac)`, `n_exploit = n - n_explore`.

2. **Exploitation samples** (`n_exploit`):
   - Create an inference `DDPMScheduler` using the same scheduler config but with `clip_sample_range` replaced by `_pc_clip_range` (dynamic, tracked from training data in preconditioned space).
   - Sample initial noise: \( x_T \sim \mathcal{N}(0, I) \) in **preconditioned space**.
   - Compute conditioning target tensor \(y_\text{cond}\) using the selected conditioning strategy (see Step 3).
   - Run the reverse diffusion loop for `num_timesteps` steps:
     - Predict noise: \( \epsilon_\theta(x_t, t, y_\text{cond}) \)
     - Optionally apply regressor guidance (if enabled)
     - Scheduler step: produce \(x_{t-1}\)
   - After reverse diffusion ends:
     - Convert output to numpy: `x_np_pc` (still in preconditioned space)
     - Inverse preconditioning: `x_np_norm = _unprecondition_x(x_np_pc)` (back to \([-1,1]\) box-normalized)
     - Denormalize: `_denormalize_x(x_np_norm)` back to original bounds
     - Clip to `[lower, upper]`
     - Replace any NaN/Inf rows with uniform random samples

3. **Exploration samples** (`n_explore`):
   - Uniform random samples in the original bounded domain.

4. Concatenate exploitation + exploration candidates, shuffle them, register the ask, return.

---

### Step 2: `tell(x, y)` — Incorporate evaluations and train

This step appends new evaluated points, maintains the elite buffer, updates transforms/statistics, and trains the model online.

#### 2a. Data ingestion and objective direction

- Inputs `x` are cast to float32.
- Objective values `y_obj` are treated as **minimization** externally; internally the optimizer stores:
  - `y = -y_obj` so it becomes a maximization signal.
- For each valid `(x_i, y_i)`:
  - increment `num_evals`
  - append to `x_data`, `y_data`
  - update global best `best_x`, `best_y` if this point is best so far

#### 2b. Exploration → exploitation transition (data threshold)

If `exploration_stage` is still `True`:
- If `len(x_data) < _min_data_count`: just register `tell()` for verbose logs and return (no training).
- If `len(x_data) >= _min_data_count`: set `exploration_stage = False` and proceed with training.

`_min_data_count` is derived primarily from `min_data_per_dim × d` (if provided) and is clamped so it does not exceed `elite_size`. This guarantees that once the elite buffer is filled to capacity, the optimizer can exit exploration stage and remain in exploitation stage.

#### 2c. Elite buffer management (`_update_elite`)

The optimizer keeps only a fixed-size set of “elite” points (or an adaptively-sized set if enabled).

- **Fixed-size mode** (default):
  - If buffer size `n` exceeds `elite_size`, compute a score for each point using `elite_filter` and keep the top `elite_size`.

- **Adaptive elite mode** (`elite_adaptive.enabled = true`):
  - Compute progress \(p = \min(\text{num\_evals}/\text{budget}, 1)\)
  - Shrink the elite cap linearly from `_elite_max_init` down to `elite_min` (LSHADE-style).
  - Maintain a **soft floor** on y-values:
    \[
      \text{floor}_t = \max\bigl(\text{floor}_{t-1} (1-\delta),\ \text{percentile}_p(y)\bigr)
    \]
  - Prefer keeping points above the floor; if too few, fall back to keeping more broadly.
  - If too many points are above the floor, select the best `elite_cap` among them using the elite scoring rule.

**Elite selection / scoring strategies** (`elite_filter.type`):

- **`quality`**:
  - Score = y-value (keep best y).
  - Very exploitative; can lose diversity.

- **`quality_knn`**:
  - Score = normalized y + `diversity_weight` × normalized kNN distance.
  - Diversity is computed in **preconditioned normalized space** (apply `_precondition_x` to `_normalize_x(x)`), so distance reflects the geometry the model sees.
  - Good for multimodal problems: keeps several separated basins.

- **`crowding`**:
  - Sort points by y into `n_tiers` tiers (quality bins).
  - Within each tier compute NSGA-II style crowding distance (per-dimension neighbor spans).
  - Final score makes tier dominate, crowding breaks ties.
  - Good when you want strict quality dominance but still want spread.

- **`grid`**:
  - Grid archive over normalized \([-1,1]^d\):
    - Compute grid cell index per point via discretization.
    - Keep the best point per cell (cell-winners get a big score bonus).
    - Fill remaining capacity by global quality.
  - For \(d>5\), use a hashed cell key to avoid combinatorial explosion.
  - Good for explicit coverage across the domain / MAP-Elites-like behavior.

#### 2d. Update y-normalization statistics (`_update_normalization`)

- Recompute `y_mean` and `y_std` from the **elite buffer only**.
- This keeps normalization aligned with what the model is trained on (and reduces drift).

#### 2e. Update preconditioning transform (`_update_precondition`)

Preconditioning acts on **box-normalized** inputs (\([-1,1]^d\)) to improve geometry for diffusion training and sampling. The transform is computed from the current elite buffer.

Let \(x_\text{norm}\) be box-normalized and \(x_c = x_\text{norm} - \mu\).

Supported `precondition.type`:

- **`none`**:
  - No transform. Model trains and samples directly in \([-1,1]^d\).
  - Best when covariance estimation is unreliable or correlations are weak.

- **`standardize`**:
  - Per-dimension scaling:
    - \(\sigma = \text{std}(x_c)\)
    - transform = diag(\(1/\sigma\)), inverse = diag(\(\sigma\))
  - Motivation: make each dimension roughly unit variance; helps match the diffusion Gaussian prior and balances learning across dims.

- **`whiten`**:
  - Full covariance whitening using eigendecomposition of the sample covariance.
  - Motivation: remove correlations so the model learns an isotropic distribution.
  - If \(n \le d\), fall back to `standardize` because covariance is poorly conditioned.

- **`whiten_shrink`**:
  - Compute covariance via Ledoit–Wolf analytical shrinkage (toward scaled identity), then whiten.
  - Motivation: stabilizes whitening when \(n\) is not much larger than \(d\); prevents tiny eigenvalues from causing huge scaling.

- **`whiten_ema`**:
  - Compute shrinkage covariance each iteration and maintain EMA:
    \(\Sigma_t = (1-\eta)\Sigma_{t-1} + \eta \Sigma_{\text{current}}\)
  - Then whiten using the EMA covariance.
  - Motivation: preconditioning changes smoothly as the elite buffer drifts; reduces non-stationarity in the model’s input distribution.

**How preconditioning changes behavior**:
- It changes *what the diffusion model sees*: the model is trained and sampled in the transformed coordinates.
- It changes *diversity scoring* for `quality_knn` and `crowding` because distances are computed after preconditioning (better matches the model’s geometry).
- It can expand or rotate the data distribution; therefore the scheduler’s internal clipping must be compatible (hence `_pc_clip_range`).

#### 2f. Optional periodic reinitialization (`reinit_interval`)

If `reinit_interval > 0` and `_tell_count % reinit_interval == 0`:
- reinitialize the noise predictor and optimizer state from scratch (`_reinit_model`)
- then train for `reinit_train_steps` (typically larger than the usual adaptive steps)

Motivation: the elite distribution drifts; periodic reset prevents accumulation of stale representations.

#### 2g. Prepare training tensors

1. Convert elite buffer to arrays: `x_arr`, `y_arr`, filter non-finite.
2. Compute:
   - `x_norm = _normalize_x(x_arr)` → \([-1,1]\)
   - `x_pc = _precondition_x(x_norm)` → preconditioned training space
3. Update `_pc_clip_range`:
   - If preconditioning is active, set it to cover the observed training range in preconditioned space (plus a small margin).
   - If not, keep it at the configured scheduler range.
4. Build conditioning labels `y_train` depending on `y_norm_type`:
   - **`rank`**:
     - map y-values to percentile ranks in \([0,1]\) (worst→0, best→1)
     - optional label augmentation if `conditioning.aug = true` (top-ranked labels get small positive noise so the model is trained on values slightly above 1.0)
   - **`mean_std`**:
     - z-score normalization using `(y - y_mean) / y_std`
5. Convert to torch tensors (`x_train`, `y_train`).
6. Compute rank-based sampling weights from raw `y_arr`:
   - weights \( \propto \exp(-\text{rank}/\text{temp}) \), where `temp = max(rank_temperature * elite_min, 1.0)`.

#### 2h. Train diffusion (and optional regressor) (`_train_networks`)

Training is online and repeated every `tell()` after entering exploitation stage.

- **Adaptive number of steps**:
  - Determine `steps_per_epoch = max(1, n_data // effective_batch)`.
  - If not reinit-training, choose a number of epochs based on `train_steps` (capped), ensuring at least a few epochs even for small buffers.
- **Per-step procedure**:
  1. Sample indices for a mini-batch, using rank weights if available.
  2. Optionally add small Gaussian noise to x0 (`x_noise_std`) as data augmentation.
  3. Sample random diffusion timesteps \(t\).
  4. Add noise via the scheduler: \(x_t = \text{add\_noise}(x_0, \epsilon, t)\).
  5. Predict noise: \(\hat{\epsilon} = f_\theta(x_t, t, y_\text{cond})\).
  6. Compute loss:
     - MSE between predicted and true noise.
     - If Min-SNR loss weighting is enabled, weight by a function of SNR(t) to prevent easy high-SNR steps dominating training.
  7. Backprop with AdamW, clip gradients, step optimizer.
  8. Update EMA weights.

- **Early stopping**:
  - Track a sliding window average loss.
  - Stop if the average has not improved sufficiently for a patience window after a warmup.

---

### Step 3: Conditioning strategies (used in `ask()` exploitation)

The conditioning target \(y_\text{cond}\) tells the diffusion model what “quality level” to generate.

`conditioning.type` determines how \(y_\text{cond}\) is produced:

- **`optimistic`**:
  - In `rank` mode: centre at \(1.0 + \text{optimism}\), optionally add Gaussian `spread`.
  - Motivation: ask for slightly better-than-best samples.

- **`percentile_annealing`**:
  - Choose a percentile \(p\) that rises from `p_low` to `p_high` as progress increases.
  - In `rank` mode: centre is just \(p\) (no extrapolation).
  - Motivation: conservative early, aggressive late, keeps conditioning within training distribution.

- **`diverse_batch`**:
  - Create a per-sample conditioning vector linearly spanning from `low_percentile` up to `best + high_offset` (or in `rank` mode, up to \(1.0 + \text{high_offset}\)).
  - Motivation: within one batch, some samples explore moderate quality while others exploit the top.

- **`combined`**:
  - Use percentile annealing to set the “upper target” over time and a lower quantile to set the “lower target,” then spread linearly between them.
  - Motivation: combines temporal annealing with within-batch diversity.

---

### Step 4: Reverse diffusion inference loop (inside `ask()` exploitation)

1. Sample \(x_T \sim \mathcal{N}(0, I)\) in **preconditioned space**.
2. For each scheduler timestep \(t\) in reverse:
   - predict noise \(\epsilon_\theta(x_t, t, y_\text{cond})\) with the inference net (EMA if enabled)
   - optionally apply regressor guidance by subtracting a scaled gradient of predicted y wrt \(x_t\)
   - scheduler `step()` computes the next \(x_{t-1}\), with optional internal clipping of its \(x_0\) estimate to `[-_pc_clip_range, +_pc_clip_range]`
3. After finishing:
   - output is \(x_0\) in preconditioned space
   - apply `_unprecondition_x` → back to box-normalized \([-1,1]\)
   - apply `_denormalize_x` → back to original bounds
   - clip to bounds

---

### Step 5: Reset (between problem instances)

When moving to a new problem instance, `reset()` clears:
- the elite buffers (`x_data`, `y_data`), counters, and best-so-far
- y-normalization stats
- preconditioning state (`_pc_mean`, transforms, EMA covariance) and `_pc_clip_range`
- adaptive elite floor
- (if Sobol is enabled) the Sobol engine
- model weights (reinitializes networks and optimizer state)
- verbose trace state and loss logging state

Each problem instance starts completely fresh.


### Summary of data flow through one `tell → ask` cycle
```
tell(x_raw, y_raw)
  │
  ├─ Negate y (min→max)
  ├─ Append to elite buffer
  ├─ Check exploration→exploitation transition
  ├─ _update_elite() ─── score & evict ──→ elite buffer ≤ elite_size
  ├─ _update_normalization() ──→ y_mean, y_std
  ├─ _update_precondition() ──→ _pc_mean, _pc_transform, _pc_inverse
  ├─ x_norm = _normalize_x(x) ──→ [-1, 1]
  ├─ x_pc = _precondition_x(x_norm) ──→ whitened/standardized
  ├─ Update _pc_clip_range from max(|x_pc|)
  ├─ y_cond = rank(y) ∈ [0, 1] (+ optional augmentation)
  ├─ weights = exp(-rank / temp) (rank-weighted sampling)
  └─ _train_networks(x_pc, y_cond, weights)
       ├─ Forward: add noise → predict noise → Min-SNR weighted MSE
       ├─ Backward: AdamW + grad clip + EMA update
       └─ Early stopping if loss plateaus

ask(n)
  │
  ├─ Compute explore fraction (cosine annealed)
  ├─ Exploitation (n_exploit samples):
  │    ├─ x_T ~ N(0, I)
  │    ├─ y_cond = conditioning_strategy(n_exploit)
  │    ├─ Reverse diffusion T→0 using EMA net + DDPMScheduler(clip_range=_pc_clip_range)
  │    ├─ x_pc = x_0 (preconditioned space)
  │    ├─ x_norm = _unprecondition_x(x_pc) ──→ [-1, 1]
  │    └─ x_raw = _denormalize_x(x_norm) ──→ [lower, upper]
  ├─ Exploration (n_explore samples): uniform random in [lower, upper]
  └─ Shuffle & return
```