"""Improved Diffusion-based black-box optimizer (v2).

Addresses training-data distribution degradation observed on the COCO/BBOB
benchmark by replacing the quantile-over-all-data filtering with a
fixed-size elite buffer and several additional improvements.

Changes from :class:`DiffusionOptimizer`
----------------------------------------
1. **Dimension-scaled elite buffer** — the buffer capacity is
   ``elite_max_per_dim × d`` with a floor of ``elite_min_per_dim × d``.
   This prevents the training distribution from being diluted by mediocre
   early-exploration data as the total number of evaluations grows.

2. **X-space normalisation** — inputs are mapped to [-1, 1] via the
   domain bounds so the DDPM noise schedule operates on a standardised
   scale regardless of the original variable ranges.

3. **Target conditioning** — sampling conditions on the best observed
   objective value (with a configurable strategy) rather than on the
   quantile threshold τ, focusing generation on the most promising
   region of the search space.

4. **Exploration mixing** — a configurable fraction of candidates are
   drawn uniformly at random with cosine annealing, preventing premature
   convergence when the model has learned a poor distribution.

5. **Rank-weighted training** — within the elite buffer, better-
   performing points receive higher sampling probability during SGD,
   further sharpening the learned distribution around the optimum.

6. **Stable normalisation** — y-statistics are computed over the elite
   buffer only (not all data ever seen), keeping the normalisation
   tightly coupled with the training distribution.

7. **EMA model for inference** — an exponential moving average of the
   noise predictor weights is maintained for use during ``ask()``,
   smoothing out online-learning instability.

8. **Periodic model reinitialisation** — every ``reinit_interval``
   ``tell()`` calls the model weights and optimiser state are reset and
   trained from scratch on the current elite, preventing accumulation of
   stale representations.

9. **Epoch-based training with early stopping** — training runs for a
   configurable number of epochs (full passes over the elite buffer)
   and stops early when loss plateaus, preventing overfitting.

10. **Min-SNR loss weighting** — per-timestep loss weights based on
    signal-to-noise ratio reduce dominance of easy high-SNR timesteps.

11. **Rank-based y-conditioning** (``y_norm_type="rank"``) — objective
    values are mapped to their percentile rank in [0, 1] (worst→0,
    best→1) rather than using mean/std normalisation.  This keeps the
    conditioning signal stable across iterations and avoids
    extrapolation: every conditioning target used during inference lies
    within (or just above) the distribution the model was trained on.
    The ``"mean_std"`` mode is retained for backward compatibility.

The ask/tell/warm_start/reset interface is identical to the original
so the class is a drop-in replacement in both the COCO wrapper and the
``main.py`` experiment runner.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from methods.base import BaseOptimizer
from methods.diffusion import (
    _Regressor,
    _to_tensor,
    create_noise_predictor,
)

from diffusers import DDIMScheduler, DDPMScheduler


# ------------------------------------------------------------------ #
# Elite filter defaults                                                #
# ------------------------------------------------------------------ #
_ELITE_FILTER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "quality": {},
    "quality_knn": {"k": 5, "diversity_weight": 0.5},
    "crowding": {"n_tiers": 10},
    "grid": {"cells_per_dim": 5},
}

# ------------------------------------------------------------------ #
# Conditioning strategy defaults                                       #
# ------------------------------------------------------------------ #
_CONDITIONING_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "optimistic": {
        "optimism": 0.1,
        "spread": 0.3,
    },
    "percentile_annealing": {
        "p_low": 0.75,
        "p_high": 0.99,
        "spread": 0.0,
    },
    "diverse_batch": {
        "low_percentile": 0.80,
        "high_offset": 0.1,
    },
    "combined": {
        "p_low": 0.75,
        "p_high": 0.99,
        "low_quantile": 0.50,
        "high_offset": 0.1,
    },
}

# ------------------------------------------------------------------ #
# Preconditioning defaults                                             #
# ------------------------------------------------------------------ #
_PRECONDITION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "none": {},
    "standardize": {},
    "whiten": {},
    "whiten_shrink": {},
    "whiten_ema": {"ema_rate": 0.1},
}

# ------------------------------------------------------------------ #
# Exploration strategy defaults                                        #
# ------------------------------------------------------------------ #
_EXPLORATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "uniform": {},
    "sobol": {},
    "lhs": {},
}

# ------------------------------------------------------------------ #
# Inference scheduler defaults                                         #
# ------------------------------------------------------------------ #
_INFERENCE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "ddpm": {},
    "ddim": {"eta": 0.0},
}

# ------------------------------------------------------------------ #
# Local search defaults                                                #
# ------------------------------------------------------------------ #
_LOCAL_SEARCH_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "frac": 0.15,
    "frac_max": 0.30,
    "anneal": True,
    "n_anchors": 5,
    "sigma_init": 0.2,
    "sigma_decay": "quadratic",
    "use_covariance": True,
    "anchor_selection": "elite_score",
}


# ------------------------------------------------------------------ #
# Ledoit-Wolf analytical shrinkage                                     #
# ------------------------------------------------------------------ #

def _ledoit_wolf_shrinkage(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimator toward scaled identity.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Mean-centred data matrix.

    Returns
    -------
    np.ndarray, shape (d, d)
        Well-conditioned covariance estimate.
    """
    n, d = X.shape
    if n < 2:
        return np.eye(d, dtype=np.float64)
    S = X.T @ X / n  # biased sample covariance
    trace_S = np.trace(S)
    mu = trace_S / d

    delta = np.sum((S - mu * np.eye(d)) ** 2) / d
    X2 = X ** 2
    beta_raw = np.sum(np.sum(X2.T @ X2, axis=None)) / (n ** 2) - np.sum(S ** 2)
    beta_raw /= d
    beta = min(beta_raw, delta)
    alpha = beta / (delta + 1e-12)
    alpha = max(0.0, min(alpha, 1.0))
    return (1.0 - alpha) * S + alpha * mu * np.eye(d)

class DiffusionOptimizerV2(BaseOptimizer):
    """Diffusion-based optimizer with elite buffer (v2).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the search space.
    bounds : tuple of np.ndarray
        ``(lower_bounds, upper_bounds)`` for the search domain.
    num_timesteps : int
        Number of diffusion timesteps.
    beta_schedule : str
        Noise schedule type passed to ``DDPMScheduler``.
    beta_start, beta_end : float
        Schedule endpoints (linear / scaled_linear).
    prediction_type : str
        ``"epsilon"`` / ``"sample"`` / ``"v_prediction"``.
    clip_sample : bool
        Whether the scheduler clips its internal x₀ prediction.
    clip_sample_range : float
        Clipping range when *clip_sample* is True.
    hidden_dim, time_embed_dim : int
        Widths of the MLP noise predictor and its time embedding.
    depth : int
        Number of ``nn.Linear`` layers in the noise predictor (>= 2).
    batch_size : int
        Mini-batch size for SGD.
    lr_diffusion, lr_regressor : float
        Learning rates.
    num_epochs : int
        Number of training epochs per ``tell()`` call.  Each epoch
        iterates once over the elite buffer.  Training may terminate
        early if the loss plateaus.
    elite_min_per_dim : int
        Minimum elite buffer size per dimension.  The actual floor is
        ``max(2, elite_min_per_dim × input_dim)``.
    elite_max_per_dim : int
        Maximum elite buffer size per dimension.  The actual capacity is
        ``max(elite_min, elite_max_per_dim × input_dim)``.
    elite_filter : dict or None
        Strategy used to select which points are kept when the elite
        buffer overflows.  A dict with a mandatory ``"type"`` key and
        type-specific parameters.  ``None`` defaults to
        ``{"type": "quality"}`` (pure objective-value ranking).

        Supported types:

        * ``{"type": "quality"}``
            Keep the points with the highest y-value (default).

        * ``{"type": "quality_knn", "k": 5, "diversity_weight": 0.5}``
            Combined quality + spatial diversity score.  For each
            candidate point the mean Euclidean distance to its *k*
            nearest neighbours (in normalised x-space) is computed,
            min–max scaled to [0, 1], and added to the min–max scaled
            y-value weighted by ``diversity_weight``.  The top
            ``elite_size`` points by this combined score are kept.

    guidance_strength : float
        Scale of the regressor gradient during the reverse process.
    use_regressor : bool
        Whether to train/use the auxiliary regressor for guidance.
    explore_frac : float
        Fraction of candidates in ``ask()`` drawn uniformly at random
        (0 = pure exploitation, 1 = pure exploration).
    conditioning_optimism : float
        Standard-deviation units added above the best observed value
        when setting the conditioning target for the reverse process.
        Only used when ``conditioning`` is ``None`` (falls back to the
        ``"optimistic"`` strategy with these values).
    conditioning_spread : float
        Standard deviation of Gaussian noise added to the conditioning
        target across the batch (in normalised y units).  Only used when
        ``conditioning`` is ``None``.
    conditioning : dict or None
        Strategy for choosing the conditioning value ``y_cond`` during
        the reverse diffusion process.  A dict with a mandatory
        ``"type"`` key and type-specific parameters.  ``None`` defaults
        to ``{"type": "optimistic", "optimism": conditioning_optimism,
        "spread": conditioning_spread}``.

        Supported types:

        * ``{"type": "optimistic", "optimism": 0.1, "spread": 0.3}``
            Centre on ``best_y + optimism`` with Gaussian per-sample
            noise of std ``spread`` (the original V2 approach).

        * ``{"type": "percentile_annealing", "p_low": 0.75,
            "p_high": 0.99, "spread": 0.0}``
            Centre on a percentile of the elite buffer that anneals
            from ``p_low`` to ``p_high`` as optimisation progresses
            (measured by ``num_evals / budget``).  Always within the
            training distribution so no extrapolation.  Optional
            Gaussian ``spread`` around the centre.

        * ``{"type": "diverse_batch", "low_percentile": 0.8,
            "high_offset": 0.1}``
            Each sample in the batch receives a distinct ``y_cond``
            linearly spaced from the ``low_percentile`` of the elite
            buffer to ``best_y + high_offset``.  Creates a natural
            exploration–exploitation spectrum within each batch.

        * ``{"type": "combined", "p_low": 0.75, "p_high": 0.99,
            "low_quantile": 0.5, "high_offset": 0.1}``
            Percentile annealing determines the upper bound
            (which rises with progress) and diverse batch provides
            within-batch spread from ``low_quantile`` to the annealed
            centre + ``high_offset``.
    noise_pred_arch : str
        Architecture of the noise predictor network.  Supported:

        * ``"concat_mlp"`` — original plain MLP that concatenates
          ``[x_t, time_emb, y_cond]`` at input (baseline, no residual
          connections, conditioning only at input layer).
        * ``"film_resnet"`` — MLP with residual blocks and FiLM
          (Feature-wise Linear Modulation) conditioning.  Time+y are
          injected at every layer via learned scale+shift, preventing
          conditioning signal dilution.  LayerNorm stabilises online
          training.  Recommended default.
        * ``"adaln_resnet"`` — DiT-style MLP with Adaptive LayerNorm
          residual blocks.  The conditioning vector controls
          normalisation parameters and a learned gate.  Zero-initialised
          gates make the network start as an identity (predict zero
          noise), which is the correct inductive bias for epsilon
          prediction.  Slightly more parameters per block than FiLM.
    local_search : dict or None
        Configuration for the local search component.  ``None`` uses
        defaults (disabled).  A dict with any of these keys:

        * ``"enabled"`` (bool, default False) — master switch.
        * ``"frac"`` (float, default 0.15) — initial fraction of
          ``ask()`` candidates generated via local mutations.
        * ``"frac_max"`` (float, default 0.30) — ceiling when annealing.
        * ``"anneal"`` (bool, default True) — cosine-grow ``frac``
          toward ``frac_max`` as optimisation progresses.
        * ``"n_anchors"`` (int, default 5) — number of anchor points
          selected from the elite buffer by ``elite_filter`` score.
        * ``"sigma_init"`` (float, default 0.2) — initial mutation
          scale in normalised ``[-1, 1]`` space.
        * ``"sigma_decay"`` (str, default ``"quadratic"``) — sigma
          adaptation method: ``"quadratic"`` / ``"linear"`` (progress-
          based) or ``"one_fifth"`` (adaptive 1/5 success rule).
        * ``"use_covariance"`` (bool, default True) — if True and
          preconditioning is active, use the inverse transform to
          produce covariant mutations along learned correlation axes.
        * ``"anchor_selection"`` (str, default ``"elite_score"``) —
          reserved for future anchor selection strategies.
    rank_temperature : float
        Controls the sharpness of rank-weighted sampling (lower → more
        uniform, higher → more concentrated on the best points).
    weight_decay : float
        L2 regularisation for both optimisers.
    x_noise_std : float
        Standard deviation of Gaussian noise added to *normalised* x₀
        during training (before the diffusion noise step).  Acts as
        data augmentation / kernel smoothing to prevent the model from
        memorising individual training points.  0 disables.
    snr_loss_weighting : bool
        If ``True``, weight the per-timestep diffusion loss by
        ``min(SNR(t), snr_gamma) / SNR(t)`` to reduce dominance of
        easy high-SNR timesteps.
    snr_gamma : float
        Clamping threshold for min-SNR loss weighting.
    explore_anneal : bool
        If ``True``, cosine-anneal ``explore_frac`` from its configured
        value down to 0.01 over the optimisation budget.
    ema_decay : float
        Exponential moving average decay for the inference copy of the
        noise predictor.  Set to 0 to disable EMA (use training weights
        directly).
    reinit_interval : int
        If > 0, reinitialise model weights and optimiser state every
        this many ``tell()`` calls, then retrain from scratch on the
        current elite with ``reinit_epochs`` epochs.  0 disables.
    reinit_epochs : int
        Number of training epochs after a model reinitialisation.
        Typically larger than ``num_epochs`` to allow convergence.
    seed : int
        Random seed.
    min_data : float or int
        If <= 1, interpreted as a fraction of ``budget``.  If > 1,
        interpreted as an absolute point count.  Ignored when
        ``min_data_per_dim`` is set.
    min_data_per_dim : int or None
        When set, ``_min_data_count = min_data_per_dim × input_dim``.
        Takes precedence over ``min_data``.
    budget : int or None
        Total evaluation budget (set externally by the experiment runner
        when not provided at construction time).
    device : str or None
        Torch device.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        *,
        num_timesteps: int = 50,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        prediction_type: str = "epsilon",
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        noise_pred_arch: str = "film_resnet",
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
        depth: int = 3,
        batch_size: int = 64,
        lr_diffusion: float = 3e-4,
        lr_regressor: float = 1e-3,
        num_epochs: int = 10,
        elite_min_per_dim: int = 5,
        elite_max_per_dim: int = 100,
        elite_filter: Optional[Dict[str, Any]] = None,
        elite_adaptive: Optional[Dict[str, Any]] = None,
        guidance_strength: float = 1.0,
        use_regressor: bool = False,
        explore_frac: float = 0.05,
        conditioning_optimism: float = 0.1,
        conditioning_spread: float = 0.3,
        conditioning: Optional[Dict[str, Any]] = None,
        precondition: Optional[Dict[str, Any]] = None,
        exploration: Optional[Dict[str, Any]] = None,
        local_search: Optional[Dict[str, Any]] = None,
        inference: Optional[Dict[str, Any]] = None,
        rank_temperature: float = 0.5,
        weight_decay: float = 1e-4,
        x_noise_std: float = 0.01,
        ema_decay: float = 0.995,
        reinit_interval: int = 0,
        reinit_epochs: int = 50,
        y_norm_type: str = "rank",
        explore_anneal: bool = True,
        snr_loss_weighting: bool = True,
        snr_gamma: float = 5.0,
        seed: int = 42,
        min_data: float = 0.1,
        min_data_per_dim: Optional[int] = None,
        verbose: bool = False,
        budget: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super().__init__(
            name="DiffusionV2",
            input_dim=input_dim,
            bounds=bounds,
            verbose=verbose,
        )
        self._seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)

        self.lower = np.atleast_1d(bounds[0]).astype(np.float32)
        self.upper = np.atleast_1d(bounds[1]).astype(np.float32)
        if self.lower.size == 1:
            self.lower = np.full(input_dim, self.lower.item(), dtype=np.float32)
        if self.upper.size == 1:
            self.upper = np.full(input_dim, self.upper.item(), dtype=np.float32)

        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if elite_min_per_dim > elite_max_per_dim:
            raise ValueError(
                f"elite_min_per_dim ({elite_min_per_dim}) must be <= "
                f"elite_max_per_dim ({elite_max_per_dim})"
            )
        self.elite_min_per_dim = elite_min_per_dim
        self.elite_max_per_dim = elite_max_per_dim
        self.elite_min = max(2, elite_min_per_dim * input_dim)
        self.elite_size = max(self.elite_min, elite_max_per_dim * input_dim)

        # Resolve elite filter config
        if elite_filter is None:
            elite_filter = {"type": "quality"}
        ftype = elite_filter.get("type", "quality")
        if ftype not in _ELITE_FILTER_DEFAULTS:
            raise ValueError(
                f"Unknown elite_filter type {ftype!r}. "
                f"Supported: {list(_ELITE_FILTER_DEFAULTS)}"
            )
        merged: Dict[str, Any] = {**_ELITE_FILTER_DEFAULTS[ftype], **elite_filter}
        self.elite_filter: Dict[str, Any] = merged

        # Resolve adaptive elite config
        self.elite_adaptive: Optional[Dict[str, Any]] = elite_adaptive
        self._elite_floor: float = -np.inf
        self._elite_max_init: int = self.elite_size

        # Resolve preconditioning config
        if precondition is None:
            precondition = {"type": "none"}
        ptype = precondition.get("type", "none")
        if ptype not in _PRECONDITION_DEFAULTS:
            raise ValueError(
                f"Unknown precondition type {ptype!r}. "
                f"Supported: {list(_PRECONDITION_DEFAULTS)}"
            )
        self.precondition: Dict[str, Any] = {
            **_PRECONDITION_DEFAULTS[ptype], **precondition,
        }
        self._pc_mean: Optional[np.ndarray] = None
        self._pc_transform: Optional[np.ndarray] = None
        self._pc_inverse: Optional[np.ndarray] = None
        self._pc_cov_ema: Optional[np.ndarray] = None

        # Resolve exploration strategy config
        if exploration is None:
            exploration = {"type": "uniform"}
        etype = exploration.get("type", "uniform")
        if etype not in _EXPLORATION_DEFAULTS:
            raise ValueError(
                f"Unknown exploration type {etype!r}. "
                f"Supported: {list(_EXPLORATION_DEFAULTS)}"
            )
        self.exploration: Dict[str, Any] = {
            **_EXPLORATION_DEFAULTS[etype], **exploration,
        }

        # Resolve local search config
        if local_search is None:
            local_search = {}
        self.local_search: Dict[str, Any] = {**_LOCAL_SEARCH_DEFAULTS, **local_search}
        self._ls_sigma: float = float(self.local_search["sigma_init"])
        self._last_n_local: int = 0

        # Resolve inference scheduler config
        if inference is None:
            inference = {"type": "ddpm"}
        itype = inference.get("type", "ddpm")
        if itype not in _INFERENCE_DEFAULTS:
            raise ValueError(
                f"Unknown inference type {itype!r}. "
                f"Supported: {list(_INFERENCE_DEFAULTS)}"
            )
        self.inference: Dict[str, Any] = {
            **_INFERENCE_DEFAULTS[itype], **inference,
        }

        # Resolve conditioning strategy config
        if conditioning is None:
            conditioning = {
                "type": "optimistic",
                "optimism": conditioning_optimism,
                "spread": conditioning_spread,
            }
        ctype = conditioning.get("type", "optimistic")
        if ctype not in _CONDITIONING_DEFAULTS:
            raise ValueError(
                f"Unknown conditioning type {ctype!r}. "
                f"Supported: {list(_CONDITIONING_DEFAULTS)}"
            )
        self.conditioning: Dict[str, Any] = {
            **_CONDITIONING_DEFAULTS[ctype], **conditioning,
        }

        self.guidance_strength = guidance_strength
        self.use_regressor = use_regressor
        self.explore_frac = explore_frac
        self.conditioning_optimism = conditioning_optimism
        self.conditioning_spread = conditioning_spread
        self.rank_temperature = rank_temperature
        self.weight_decay = weight_decay
        self.x_noise_std = x_noise_std
        self.ema_decay = ema_decay
        self.reinit_interval = reinit_interval
        self.reinit_epochs = reinit_epochs
        if y_norm_type not in ("rank", "mean_std"):
            raise ValueError(
                f"Unknown y_norm_type {y_norm_type!r}. "
                f"Supported: 'rank', 'mean_std'"
            )
        self.y_norm_type = y_norm_type
        self.min_data_frac = float(min_data)
        self.min_data_per_dim = min_data_per_dim

        self.exploration_stage = True

        self.explore_anneal = explore_anneal
        self.snr_loss_weighting = snr_loss_weighting
        self.snr_gamma = snr_gamma
        self.budget: Optional[int] = budget
        self._tell_count: int = 0

        # ---- diffusers noise scheduler ----------------------------------------
        self._scheduler_config = dict(
            num_train_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        self.noise_scheduler = DDPMScheduler(**self._scheduler_config)

        # ---- networks ----------------------------------------------------------
        self.noise_pred_arch = noise_pred_arch
        self.noise_pred_net = create_noise_predictor(
            noise_pred_arch, input_dim, hidden_dim, time_embed_dim, depth=depth,
        ).to(self.device)
        self.noise_pred_opt = AdamW(
            self.noise_pred_net.parameters(), lr=lr_diffusion, weight_decay=weight_decay,
        )

        # EMA copy for stable inference
        self.ema_net: Optional[nn.Module] = None
        if self.ema_decay > 0:
            self.ema_net = copy.deepcopy(self.noise_pred_net)
            for p in self.ema_net.parameters():
                p.requires_grad_(False)

        self.regressor: Optional[_Regressor] = None
        self.regressor_opt: Optional[AdamW] = None
        if self.use_regressor:
            self.regressor = _Regressor(input_dim, hidden_dim).to(self.device)
            self.regressor_opt = AdamW(
                self.regressor.parameters(), lr=lr_regressor, weight_decay=weight_decay,
            )

        # ---- exploration engine (Sobol) -----------------------------------------
        self._sobol_engine: Optional[Any] = None
        if self.exploration["type"] == "sobol":
            from scipy.stats.qmc import Sobol
            self._sobol_engine = Sobol(
                d=input_dim, scramble=True, seed=int(seed),
            )

        # ---- data buffers (elite buffer) --------------------------------------
        self.x_data: List[np.ndarray] = []
        self.y_data: List[float] = []
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

        # Global best (never evicted, used for conditioning target)
        self.best_y: float = -np.inf
        self.best_x: Optional[np.ndarray] = None

        # Store init config for reinitialisation
        self._init_lr_diff = lr_diffusion
        self._init_lr_reg = lr_regressor
        self._init_hidden_dim = hidden_dim
        self._init_time_embed_dim = time_embed_dim
        self._init_depth = depth

        self._pc_clip_range: float = clip_sample_range

        # ---- loss logging ------------------------------------------------------
        self._loss_log_path: Optional[Path] = None
        self._loss_header_written: bool = False

    # ------------------------------------------------------------------ #
    # Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @property
    def _min_data_count(self) -> int:
        """Absolute number of points required before training starts.

        Clamped so it never exceeds ``elite_size``.  Without this,
        elite eviction in ``_update_elite`` would push ``len(x_data)``
        below the threshold every iteration, causing ``ask()`` to
        permanently fall back to random exploration.
        """
        if self.min_data_per_dim is not None:
            base = max(1, self.min_data_per_dim * self.input_dim)
        elif self.min_data_frac > 1.0:
            base = int(self.min_data_frac)
        elif self.budget is not None and self.budget > 0:
            base = max(1, int(self.min_data_frac * self.budget))
        else:
            raise ValueError("min_data_frac must be <= 1.0 or budget must be set")
        return min(base, self.elite_size)

    # -- normalisation ------------------------------------------------- #

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        """Map x from [lower, upper] to [-1, 1]."""
        return 2.0 * (x - self.lower) / (self.upper - self.lower + 1e-8) - 1.0

    def _denormalize_x(self, x_norm: np.ndarray) -> np.ndarray:
        """Map x from [-1, 1] back to [lower, upper]."""
        return self.lower + (x_norm + 1.0) / 2.0 * (self.upper - self.lower + 1e-8)

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def _update_normalization(self) -> None:
        """Recompute y-statistics from the elite buffer only."""
        if len(self.y_data) < 2:
            self.y_mean = 0.0
            self.y_std = 1.0
            return
        y_arr = np.array(self.y_data, dtype=np.float32)
        finite = y_arr[np.isfinite(y_arr)]
        if finite.size < 2:
            self.y_mean = 0.0
            self.y_std = 1.0
            return
        self.y_mean = float(np.mean(finite))
        self.y_std = float(np.std(finite) + 1e-8)

    # -- x-space preconditioning --------------------------------------- #

    def _update_precondition(self) -> None:
        """Recompute the preconditioning transform from the elite buffer."""
        ptype = self.precondition["type"]
        if ptype == "none":
            return
        if len(self.x_data) < 2:
            return

        x_arr = np.array(self.x_data, dtype=np.float64)
        x_norm = self._normalize_x(x_arr)
        n, d = x_norm.shape
        mean = np.mean(x_norm, axis=0)
        x_c = x_norm - mean

        if ptype == "standardize":
            std = np.std(x_c, axis=0) + 1e-8
            self._pc_mean = mean
            self._pc_transform = np.diag(1.0 / std)
            self._pc_inverse = np.diag(std)
            return

        if n <= d and ptype == "whiten":
            warnings.warn(
                f"precondition='whiten' with n={n} <= d={d}; "
                f"falling back to 'standardize'.",
                stacklevel=2,
            )
            std = np.std(x_c, axis=0) + 1e-8
            self._pc_mean = mean
            self._pc_transform = np.diag(1.0 / std)
            self._pc_inverse = np.diag(std)
            return

        if ptype == "whiten":
            cov = x_c.T @ x_c / n
        elif ptype in ("whiten_shrink", "whiten_ema"):
            cov = _ledoit_wolf_shrinkage(x_c)
        else:
            return

        if ptype == "whiten_ema":
            eta = float(self.precondition.get("ema_rate", 0.1))
            if self._pc_cov_ema is None:
                self._pc_cov_ema = cov
            else:
                self._pc_cov_ema = (1.0 - eta) * self._pc_cov_ema + eta * cov
            cov = self._pc_cov_ema.copy()

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        sqrt_diag = np.diag(np.sqrt(eigvals))
        self._pc_mean = mean
        self._pc_transform = inv_sqrt @ eigvecs.T   # (d, d)
        self._pc_inverse = eigvecs @ sqrt_diag       # (d, d)

    def _precondition_x(self, x_norm: np.ndarray) -> np.ndarray:
        """Apply preconditioning: box-normalised x -> preconditioned z."""
        if self._pc_transform is None:
            return x_norm
        return (x_norm - self._pc_mean) @ self._pc_transform.T

    def _unprecondition_x(self, z: np.ndarray) -> np.ndarray:
        """Inverse preconditioning: preconditioned z -> box-normalised x."""
        if self._pc_inverse is None:
            return z
        return z @ self._pc_inverse.T + self._pc_mean

    # -- initial exploration ------------------------------------------- #

    def _sample_initial(self, n: int) -> np.ndarray:
        """Generate space-filling initial samples before training starts."""
        etype = self.exploration["type"]

        if etype == "sobol" and self._sobol_engine is not None:
            raw = self._sobol_engine.random(n)
            return (self.lower + raw * (self.upper - self.lower)).astype(
                np.float32
            )

        if etype == "lhs":
            from scipy.stats.qmc import LatinHypercube
            sampler = LatinHypercube(d=self.input_dim, seed=self.rng)
            raw = sampler.random(n)
            return (self.lower + raw * (self.upper - self.lower)).astype(
                np.float32
            )

        return self._sample_uniform(n)

    # -- local search -------------------------------------------------- #

    def _generate_local_samples(self, n: int) -> np.ndarray:
        """Generate *n* candidates via mutation of anchor points.

        Anchors are selected from the elite buffer using the same scoring
        function as ``elite_filter``.  Mutations are optionally covariant
        (using the preconditioning inverse matrix) so the perturbation
        follows the learned correlation structure.

        Parameters
        ----------
        n : int
            Number of local search samples to generate.

        Returns
        -------
        np.ndarray, shape ``(n, input_dim)``
            Candidates clipped to ``[lower, upper]``.
        """
        if len(self.y_data) == 0:
            return self._sample_uniform(n)

        ls = self.local_search
        n_anchors = min(max(int(ls["n_anchors"]), 1), len(self.y_data))

        # Compute scores once and reuse for both anchor selection and weighting
        scores = self._compute_elite_scores()
        top_idx = np.argpartition(scores, -n_anchors)[-n_anchors:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        anchor_idx = top_idx.tolist()

        anchor_scores = np.array([scores[i] for i in anchor_idx], dtype=np.float64)
        # Shift to positive for softmax-like weighting
        anchor_scores -= anchor_scores.min()
        total = anchor_scores.sum()
        if total < 1e-12:
            # Uniform distribution across anchors
            counts = np.full(len(anchor_idx), n // len(anchor_idx), dtype=int)
            counts[0] += n - counts.sum()
        else:
            fracs = anchor_scores / total
            counts = np.round(fracs * n).astype(int)
            # Fix rounding: adjust the best anchor
            counts[0] += n - counts.sum()

        sigma = self._ls_sigma
        use_cov = bool(ls["use_covariance"]) and self._pc_inverse is not None

        candidates = []
        for idx, count in zip(anchor_idx, counts):
            if count <= 0:
                continue
            x_anchor = self.x_data[idx]  # original space
            x_norm = self._normalize_x(x_anchor)  # [-1, 1]

            noise = self.rng.standard_normal((count, self.input_dim)).astype(np.float64)
            if use_cov:
                # Covariant mutations along learned correlation axes
                noise = noise @ self._pc_inverse.T
            noise *= sigma

            x_new_norm = x_norm + noise
            x_new = self._denormalize_x(x_new_norm.astype(np.float32))
            x_new = np.clip(x_new, self.lower, self.upper)
            candidates.append(x_new)

        return np.concatenate(candidates, axis=0) if candidates else self._sample_uniform(n)

    def _update_ls_sigma(self, y_batch: np.ndarray, prev_best_y: float) -> None:
        """Adapt local search sigma.

        For ``"one_fifth"`` mode: if more than 20% of the batch
        evaluations improved upon ``prev_best_y`` (the elite best
        *before* ingestion), increase sigma; otherwise shrink it.

        For ``"quadratic"`` / ``"linear"`` modes the sigma is a
        deterministic function of optimisation progress.

        Parameters
        ----------
        y_batch : np.ndarray
            Negated objective values of the batch (maximisation convention).
        prev_best_y : float
            Best elite y *before* this batch was ingested.
        """
        if len(y_batch) == 0:
            return

        ls = self.local_search
        decay_type = str(ls["sigma_decay"])

        if decay_type == "one_fifth":
            n_success = int(np.sum(y_batch > prev_best_y))
            ratio = n_success / len(y_batch)
            if ratio > 0.2:
                self._ls_sigma *= 1.2
            else:
                self._ls_sigma *= 0.82  # ≈ (1/1.2)
        elif decay_type == "quadratic":
            progress = self._optimization_progress
            sigma_init = float(ls["sigma_init"])
            self._ls_sigma = sigma_init * (1.0 - progress) ** 2
        else:  # "linear"
            progress = self._optimization_progress
            sigma_init = float(ls["sigma_init"])
            self._ls_sigma = sigma_init * max(1.0 - progress, 0.01)

        # Clamp sigma to reasonable bounds
        self._ls_sigma = max(self._ls_sigma, 1e-6)
        self._ls_sigma = min(self._ls_sigma, 2.0)

    # -- elite buffer management --------------------------------------- #

    @staticmethod
    def _knn_mean_distances(x: np.ndarray, k: int) -> np.ndarray:
        """Mean Euclidean distance to *k* nearest neighbours.

        Parameters
        ----------
        x : np.ndarray, shape ``(n, d)``
            Points (should already be in normalised space).
        k : int
            Number of neighbours.  Clamped to ``n − 1``.

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        n = x.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=np.float64)
        k_eff = min(k, n - 1)
        # Pairwise L2 distances — O(n²d), fine for n ≤ ~1000
        diff = x[:, None, :] - x[None, :, :]          # (n, n, d)
        dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)  # (n, n)
        np.fill_diagonal(dists, np.inf)
        # k smallest per row
        knn_idx = np.argpartition(dists, k_eff, axis=1)[:, :k_eff]
        knn_dists = np.take_along_axis(dists, knn_idx, axis=1)
        return np.mean(knn_dists, axis=1)

    def _compute_elite_scores(self) -> np.ndarray:
        """Score every point in the buffer according to *elite_filter*.

        Higher score = more likely to be retained.
        """
        y_arr = np.array(self.y_data, dtype=np.float64)
        ftype = self.elite_filter["type"]

        if ftype == "quality":
            return y_arr

        if ftype == "quality_knn":
            k: int = int(self.elite_filter["k"])
            dw: float = float(self.elite_filter["diversity_weight"])

            x_arr = np.array(self.x_data, dtype=np.float32)
            x_norm = self._normalize_x(x_arr)
            x_pc = self._precondition_x(x_norm)

            y_min, y_max = y_arr.min(), y_arr.max()
            y_01 = (y_arr - y_min) / (y_max - y_min + 1e-12)

            diversity = self._knn_mean_distances(x_pc, k)
            d_min, d_max = diversity.min(), diversity.max()
            d_01 = (diversity - d_min) / (d_max - d_min + 1e-12)

            return y_01 + dw * d_01

        if ftype == "crowding":
            n_tiers: int = int(self.elite_filter.get("n_tiers", 10))
            n = len(y_arr)
            x_arr = np.array(self.x_data, dtype=np.float32)
            x_norm = self._normalize_x(x_arr)
            x_pc = self._precondition_x(x_norm)

            rank_order = np.argsort(-y_arr)
            tier_ids = np.zeros(n, dtype=np.int64)
            pts_per_tier = max(1, n // n_tiers)
            for i, idx in enumerate(rank_order):
                tier_ids[idx] = i // pts_per_tier

            crowding = np.zeros(n, dtype=np.float64)
            d = x_pc.shape[1]
            for dim in range(d):
                sorted_idx = np.argsort(x_pc[:, dim])
                crowding[sorted_idx[0]] += 1e12
                crowding[sorted_idx[-1]] += 1e12
                span = x_pc[sorted_idx[-1], dim] - x_pc[sorted_idx[0], dim]
                if span < 1e-12:
                    continue
                for j in range(1, n - 1):
                    crowding[sorted_idx[j]] += (
                        x_pc[sorted_idx[j + 1], dim]
                        - x_pc[sorted_idx[j - 1], dim]
                    ) / span

            cd_min, cd_max = crowding.min(), crowding.max()
            cd_01 = (crowding - cd_min) / (cd_max - cd_min + 1e-12)
            max_tier = tier_ids.max() + 1
            score = (max_tier - tier_ids).astype(np.float64) * (cd_01.max() + 1.0) + cd_01
            return score

        if ftype == "grid":
            cells_per_dim: int = int(self.elite_filter.get("cells_per_dim", 5))
            n = len(y_arr)
            x_arr = np.array(self.x_data, dtype=np.float32)
            x_norm = self._normalize_x(x_arr)
            d = x_norm.shape[1]

            cell_idx = np.clip(
                np.floor((x_norm + 1.0) / 2.0 * cells_per_dim).astype(np.int64),
                0,
                cells_per_dim - 1,
            )

            if d <= 5:
                mults = np.array(
                    [cells_per_dim ** i for i in range(d)], dtype=np.int64
                )
                cell_keys = cell_idx @ mults
            else:
                cell_keys = np.array([
                    int(hashlib.md5(row.tobytes()).hexdigest(), 16)
                    % (self.elite_size * 2)
                    for row in cell_idx
                ], dtype=np.int64)

            scores = np.full(n, -1e18, dtype=np.float64)
            cell_best: Dict[int, Tuple[int, float]] = {}
            for i in range(n):
                ck = int(cell_keys[i])
                yv = float(y_arr[i])
                if ck not in cell_best or yv > cell_best[ck][1]:
                    cell_best[ck] = (i, yv)

            for _ck, (idx, yv) in cell_best.items():
                scores[idx] = yv + 1e12

            for i in range(n):
                if scores[i] < 0:
                    scores[i] = y_arr[i]

            return scores

        raise ValueError(f"Unknown elite_filter type: {ftype!r}")

    def _update_elite(self) -> None:
        """Evict lowest-scoring points if buffer exceeds the target size.

        When ``elite_adaptive`` is enabled, the target size shrinks
        linearly from ``_elite_max_init`` to ``elite_min`` over the
        optimisation budget (LSHADE-style), and a soft quality floor
        prevents the buffer from retaining very poor points.

        After eviction, the global best point is guaranteed to be
        present in the buffer (pinned).
        """
        n = len(self.y_data)

        if self.elite_adaptive and self.elite_adaptive.get("enabled"):
            progress = self._optimization_progress
            elite_cap = self._elite_max_init - int(
                (self._elite_max_init - self.elite_min) * progress
            )
            elite_cap = max(elite_cap, self.elite_min)

            delta = float(self.elite_adaptive.get("floor_decay", 0.01))
            p = float(self.elite_adaptive.get("floor_percentile", 5))
            y_arr = np.array(self.y_data, dtype=np.float64)
            pct_val = float(np.percentile(y_arr, p))
            self._elite_floor = max(
                self._elite_floor * (1.0 - delta),
                pct_val,
            )

            above = [
                i for i, yv in enumerate(self.y_data) if yv >= self._elite_floor
            ]
            if len(above) < self.elite_min:
                above = list(range(n))

            if len(above) > elite_cap:
                sub_scores = self._compute_elite_scores()
                sub_scores_above = np.array([sub_scores[i] for i in above])
                keep_count = elite_cap
                top_local = np.argpartition(sub_scores_above, -keep_count)[-keep_count:]
                keep_idx = [above[j] for j in top_local]
            else:
                keep_idx = above

            if len(keep_idx) < n:
                self.x_data = [self.x_data[i] for i in keep_idx]
                self.y_data = [self.y_data[i] for i in keep_idx]

        elif n > self.elite_size:
            scores = self._compute_elite_scores()
            keep_idx = np.argpartition(scores, -self.elite_size)[-self.elite_size:]
            self.x_data = [self.x_data[i] for i in keep_idx]
            self.y_data = [self.y_data[i] for i in keep_idx]

        # Pin the global best: ensure it is always in the elite buffer
        self._pin_global_best()

    def _pin_global_best(self) -> None:
        """Ensure the global best point is present in the elite buffer.

        Called as the final step of ``_update_elite``.  If the best
        point was evicted (e.g. by a diversity-based filter), it is
        appended back to the buffer so the model always trains on it.
        """
        if self.best_x is None or not np.isfinite(self.best_y):
            return
        if len(self.y_data) == 0:
            self.x_data.append(self.best_x.copy())
            self.y_data.append(float(self.best_y))
            return
        buffer_best = max(self.y_data)
        if buffer_best < self.best_y:
            self.x_data.append(self.best_x.copy())
            self.y_data.append(float(self.best_y))

    # -- rank-weighted sampling ---------------------------------------- #

    def _compute_rank_weights(self, y_values: np.ndarray) -> torch.Tensor:
        """Rank-based sampling weights (exponential decay by rank).

        Returns a probability vector of shape ``(n,)`` where the best
        point (highest y) has the highest weight.
        """
        n = len(y_values)
        if n <= 1:
            return torch.ones(max(n, 1), dtype=torch.float32, device=self.device)
        ranks = np.argsort(np.argsort(-y_values)).astype(np.float64)
        temp = max(self.rank_temperature * n, 1.0)
        weights = np.exp(-ranks / temp)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    # -- optimisation progress ----------------------------------------- #

    @property
    def _optimization_progress(self) -> float:
        """Fraction of budget consumed, clamped to [0, 1]."""
        if self.budget is not None and self.budget > 0:
            return min(self.num_evals / self.budget, 1.0)
        return min(self.num_evals / 10000.0, 1.0)

    # -- conditioning target ------------------------------------------- #

    def _get_conditioning_target(self, n: int = 1) -> torch.Tensor:
        """Normalised conditioning values for the reverse process.

        Dispatches to the strategy specified by ``self.conditioning["type"]``.

        Supported types
        ---------------
        * ``"optimistic"`` — centre on ``best_y + optimism``, Gaussian spread.
        * ``"percentile_annealing"`` — centre on annealing percentile of
          the elite buffer, with optional Gaussian spread.
        * ``"diverse_batch"`` — each sample gets a different y_cond
          spread uniformly between a lower percentile and ``best_y + offset``.
        * ``"combined"`` — annealed percentile centre (from
          *percentile_annealing*) with batch diversity (from *diverse_batch*).
        """
        ctype = self.conditioning["type"]
        if ctype == "optimistic":
            return self._cond_optimistic(n)
        if ctype == "percentile_annealing":
            return self._cond_percentile_annealing(n)
        if ctype == "diverse_batch":
            return self._cond_diverse_batch(n)
        if ctype == "combined":
            return self._cond_combined(n)
        raise ValueError(f"Unknown conditioning type: {ctype!r}")

    def _cond_optimistic(self, n: int) -> torch.Tensor:
        """Original approach: best_y + optimism + Gaussian spread."""
        if self.y_norm_type == "rank":
            centre = 1.0 + float(self.conditioning["optimism"])
        else:
            best_y_norm = float(self._normalize_y(np.array([self.best_y]))[0])
            centre = best_y_norm + float(self.conditioning["optimism"])
        spread = float(self.conditioning["spread"])
        if spread > 0 and n > 1:
            noise = torch.randn(n, device=self.device) * spread
            return torch.full((n,), centre, device=self.device) + noise
        return torch.full((n,), centre, device=self.device)

    def _cond_percentile_annealing(self, n: int) -> torch.Tensor:
        """Centre on an annealing percentile of elite y, rising with progress."""
        progress = self._optimization_progress
        p_low = float(self.conditioning["p_low"])
        p_high = float(self.conditioning["p_high"])
        p = p_low + (p_high - p_low) * progress

        if self.y_norm_type == "rank":
            center_norm = p
        else:
            y_arr = np.array(self.y_data, dtype=np.float64)
            center_raw = float(np.percentile(y_arr, p * 100.0))
            center_norm = float(self._normalize_y(np.array([center_raw]))[0])

        spread = float(self.conditioning.get("spread", 0.0))
        if spread > 0 and n > 1:
            noise = torch.randn(n, device=self.device) * spread
            return torch.full((n,), center_norm, device=self.device) + noise
        return torch.full((n,), center_norm, device=self.device)

    def _cond_diverse_batch(self, n: int) -> torch.Tensor:
        """Each sample gets a different y_cond spanning a quality range."""
        low_pct = float(self.conditioning["low_percentile"])
        high_offset = float(self.conditioning["high_offset"])

        if self.y_norm_type == "rank":
            low_norm = low_pct
            high_norm = 1.0 + high_offset
        else:
            y_arr = np.array(self.y_data, dtype=np.float64)
            low_raw = float(np.percentile(y_arr, low_pct * 100.0))
            low_norm = float(self._normalize_y(np.array([low_raw]))[0])
            best_norm = float(self._normalize_y(np.array([self.best_y]))[0])
            high_norm = best_norm + high_offset

        if low_norm >= high_norm:
            return torch.full((n,), high_norm, device=self.device)
        if n == 1:
            mid = (low_norm + high_norm) / 2.0
            return torch.tensor([mid], dtype=torch.float32, device=self.device)
        vals = np.linspace(low_norm, high_norm, n).astype(np.float32)
        return torch.tensor(vals, device=self.device)

    def _cond_combined(self, n: int) -> torch.Tensor:
        """Annealed percentile centre + diverse batch spread."""
        progress = self._optimization_progress

        p_low = float(self.conditioning["p_low"])
        p_high = float(self.conditioning["p_high"])
        p = p_low + (p_high - p_low) * progress

        low_q = float(self.conditioning["low_quantile"])
        high_offset = float(self.conditioning["high_offset"])

        if self.y_norm_type == "rank":
            center_norm = p
            low_norm = min(low_q, center_norm)
            high_norm = center_norm + high_offset
        else:
            y_arr = np.array(self.y_data, dtype=np.float64)
            center_raw = float(np.percentile(y_arr, p * 100.0))
            center_norm = float(self._normalize_y(np.array([center_raw]))[0])
            low_raw = float(np.percentile(y_arr, low_q * 100.0))
            low_norm = float(self._normalize_y(np.array([low_raw]))[0])
            high_norm = center_norm + high_offset
            low_norm = min(low_norm, center_norm)

        if low_norm >= high_norm:
            return torch.full((n,), high_norm, device=self.device)
        if n == 1:
            return torch.tensor([center_norm], dtype=torch.float32, device=self.device)
        vals = np.linspace(low_norm, high_norm, n).astype(np.float32)
        return torch.tensor(vals, device=self.device)

    # -- EMA update ---------------------------------------------------- #

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Update the EMA copy of the noise predictor."""
        if self.ema_net is None or self.ema_decay <= 0:
            return
        for p_ema, p_train in zip(self.ema_net.parameters(), self.noise_pred_net.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p_train.data, alpha=1.0 - self.ema_decay)

    def _inference_net(self) -> nn.Module:
        """Return the network used for inference (EMA if available)."""
        if self.ema_net is not None and self.ema_decay > 0:
            return self.ema_net
        return self.noise_pred_net

    # -- model reinitialisation ---------------------------------------- #

    def _reinit_model(self) -> None:
        """Reset model weights and optimiser state from scratch."""
        self.noise_pred_net = create_noise_predictor(
            self.noise_pred_arch, self.input_dim,
            self._init_hidden_dim, self._init_time_embed_dim,
            depth=self._init_depth,
        ).to(self.device)
        self.noise_pred_opt = AdamW(
            self.noise_pred_net.parameters(),
            lr=self._init_lr_diff,
            weight_decay=self.weight_decay,
        )
        if self.ema_net is not None:
            self.ema_net = copy.deepcopy(self.noise_pred_net)
            for p in self.ema_net.parameters():
                p.requires_grad_(False)
        if self.regressor is not None:
            for layer in self.regressor.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            self.regressor_opt = AdamW(
                self.regressor.parameters(),
                lr=self._init_lr_reg,
                weight_decay=self.weight_decay,
            )

    # -- sampling helpers ---------------------------------------------- #

    def _sample_uniform(self, n: int) -> np.ndarray:
        return self.rng.uniform(
            self.lower, self.upper, size=(n, self.input_dim),
        ).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #

    def _train_networks(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        epochs: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """Run SGD on the elite buffer with rank weighting.

        Parameters
        ----------
        epochs : int or None
            Override for the number of training epochs.  When ``None``,
            ``self.num_epochs`` is used.

        Returns
        -------
        diffusion_losses, regressor_losses : tuple of list[float]
            Per-step loss values for logging.
        """
        diffusion_losses: List[float] = []
        regressor_losses: List[float] = []
        n_data = x_train.shape[0]
        if n_data == 0:
            return diffusion_losses, regressor_losses

        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        effective_batch = min(self.batch_size, n_data)

        steps_per_epoch = max(1, n_data // effective_batch)
        n_epochs = epochs if epochs is not None else self.num_epochs
        n_epochs = max(n_epochs, 1)
        n_steps = steps_per_epoch * n_epochs

        _ES_WINDOW = 10
        progress = self._optimization_progress
        # Early: aggressive ES (data is volatile); Late: patient ES (fine-tuning)
        _ES_PATIENCE = int(5 + 15 * progress)       # 5 → 20
        _ES_REL_DELTA = 0.02 - 0.015 * progress     # 0.02 → 0.005
        best_avg_loss = float('inf')
        no_improve = 0
        es_losses: List[float] = []
        min_steps_for_es = max(2 * steps_per_epoch, _ES_WINDOW)

        snr_weights_all: Optional[torch.Tensor] = None
        if self.snr_loss_weighting:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
            snr_all = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8)
            snr_weights_all = torch.clamp(snr_all, max=self.snr_gamma) / (snr_all + 1e-8)

        self.noise_pred_net.train()
        if self.use_regressor and self.regressor is not None:
            self.regressor.train()

        for step_idx in range(n_steps):
            if weights is not None and n_data > effective_batch:
                idx = torch.multinomial(weights, effective_batch, replacement=True)
            else:
                idx = torch.randint(0, n_data, (effective_batch,), device=self.device)

            x0 = x_train[idx]
            if self.x_noise_std > 0:
                x0 = x0 + self.x_noise_std * torch.randn_like(x0)
            y_cond = y_train[idx]

            noise = torch.randn_like(x0)
            timesteps = torch.randint(
                0, num_train_timesteps, (effective_batch,),
                device=self.device,
            ).long()

            x_t = self.noise_scheduler.add_noise(x0, noise, timesteps)
            eps_pred = self.noise_pred_net(x_t, timesteps, y_cond)

            if self.snr_loss_weighting and snr_weights_all is not None:
                per_sample_loss = torch.mean((eps_pred - noise) ** 2, dim=-1)
                loss = torch.mean(per_sample_loss * snr_weights_all[timesteps])
            else:
                loss = torch.mean((eps_pred - noise) ** 2)

            self.noise_pred_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.noise_pred_net.parameters(), max_norm=1.0)
            self.noise_pred_opt.step()

            loss_val = float(loss.item())
            diffusion_losses.append(loss_val)
            self._update_ema()

            es_losses.append(loss_val)
            if len(es_losses) > _ES_WINDOW:
                es_losses.pop(0)
            if step_idx >= min_steps_for_es and len(es_losses) == _ES_WINDOW:
                avg_loss = sum(es_losses) / _ES_WINDOW
                if avg_loss < best_avg_loss * (1.0 - _ES_REL_DELTA):
                    best_avg_loss = avg_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= _ES_PATIENCE:
                        break

            # -- regressor -----------------------------------------------
            if self.use_regressor and self.regressor is not None and self.regressor_opt is not None:
                if weights is not None and n_data > effective_batch:
                    reg_idx = torch.multinomial(weights, effective_batch, replacement=True)
                else:
                    reg_idx = torch.randint(0, n_data, (effective_batch,), device=self.device)

                self.regressor_opt.zero_grad()
                y_pred = self.regressor(x_train[reg_idx])
                reg_loss = torch.mean((y_pred - y_train[reg_idx]) ** 2)
                reg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=1.0)
                self.regressor_opt.step()
                regressor_losses.append(float(reg_loss.item()))

        return diffusion_losses, regressor_losses

    # ------------------------------------------------------------------ #
    # Ask / Tell interface                                               #
    # ------------------------------------------------------------------ #

    def ask(self, n: int = 1) -> np.ndarray:
        if self.exploration_stage:
            x_init = self._sample_initial(n)
            self._register_asked(x_init)
            return x_init

        progress = self._optimization_progress

        # -- compute fractions for the 3-component pipeline ---------------
        #   explore: cosine decay from explore_frac → 0.01
        #   local:   cosine growth from local_frac → local_frac_max
        #   exploit: remainder

        if self.explore_anneal:
            explore_f = self.explore_frac * 0.5 * (1.0 + math.cos(math.pi * progress))
            explore_f = max(explore_f, 0.01)
        else:
            explore_f = self.explore_frac

        ls_enabled = bool(self.local_search["enabled"])
        if ls_enabled:
            ls_frac_init = float(self.local_search["frac"])
            ls_frac_max = float(self.local_search["frac_max"])
            if self.local_search["anneal"]:
                # Cosine growth from frac → frac_max
                local_f = ls_frac_init + (ls_frac_max - ls_frac_init) * 0.5 * (
                    1.0 - math.cos(math.pi * progress)
                )
            else:
                local_f = ls_frac_init
        else:
            local_f = 0.0

        # Ensure fractions don't exceed 1.0; clip explore first, then local
        if explore_f + local_f > 0.95:
            scale = 0.95 / (explore_f + local_f)
            explore_f *= scale
            local_f *= scale

        n_explore = int(n * explore_f)
        n_local = int(n * local_f)
        n_exploit = n - n_explore - n_local
        # Safety: n_exploit must be non-negative
        if n_exploit < 0:
            n_local += n_exploit  # reduce local
            n_exploit = 0

        candidates: List[np.ndarray] = []

        # -- exploitation: reverse-diffusion samples ---------------------
        if n_exploit > 0:
            net = self._inference_net()
            net.eval()
            if self.use_regressor and self.regressor is not None:
                self.regressor.eval()

            inf_cfg = {**self._scheduler_config,
                       "clip_sample_range": self._pc_clip_range}
            use_ddim = self.inference["type"] == "ddim"
            if use_ddim:
                scheduler = DDIMScheduler(**inf_cfg)
                ddim_eta = float(self.inference.get("eta", 0.0))
            else:
                scheduler = DDPMScheduler(**inf_cfg)
            scheduler.set_timesteps(self.num_timesteps, device=self.device)

            x_t = torch.randn(n_exploit, self.input_dim, device=self.device)
            y_cond = self._get_conditioning_target(n_exploit)

            for t in scheduler.timesteps:
                t_batch = t.expand(n_exploit)

                with torch.no_grad():
                    eps_theta = net(x_t, t_batch, y_cond)

                eps_guided = eps_theta
                if self.use_regressor and self.regressor is not None:
                    x_t_grad = x_t.detach().requires_grad_(True)
                    y_pred = self.regressor(x_t_grad)
                    grad = torch.autograd.grad(y_pred.sum(), x_t_grad)[0]
                    eps_guided = eps_theta - self.guidance_strength * grad

                with torch.no_grad():
                    if use_ddim:
                        scheduler_output = scheduler.step(
                            eps_guided, int(t), x_t, eta=ddim_eta,
                        )
                    else:
                        scheduler_output = scheduler.step(
                            eps_guided, int(t), x_t,
                        )
                    x_t = scheduler_output.prev_sample

            x_np_pc = x_t.detach().cpu().numpy().astype(np.float64)
            x_np_norm = self._unprecondition_x(x_np_pc).astype(np.float32)
            x_np = self._denormalize_x(x_np_norm)
            x_np = np.clip(x_np, self.lower, self.upper)

            bad_mask = ~np.all(np.isfinite(x_np), axis=1)
            if np.any(bad_mask):
                x_np[bad_mask] = self._sample_uniform(int(bad_mask.sum()))

            candidates.append(x_np)

        # -- local search: mutations of anchor points --------------------
        if n_local > 0:
            x_local = self._generate_local_samples(n_local)
            self._last_n_local = n_local  # track for sigma update in tell()
            candidates.append(x_local)
        else:
            self._last_n_local = 0

        # -- exploration: strategy-aware sampling ------------------------
        if n_explore > 0:
            candidates.append(self._sample_initial(n_explore))

        x_out = np.concatenate(candidates, axis=0) if len(candidates) > 1 else candidates[0]

        self.rng.shuffle(x_out)

        self._register_asked(x_out)
        return x_out

    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y_obj = np.asarray(y, dtype=np.float32).reshape(-1)
        y = -y_obj  # minimisation → maximisation

        prev_best_y = self.best_y  # snapshot for 1/5 rule

        for xi, yi in zip(x, y):
            self.num_evals += 1
            if not (np.all(np.isfinite(xi)) and np.isfinite(yi)):
                continue
            self.x_data.append(xi.copy())
            self.y_data.append(float(yi))
            if yi > self.best_y:
                self.best_y = float(yi)
                self.best_x = xi.copy()

        if self.exploration_stage:
            if len(self.x_data) < self._min_data_count:
                self._register_told(x, y_obj, tau=np.nan)
                return

            self.exploration_stage = False

        self._tell_count += 1

        # Evict worst points, then update stats
        self._update_elite()
        self._update_normalization()
        self._update_precondition()

        # Update local search sigma
        if self.local_search["enabled"]:
            self._update_ls_sigma(y, prev_best_y)

        # Periodic reinitialisation
        do_reinit = (
            self.reinit_interval > 0
            and self._tell_count % self.reinit_interval == 0
        )
        if do_reinit:
            self._reinit_model()

        # Prepare normalised training tensors
        x_arr = np.array(self.x_data, dtype=np.float32)
        y_arr = np.array(self.y_data, dtype=np.float32)

        finite_mask = np.isfinite(y_arr) & np.all(np.isfinite(x_arr), axis=1)
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]

        if x_arr.shape[0] == 0:
            self._register_told(x, y_obj, tau=np.nan)
            return

        x_norm = self._normalize_x(x_arr)
        x_pc = self._precondition_x(x_norm.astype(np.float64)).astype(np.float32)

        if self._pc_transform is not None:
            self._pc_clip_range = max(
                float(self._scheduler_config["clip_sample_range"]),
                float(np.max(np.abs(x_pc))) + 0.1,
            )
        else:
            self._pc_clip_range = float(self._scheduler_config["clip_sample_range"])

        if self.y_norm_type == "rank":
            n_pts = len(y_arr)
            if n_pts <= 1:
                y_normalized = np.ones(max(n_pts, 1), dtype=np.float32)
            else:
                order = np.argsort(y_arr)
                ranks = np.empty(n_pts, dtype=np.float32)
                ranks[order] = np.arange(n_pts, dtype=np.float32)
                y_normalized = ranks / (n_pts - 1)

            if self.conditioning.get("aug", False) and n_pts > 1:
                ctype = self.conditioning["type"]
                if ctype == "optimistic":
                    alpha = float(self.conditioning.get("optimism", 0.1))
                elif ctype == "percentile_annealing":
                    alpha = 1.0 - float(self.conditioning.get("p_high", 0.99))
                elif ctype in ("diverse_batch", "combined"):
                    alpha = float(self.conditioning.get("high_offset", 0.1))
                else:
                    alpha = 0.1
                alpha = max(alpha, 0.05)
                top_mask = y_normalized >= 0.8
                n_top = int(top_mask.sum())
                if n_top > 0:
                    noise = self.rng.uniform(0.0, alpha, size=n_top).astype(
                        np.float32
                    )
                    y_normalized[top_mask] = y_normalized[top_mask] + noise
        else:
            y_normalized = self._normalize_y(y_arr)

        x_train = _to_tensor(x_pc, self.device)
        y_train = _to_tensor(y_normalized, self.device)

        weights = self._compute_rank_weights(y_arr)
        reinit_ep = self.reinit_epochs if do_reinit else None
        d_losses, r_losses = self._train_networks(x_train, y_train, weights, epochs=reinit_ep)
        self._log_training_losses(d_losses, r_losses)

        # Report the elite-buffer floor as "tau" for verbose logging
        tau_val = float(min(self.y_data)) if self.y_data else np.nan
        self._register_told(x, y_obj, tau=tau_val)

    # ------------------------------------------------------------------ #
    # Training loss logging                                              #
    # ------------------------------------------------------------------ #

    def _log_training_losses(
        self,
        diffusion_losses: List[float],
        regressor_losses: List[float],
    ) -> None:
        """Append per-step losses to a CSV in the verbose_logs directory."""
        if not self.verbose or not diffusion_losses:
            return

        if self._loss_log_path is None:
            safe_name = self.name.lower().replace(" ", "_")
            ts = time.strftime("%Y%m%d_%H%M%S")
            unique = time.time_ns()
            tag = f"_{self._verbose_problem_tag}" if self._verbose_problem_tag else ""
            out_dir = self._verbose_log_dir or (Path("outputs") / "verbose_logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            self._loss_log_path = out_dir / f"{safe_name}{tag}_{ts}_{unique}_losses.csv"
            self._loss_header_written = False

        fieldnames = ["tell_count", "step", "diffusion_loss"]
        if self.use_regressor:
            fieldnames.append("regressor_loss")

        rows: List[Dict[str, Any]] = []
        for step, d_loss in enumerate(diffusion_losses):
            row: Dict[str, Any] = {
                "tell_count": self._tell_count,
                "step": step,
                "diffusion_loss": d_loss,
            }
            if self.use_regressor and step < len(regressor_losses):
                row["regressor_loss"] = regressor_losses[step]
            rows.append(row)

        mode = "a" if self._loss_header_written else "w"
        with self._loss_log_path.open(mode, newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not self._loss_header_written:
                writer.writeheader()
                self._loss_header_written = True
            writer.writerows(rows)

    # ------------------------------------------------------------------ #
    # Warm-start                                                         #
    # ------------------------------------------------------------------ #

    def warm_start(self, x: np.ndarray, y: np.ndarray) -> None:
        """Seed the elite buffer with initial observations.

        Data is stored with the min→max negation (matching ``tell()``)
        but without incrementing ``num_evals`` or triggering training.
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.atleast_1d(np.asarray(y, dtype=np.float32)).flatten()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y = -y
        for xi, yi in zip(x, y):
            if not (np.all(np.isfinite(xi)) and np.isfinite(yi)):
                continue
            self.x_data.append(xi.copy())
            self.y_data.append(float(yi))
            if yi > self.best_y:
                self.best_y = float(yi)
                self.best_x = xi.copy()
        self._update_elite()

    # ------------------------------------------------------------------ #
    # Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.x_data = []
        self.y_data = []
        self.num_evals = 0
        self._tell_count = 0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.best_y = -np.inf
        self.best_x = None

        self.exploration_stage = True

        self._pc_mean = None
        self._pc_transform = None
        self._pc_inverse = None
        self._pc_cov_ema = None
        self._pc_clip_range = float(self._scheduler_config["clip_sample_range"])

        self._elite_floor = -np.inf

        # Reset local search sigma
        self._ls_sigma = float(self.local_search["sigma_init"])
        self._last_n_local = 0

        if self.exploration["type"] == "sobol":
            from scipy.stats.qmc import Sobol
            self._sobol_engine = Sobol(
                d=self.input_dim, scramble=True, seed=self._seed,
            )

        self._reinit_model()
        self._reset_verbose_trace()
        self._loss_log_path = None
        self._loss_header_written = False
