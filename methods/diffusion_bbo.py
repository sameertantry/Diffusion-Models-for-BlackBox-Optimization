"""Diffusion-BBO: Diffusion-Based Inverse Modeling for Online BBO.

Implements the algorithm from:

    Wu, D., Kuang, N. L., Niu, R., Ma, Y.-A., & Yu, R. (2024).
    *Diffusion-BBO: Diffusion-Based Inverse Modeling for Online
    Black-Box Optimization*. arXiv:2407.00610.

Key design differences from the standard :class:`DiffusionOptimizer`:

1. **Classifier-free guidance (CFG)** — a single network is jointly
   trained as both a conditional and unconditional denoiser by randomly
   dropping the conditioning variable *y* with probability ``p_uncond``.
   No separate regressor network is needed.

2. **Ensemble of M models** — ``n_ensemble`` noise-predictor networks
   are trained from different random initialisations.  Variance across
   their predictions quantifies *epistemic* uncertainty.

3. **UaE acquisition function** (Uncertainty-aware Exploration) —
   at each ``ask()`` call the optimizer selects the conditioning
   value *y** that maximises ``y - Δ_epistemic(y)`` over a discrete
   candidate set ``{w · φ_k : w ∈ W}``, where ``φ_k`` is the best
   observed objective value and ``W`` is a set of weights (default
   ``{0.6, 0.7, 0.8, 0.9, 1.0}``).

4. **Trains on ALL observed data** — unlike the quantile-filtered
   approach, the diffusion model sees the full dataset at every
   training round.  The conditioning score *y* carries the quality
   signal instead.

Interface
---------
Exposes the standard :class:`BaseOptimizer` ask/tell/reset contract so
it can be dropped into any experiment runner (COCO wrapper, ``main.py``,
etc.) without modification.

References
----------
- Wu et al. (2024) arXiv:2407.00610
- Ho & Salimans (2022) "Classifier-free diffusion guidance"
- Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from methods.base import BaseOptimizer
from methods.diffusion import _ConditionalDDPM

from diffusers import DDPMScheduler


# =========================================================================== #
#  Optimizer                                                                   #
# =========================================================================== #


class DiffusionBBO(BaseOptimizer):
    """Diffusion-BBO optimizer (Wu et al., 2024).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the search space.
    bounds : tuple of np.ndarray
        ``(lower_bounds, upper_bounds)`` for the search domain.
    num_timesteps : int
        Number of diffusion timesteps for both training (noise schedule
        granularity) and inference (reverse-process sampling).
    beta_schedule : str
        Noise schedule type (``"linear"``, ``"squaredcos_cap_v2"``, …).
    beta_start, beta_end : float
        Linear schedule endpoints.
    prediction_type : str
        ``"epsilon"`` / ``"sample"`` / ``"v_prediction"``.
    clip_sample : bool
        Whether the scheduler clips its internal x₀ prediction.
    clip_sample_range : float
        Clipping range when *clip_sample* is True.
    hidden_dim : int
        Width of the MLP hidden layers.
    time_embed_dim : int
        Dimension of the sinusoidal time embedding.
    batch_size : int
        Mini-batch size for training.
    lr : float
        Learning rate (shared by all ensemble members).
    train_steps : int
        Gradient steps per training round.
    n_ensemble : int
        Number of ensemble models *M* (paper default: 5).
    p_uncond : float
        Probability of dropping the conditioning variable during
        training (classifier-free guidance dropout).  Paper: 0.15.
    w_candidates : list of float
        Weight multipliers for the UaE candidate set.
        Paper: ``[0.6, 0.7, 0.8, 0.9, 1.0]``.
    guidance_scale : float
        Classifier-free guidance scale *γ* at inference.
        ``ε_guided = (1+γ)·ε_cond − γ·ε_uncond``.  Paper: 2.0.
    n_uae_samples : int
        Number of samples generated per model per candidate when
        computing the UaE acquisition function.
    seed : int
        Base random seed (each ensemble member uses ``seed + i``).
    min_data : int
        Minimum data points before training begins.
    device : str or None
        Torch device.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        *,
        # DDPMScheduler
        num_timesteps: int = 100,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        prediction_type: str = "epsilon",
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        # Network
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        # Training
        batch_size: int = 256,
        lr: float = 1e-3,
        train_steps: int = 500,
        # Diffusion-BBO specific
        n_ensemble: int = 5,
        p_uncond: float = 0.15,
        w_candidates: Optional[List[float]] = None,
        guidance_scale: float = 2.0,
        n_uae_samples: int = 20,
        # General
        seed: int = 42,
        min_data: int = 64,
        device: Optional[str] = None,
    ):
        super().__init__(
            name="DiffusionBBO",
            input_dim=input_dim,
            bounds=bounds,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        torch.manual_seed(seed)

        # Bounds
        self.lower = np.atleast_1d(bounds[0]).astype(np.float32)
        self.upper = np.atleast_1d(bounds[1]).astype(np.float32)
        if self.lower.size == 1:
            self.lower = np.full(input_dim, self.lower.item(), dtype=np.float32)
        if self.upper.size == 1:
            self.upper = np.full(input_dim, self.upper.item(), dtype=np.float32)

        # Store hyper-parameters
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.lr = lr
        self.train_steps = train_steps
        self.min_data = min_data
        self.p_uncond = p_uncond
        self.w_candidates: List[float] = (
            w_candidates if w_candidates is not None
            else [0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self.guidance_scale = guidance_scale
        self.n_ensemble = n_ensemble
        self.n_uae_samples = n_uae_samples

        # ---- Noise scheduler (shared) -----------------------------------
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

        # ---- Ensemble of CFG noise predictors ---------------------------
        self.ensemble: List[_ConditionalDDPM] = []
        self.ensemble_opts: List[Adam] = []
        for i in range(n_ensemble):
            torch.manual_seed(seed + i)
            model = _ConditionalDDPM(
                input_dim, hidden_dim, time_embed_dim,
            ).to(self.device)
            self.ensemble.append(model)
            self.ensemble_opts.append(Adam(model.parameters(), lr=lr))

        torch.manual_seed(seed)

        # ---- Data buffers -----------------------------------------------
        self.x_data: List[np.ndarray] = []
        self.y_data: List[float] = []  # maximisation scores (negated COCO)
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

        # ---- State flags ------------------------------------------------
        self._needs_retrain: bool = False

    # ------------------------------------------------------------------ #
    # Normalisation                                                      #
    # ------------------------------------------------------------------ #

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        """Normalise x to [-1, 1] using the domain bounds."""
        return 2.0 * (x - self.lower) / (self.upper - self.lower + 1e-8) - 1.0

    def _denormalize_x(self, x_norm: np.ndarray) -> np.ndarray:
        """Map from [-1, 1] back to the original domain."""
        return self.lower + (x_norm + 1.0) / 2.0 * (self.upper - self.lower + 1e-8)

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Z-score normalisation for objective values."""
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def _update_normalization(self) -> None:
        if len(self.y_data) < 2:
            self.y_mean = 0.0
            self.y_std = 1.0
            return
        y_arr = np.array(self.y_data, dtype=np.float32)
        self.y_mean = float(np.mean(y_arr))
        self.y_std = float(np.std(y_arr) + 1e-8)

    # ------------------------------------------------------------------ #
    # Sampling helpers                                                   #
    # ------------------------------------------------------------------ #

    def _sample_uniform(self, n: int) -> np.ndarray:
        return self.rng.uniform(
            self.lower, self.upper, size=(n, self.input_dim),
        ).astype(np.float32)

    @torch.no_grad()
    def _cfg_sample(
        self,
        model: _ConditionalDDPM,
        y_cond_norm: float,
        n: int,
    ) -> torch.Tensor:
        """Generate *n* samples via classifier-free guided reverse diffusion.

        Parameters
        ----------
        model : _ConditionalDDPM
            A single ensemble member.
        y_cond_norm : float
            Normalised conditioning value.
        n : int
            Number of samples to generate.

        Returns
        -------
        torch.Tensor
            Denoised samples in *normalised* x-space, shape ``(n, input_dim)``.
        """
        model.eval()

        # Use a fresh scheduler to avoid state leaks between calls.
        scheduler = DDPMScheduler(**self._scheduler_config)
        scheduler.set_timesteps(self.num_timesteps, device=self.device)

        x_t = torch.randn(n, self.input_dim, device=self.device)
        y_cond = torch.full((n,), y_cond_norm, device=self.device)
        y_uncond = torch.zeros(n, device=self.device)

        for t in scheduler.timesteps:
            t_batch = t.expand(n)

            # Conditional prediction  ε_θ(x_t, t, y)
            eps_cond = model(x_t, t_batch, y_cond)
            # Unconditional prediction  ε_θ(x_t, t, ∅)
            eps_uncond = model(x_t, t_batch, y_uncond)

            # CFG:  ε = (1+γ)·ε_cond − γ·ε_uncond
            eps_guided = (
                (1.0 + self.guidance_scale) * eps_cond
                - self.guidance_scale * eps_uncond
            )

            output = scheduler.step(eps_guided, int(t), x_t)
            x_t = output.prev_sample

        return x_t

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #

    def _train_single_model(
        self,
        model: _ConditionalDDPM,
        optimizer: Adam,
    ) -> None:
        """Train one ensemble member with the CFG loss (Eq. 4 of the paper).

        The conditioning variable *y* is dropped to 0 with probability
        ``p_uncond``, jointly training the conditional and unconditional
        denoiser in a single network.
        """
        x_arr = np.array(self.x_data, dtype=np.float32)
        y_arr = np.array(self.y_data, dtype=np.float32)

        x_norm = self._normalize_x(x_arr)
        y_norm = self._normalize_y(y_arr)

        x_tensor = torch.tensor(x_norm, device=self.device)
        y_tensor = torch.tensor(y_norm, device=self.device)

        n_data = x_tensor.shape[0]
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        model.train()
        for _ in range(self.train_steps):
            idx = torch.randint(
                0, n_data, (self.batch_size,), device=self.device,
            )
            x0 = x_tensor[idx]
            y_batch = y_tensor[idx]

            # Classifier-free dropout: set y → 0 with probability p_uncond.
            mask = torch.bernoulli(
                torch.full(
                    (self.batch_size,), self.p_uncond, device=self.device,
                ),
            )
            y_masked = y_batch * (1.0 - mask)

            # Forward diffusion
            noise = torch.randn_like(x0)
            timesteps = torch.randint(
                0, num_train_timesteps, (self.batch_size,),
                device=self.device,
            ).long()
            x_t = self.noise_scheduler.add_noise(x0, noise, timesteps)

            eps_pred = model(x_t, timesteps, y_masked)
            loss = torch.mean((eps_pred - noise) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _train_ensemble(self) -> None:
        """Train all M ensemble members from scratch on the current data."""
        for model, opt in zip(self.ensemble, self.ensemble_opts):
            self._train_single_model(model, opt)

    # ------------------------------------------------------------------ #
    # UaE acquisition function                                           #
    # ------------------------------------------------------------------ #

    def _compute_epistemic_uncertainty(
        self,
        y_cond_norm: float,
    ) -> float:
        r"""Compute epistemic uncertainty Δ_epistemic(y, D).

        From Proposition 1 of the paper:

        .. math::

            \Delta_{\text{epistemic}}(y, \mathcal{D}) =
                \mathrm{Var}_{\theta_i \sim p(\cdot|\mathcal{D})}
                \left(
                    \mathbb{E}_{\mathbf{x}_{i,j} \sim p_{\theta_i}(\cdot|y)}
                    \left[ \|\mathbf{x}_{i,j}\| \right]
                \right)

        i.e. variance *across models* of the mean sample norm.
        """
        mean_norms: List[float] = []
        for model in self.ensemble:
            samples = self._cfg_sample(model, y_cond_norm, self.n_uae_samples)
            mean_norm = float(torch.norm(samples, dim=-1).mean().item())
            mean_norms.append(mean_norm)

        return float(np.var(mean_norms))

    def _select_conditioning_value(self) -> float:
        """Select y* = argmax_y  α(y, D)  via UaE over the candidate set.

        Candidate set:  V_k = {w · φ_k_norm : w ∈ W}
        where φ_k_norm is the normalised best-observed value.

        Acquisition:  α(y, D) = y − Δ_epistemic(y, D)
        (paper Section 5.3, Eq. 8).
        """
        best_y_raw = max(self.y_data)
        phi_k_norm = float(self._normalize_y(np.array([best_y_raw]))[0])

        best_acq = -np.inf
        best_y_norm = phi_k_norm

        for w in self.w_candidates:
            y_cand = w * phi_k_norm
            epistemic = self._compute_epistemic_uncertainty(y_cand)
            acq = y_cand - epistemic
            if acq > best_acq:
                best_acq = acq
                best_y_norm = y_cand

        return best_y_norm

    # ------------------------------------------------------------------ #
    # Ask / Tell interface                                               #
    # ------------------------------------------------------------------ #

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose *n* candidate solutions.

        Before ``min_data`` points have been collected the optimizer
        returns uniform random samples.  After that it retrains the
        ensemble (if new data is available), selects the optimal
        conditioning value via UaE, and generates samples using
        classifier-free guided reverse diffusion.
        """
        if len(self.x_data) < self.min_data:
            return self._sample_uniform(n)

        # Retrain the ensemble when new data has arrived since last ask.
        if self._needs_retrain:
            self._update_normalization()
            self._train_ensemble()
            self._needs_retrain = False

        # Select y* via UaE and generate candidates.
        y_star_norm = self._select_conditioning_value()
        samples_norm = self._cfg_sample(self.ensemble[0], y_star_norm, n)

        # Denormalise and clip to bounds.
        # Use .tolist() to safely bypass numpy 2.x / torch interop issues.
        samples_np = np.array(
            samples_norm.detach().cpu().tolist(), dtype=np.float32,
        )
        samples_denorm = self._denormalize_x(samples_np)
        samples_denorm = np.clip(samples_denorm, self.lower, self.upper)
        return samples_denorm

    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        """Report evaluation results.

        Objective values are negated internally to convert COCO-style
        minimisation into the maximisation formulation used by the paper.
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        # Negate: COCO minimises, Diffusion-BBO maximises.
        y = -y

        for xi, yi in zip(x, y):
            self.x_data.append(xi.copy())
            self.y_data.append(float(yi))
            self.num_evals += 1

        if len(self.x_data) >= self.min_data:
            self._needs_retrain = True

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------

    def warm_start(self, x: np.ndarray, y: np.ndarray) -> None:
        """Seed the replay buffer with initial observations.

        Data is stored with the minimisation-to-maximisation negation
        (matching ``tell()``) but **without** incrementing ``num_evals``.
        If the buffer reaches ``min_data``, the retrain flag is set so
        that the next ``ask()`` triggers model training.

        Args:
            x: Array of shape ``(n, input_dim)`` – evaluated inputs.
            y: Array of shape ``(n,)`` – objective values (lower is better).
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.atleast_1d(np.asarray(y, dtype=np.float32)).flatten()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Negate: framework minimises, internal model maximises.
        y = -y
        for xi, yi in zip(x, y):
            self.x_data.append(xi.copy())
            self.y_data.append(float(yi))
        if len(self.x_data) >= self.min_data:
            self._needs_retrain = True

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state for a fresh run."""
        self.x_data = []
        self.y_data = []
        self.num_evals = 0
        self.y_mean = 0.0
        self.y_std = 1.0
        self._needs_retrain = False

        # Re-initialise all ensemble models with their original seeds.
        new_ensemble: List[_ConditionalDDPM] = []
        new_opts: List[Adam] = []
        for i, model in enumerate(self.ensemble):
            torch.manual_seed(self.seed + i)
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            new_opts.append(Adam(model.parameters(), lr=self.lr))
            new_ensemble.append(model)
        self.ensemble = new_ensemble
        self.ensemble_opts = new_opts
        torch.manual_seed(self.seed)
