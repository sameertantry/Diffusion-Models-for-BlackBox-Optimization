"""Diffusion-based black-box optimizer with conditional guidance.

Uses the Hugging Face ``diffusers`` library (``DDPMScheduler``) for noise
schedule management and forward/reverse diffusion process, while keeping
custom MLP networks for the noise predictor and guidance regressor (since
the data is low-dimensional vectors, not images).

Key design points:
- Learns a conditional distribution p(x | y >= τ) via a conditional DDPM.
- DDPMScheduler handles beta schedule, forward noising, and reverse steps.
- Uses a separate regressor for classifier-free guidance in the reverse process.
- Trains both models online as (x, y) data arrives.
- Exposes a numpy-based ask/tell interface while using torch internally.

References:
- "Diffusion Models for Black-Box Optimization"
- "Conditional Diffusion Models are Minimax-Optimal and Manifold-Adaptive ..."
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from methods.base import BaseOptimizer

# --------------------------------------------------------------------------- #
# numpy 2.x + PyTorch compatibility shim                                      #
#                                                                              #
# ``torch.from_numpy`` may raise                                              #
#   TypeError: expected np.ndarray (got numpy.ndarray)                         #
# when PyTorch was compiled against numpy 1.x but runtime uses numpy 2.x.     #
# DDPMScheduler (and its ``set_timesteps``) calls ``torch.from_numpy``         #
# internally, so we patch it once at import time.                              #
# --------------------------------------------------------------------------- #
_torch_from_numpy_original = torch.from_numpy


def _torch_from_numpy_compat(ndarray):  # type: ignore[override]
    try:
        return _torch_from_numpy_original(ndarray)
    except TypeError:
        try:
            return torch.tensor(ndarray)
        except TypeError:
            return torch.tensor(ndarray.tolist())


torch.from_numpy = _torch_from_numpy_compat  # type: ignore[assignment]

from diffusers import DDPMScheduler  # noqa: E402  (must come after the shim)


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


# ------------------------------------------------------------------ #
# Network components                                                  #
# ------------------------------------------------------------------ #


class _TimeEmbedding(nn.Module):
    """Sinusoidal time embedding used for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class _ConditionalDDPM(nn.Module):
    """Conditional noise predictor ε_θ(x_t, t, y_cond).

    Parameters
    ----------
    input_dim : int
        Dimensionality of x.
    hidden_dim : int
        Width of every hidden linear layer.
    time_dim : int
        Dimensionality of the sinusoidal time embedding.
    depth : int
        Total number of ``nn.Linear`` layers (including the final
        projection).  Must be >= 2.  With ``depth=5`` the network has
        4 hidden layers (each followed by SiLU) and 1 output projection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, time_dim: int, depth: int = 3):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        self.time_embed = _TimeEmbedding(time_dim)

        layers: List[nn.Module] = [
            nn.Linear(input_dim + time_dim + 1, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y_cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        y_cond = y_cond.view(-1, 1)
        inp = torch.cat([x_t, t_emb, y_cond], dim=-1)
        return self.net(inp)


class _Regressor(nn.Module):
    """Regressor f_φ(x) ≈ y for guidance."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------ #
# Optimizer                                                           #
# ------------------------------------------------------------------ #


class DiffusionOptimizer(BaseOptimizer):
    """Diffusion-based optimizer with conditional guidance (using diffusers).

    Implements distributional optimization by modeling p(x | y >= τ) and
    sampling from it using a conditional DDPM.  The noise schedule and
    forward/reverse process are managed by ``diffusers.DDPMScheduler``.
    Guidance is added using a learned regressor f_φ(x) to bias sampling
    toward higher objective values.

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
        Type of noise schedule.  Supported by ``DDPMScheduler``:
        ``"linear"``, ``"cosine"``, ``"squaredcos_cap_v2"``,
        ``"scaled_linear"``, ``"sigmoid"``.
    beta_start, beta_end : float
        Start / end values for the ``"linear"`` and ``"scaled_linear"``
        schedules (ignored by ``"squaredcos_cap_v2"``).
    prediction_type : str
        What the noise predictor network outputs:
        ``"epsilon"`` (predict noise), ``"sample"`` (predict x₀),
        or ``"v_prediction"``.
    clip_sample : bool
        Whether the scheduler clips its internal x₀ prediction.
        ``False`` by default because optimisation data has arbitrary bounds
        and final clipping is done explicitly.
    clip_sample_range : float
        Clipping range when *clip_sample* is ``True``.
    hidden_dim, time_embed_dim : int
        Widths of the MLP noise predictor and its time embedding.
    depth : int
        Total number of ``nn.Linear`` layers in the noise predictor
        (including the output projection).  Must be >= 2.  For example,
        ``depth=5`` gives 4 hidden layers + 1 output layer.
    batch_size : int
        Mini-batch size for training both networks.
    lr_diffusion, lr_regressor : float
        Learning rates for the noise predictor and regressor respectively.
    train_steps : int
        Number of gradient steps per ``tell()`` call.
    quantile : float or dict
        Quantile threshold τ for filtering top-performing data.  Can be a
        single float (constant) or a dict mapping percentage-of-budget
        thresholds (as strings) to quantile values, e.g.
        ``{"0": 0.25, "15": 0.5, "50": 0.75}``.  Requires ``budget`` to
        be set (see below) for percentage-based scheduling.
    guidance_strength : float
        Scale of the regressor gradient added to the noise prediction
        during the reverse process.
    use_regressor : bool
        Whether to train/use the auxiliary regressor for guidance.
        If ``False``, sampling runs without guidance.
    seed : int
        Random seed.
    min_data : float
        Fraction of the total evaluation budget that must be reached
        before training begins (uniform random sampling is used until
        then).  For example, ``0.1`` means 10 % of the budget.
        Requires ``budget`` to be set before the first ``ask()`` call.
    budget : int or None
        Total evaluation budget.  Used to compute progress for the
        quantile schedule and the ``min_data`` threshold.  Can also be
        set externally after construction (e.g. by the experiment runner).
    device : str or None
        Torch device (``"cuda"``, ``"cpu"``, or ``None`` for auto-detect).
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
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
        depth: int = 3,
        batch_size: int = 64,
        lr_diffusion: float = 3e-4,
        lr_regressor: float = 1e-3,
        train_steps: int = 50,
        quantile: Union[float, Dict[str, float]] = 0.5,
        guidance_strength: float = 1.0,
        use_regressor: bool = False,
        seed: int = 42,
        min_data: float = 0.1,
        verbose: bool = False,
        budget: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super().__init__(
            name="Diffusion",
            input_dim=input_dim,
            bounds=bounds,
            verbose=verbose,
        )
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
        self.train_steps = train_steps
        self.quantile = quantile
        self.guidance_strength = guidance_strength
        self.use_regressor = use_regressor
        self.min_data_frac = float(min_data)
        self.budget: Optional[int] = budget

        # ---- diffusers noise scheduler ----------------------------------------
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )

        # ---- networks ----------------------------------------------------------
        self.noise_pred_net = _ConditionalDDPM(input_dim, hidden_dim, time_embed_dim, depth=depth).to(self.device)
        self.noise_pred_opt = Adam(self.noise_pred_net.parameters(), lr=lr_diffusion)
        self.regressor: Optional[_Regressor] = None
        self.regressor_opt: Optional[Adam] = None
        if self.use_regressor:
            self.regressor = _Regressor(input_dim, hidden_dim).to(self.device)
            self.regressor_opt = Adam(self.regressor.parameters(), lr=lr_regressor)

        # ---- data buffers -------------------------------------------------------
        self.x_data: List[np.ndarray] = []
        self.y_data: List[float] = []
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.tau: float = -np.inf

    # ------------------------------------------------------------------ #
    # Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @property
    def _min_data_count(self) -> int:
        """Absolute number of points required before training starts."""
        if self.budget is not None and self.budget > 0:
            return max(1, int(self.min_data_frac * self.budget))
        return max(1, int(self.min_data_frac * 1000))

    def _get_current_quantile(self) -> float:
        """Return the active quantile value based on the schedule."""
        if isinstance(self.quantile, (int, float)):
            return float(self.quantile)

        if self.budget is not None and self.budget > 0:
            progress_pct = (self.num_evals / self.budget) * 100.0
        else:
            progress_pct = 0.0

        thresholds = sorted(self.quantile.keys(), key=lambda k: float(k))
        current_q = float(self.quantile[thresholds[0]])
        for key in thresholds:
            if progress_pct >= float(key):
                current_q = float(self.quantile[key])
            else:
                break
        return current_q

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def _update_normalization(self) -> None:
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

    def _filter_top(self) -> Tuple[np.ndarray, np.ndarray]:
        y_arr = np.array(self.y_data, dtype=np.float32)
        x_arr = np.array(self.x_data, dtype=np.float32)
        if y_arr.size == 0:
            return x_arr, y_arr

        finite_mask = np.isfinite(y_arr) & np.all(np.isfinite(x_arr), axis=1)
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        if y_arr.size == 0:
            return x_arr, y_arr

        q = self._get_current_quantile()
        self.tau = float(np.quantile(y_arr, q))
        mask = y_arr >= self.tau
        if not np.any(mask):
            best_idx = np.argmax(y_arr)
            mask[best_idx] = True
        return x_arr[mask], y_arr[mask]

    def _sample_uniform(self, n: int) -> np.ndarray:
        return self.rng.uniform(self.lower, self.upper, size=(n, self.input_dim)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Training (extracted)                                               #
    # ------------------------------------------------------------------ #

    def _train_networks(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> None:
        """Run a fixed number of gradient steps for both networks.

        Parameters
        ----------
        x_train : torch.Tensor
            Training inputs of shape ``(N, input_dim)``.
        y_train : torch.Tensor
            Normalised training targets of shape ``(N,)``.
        """
        if x_train.shape[0] == 0:
            return

        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        effective_batch = min(self.batch_size, x_train.shape[0])

        self.noise_pred_net.train()
        if self.use_regressor and self.regressor is not None:
            self.regressor.train()

        for _ in range(self.train_steps):
            # -- noise predictor (DDPM) ----------------------------------
            idx = torch.randint(0, x_train.shape[0], (effective_batch,), device=self.device)
            x0 = x_train[idx]
            y_cond = y_train[idx]

            noise = torch.randn_like(x0)
            timesteps = torch.randint(
                0, num_train_timesteps, (effective_batch,),
                device=self.device,
            ).long()

            # Forward diffusion via diffusers:
            #   x_t = sqrt(ᾱ_t) · x₀ + sqrt(1 - ᾱ_t) · ε
            x_t = self.noise_scheduler.add_noise(x0, noise, timesteps)

            eps_pred = self.noise_pred_net(x_t, timesteps, y_cond)
            loss = torch.mean((eps_pred - noise) ** 2)

            self.noise_pred_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.noise_pred_net.parameters(), max_norm=1.0)
            self.noise_pred_opt.step()

            # -- regressor -----------------------------------------------
            if self.use_regressor and self.regressor is not None and self.regressor_opt is not None:
                self.regressor_opt.zero_grad()

                reg_idx = torch.randint(0, x_train.shape[0], (effective_batch,), device=self.device)
                x_reg = x_train[reg_idx]
                y_reg = y_train[reg_idx]
                y_pred = self.regressor(x_reg)
                reg_loss = torch.mean((y_pred - y_reg) ** 2)

                reg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=1.0)
                self.regressor_opt.step()

    # ------------------------------------------------------------------ #
    # Ask / Tell interface                                               #
    # ------------------------------------------------------------------ #

    def ask(self, n: int = 1) -> np.ndarray:
        if len(self.x_data) < self._min_data_count:
            x_uniform = self._sample_uniform(n)
            self._register_asked(x_uniform)
            return x_uniform

        self.noise_pred_net.eval()
        if self.use_regressor and self.regressor is not None:
            self.regressor.eval()

        self.noise_scheduler.set_timesteps(self.num_timesteps, device=self.device)

        x_t = torch.randn(n, self.input_dim, device=self.device)
        y_cond = torch.full(
            (n,), self._normalize_y(np.array([self.tau]))[0], device=self.device,
        )

        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(n)
            eps_theta = self.noise_pred_net(x_t, t_batch, y_cond)

            eps_guided = eps_theta
            if self.use_regressor and self.regressor is not None:
                x_t.requires_grad_(True)
                y_pred = self.regressor(x_t)
                grad = torch.autograd.grad(y_pred.sum(), x_t, create_graph=False)[0]
                x_t = x_t.detach()
                eps_guided = eps_theta - self.guidance_strength * grad

            scheduler_output = self.noise_scheduler.step(eps_guided, int(t), x_t)
            x_t = scheduler_output.prev_sample

        lower_t = torch.as_tensor(self.lower, dtype=torch.float32, device=self.device)
        upper_t = torch.as_tensor(self.upper, dtype=torch.float32, device=self.device)
        x_t = torch.max(torch.min(x_t, upper_t), lower_t)
        x_np = x_t.detach().cpu().numpy().astype(np.float32)

        bad_mask = ~np.all(np.isfinite(x_np), axis=1)
        if np.any(bad_mask):
            x_np[bad_mask] = self._sample_uniform(int(bad_mask.sum()))

        self._register_asked(x_np)
        return x_np

    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y_obj = np.asarray(y, dtype=np.float32).reshape(-1)
        # Convert minimization objective into a maximization score.
        y = -y_obj

        for xi, yi in zip(x, y):
            self.num_evals += 1
            if not (np.all(np.isfinite(xi)) and np.isfinite(yi)):
                continue
            self.x_data.append(xi.copy())
            self.y_data.append(float(yi))

        if len(self.x_data) < self._min_data_count:
            self._register_told(x, y_obj, tau=np.nan)
            return

        self._update_normalization()
        x_top, y_top = self._filter_top()

        x_train = _to_tensor(x_top, self.device)
        y_train = _to_tensor(self._normalize_y(y_top), self.device)

        self._train_networks(x_train, y_train)
        tau_val = float(self.tau) if np.isfinite(self.tau) else np.nan
        self._register_told(x, y_obj, tau=tau_val)

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------

    def warm_start(self, x: np.ndarray, y: np.ndarray) -> None:
        """Seed the replay buffer with initial observations.

        Data is stored exactly as ``tell()`` would store it (with the
        minimisation-to-maximisation negation) but **without** incrementing
        ``num_evals`` and without triggering an immediate training pass.
        Training will start once additional ``tell()`` calls push the
        buffer past ``min_data``.

        Args:
            x: Array of shape ``(n, input_dim)`` – evaluated inputs.
            y: Array of shape ``(n,)`` – objective values (lower is better).
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.atleast_1d(np.asarray(y, dtype=np.float32)).flatten()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Negate to match internal maximisation convention
        y = -y
        for xi, yi in zip(x, y):
            if not (np.all(np.isfinite(xi)) and np.isfinite(yi)):
                continue
            self.x_data.append(xi.copy())
            self.y_data.append(float(yi))

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.x_data = []
        self.y_data = []
        self.num_evals = 0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.tau = -np.inf

        modules: List[nn.Module] = [self.noise_pred_net]
        if self.regressor is not None:
            modules.append(self.regressor)
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        self.noise_pred_opt = Adam(
            self.noise_pred_net.parameters(),
            lr=self.noise_pred_opt.param_groups[0]["lr"],
        )
        if self.regressor is not None and self.regressor_opt is not None:
            self.regressor_opt = Adam(
                self.regressor.parameters(),
                lr=self.regressor_opt.param_groups[0]["lr"],
            )
        self._reset_verbose_trace()
