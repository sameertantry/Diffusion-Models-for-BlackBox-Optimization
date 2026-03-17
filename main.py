"""Main entry point for black-box optimization experiments.

This module launches multiple experiments defined by a JSON config list.
Each experiment entry specifies benchmark, method, dimensions, and a seed count.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from benchmarks import BaseBenchmark
from benchmarks.analytical import Ackley, Rastrigin, Rosenbrock, Sphere
from benchmarks.design_benchmarks import (
    ANT,
    CHEMBL,
    DKITTY,
    SUPERCON,
    TFBIND,
    TFBIND8,
    TFBIND10,
)
from benchmarks.embedded_subspace import EmbeddedSubspaceBenchmark
from benchmarks.rotated_subspace import RotatedSubspaceBenchmark
from methods import BaseOptimizer, CMAES, DiffusionBBO, DiffusionOptimizer, REINFORCE, TPE


@dataclass
class BenchmarkConfig:
    """Configuration for the optimization benchmark."""
    name: str = "sphere"
    input_dim: Optional[int] = None
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    base_function: str = "sphere"
    ambient_dim: Optional[int] = None
    intrinsic_dim: Optional[int] = None
    seed: int = 42
    invalid_penalty: float = 1e6


@dataclass
class MethodConfig:
    """Configuration for the optimization method."""
    name: str = "random"
    num_iterations: int = 200
    eval_batch_size: int = 1
    seed: int = 42
    verbose: bool = False
    # --- Diffusion-specific (DDPMScheduler from diffusers) ---
    num_timesteps: int = 100
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_type: str = "epsilon"
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    # --- Diffusion network / training ---
    hidden_dim: int = 256
    time_embed_dim: int = 64
    depth: int = 3
    batch_size: int = 128
    lr_diffusion: float = 1e-3
    lr_regressor: float = 1e-3
    train_steps: int = 500
    quantile: Union[float, Dict[str, float]] = 0.75
    guidance_strength: float = 5.0
    use_regressor: bool = True
    min_data: float = 0.1
    device: Optional[str] = None
    # --- CMA-ES-specific ---
    sigma0: float = 0.5
    # --- DiffusionBBO-specific ---
    lr: float = 1e-3
    n_ensemble: int = 5
    p_uncond: float = 0.15
    guidance_scale: float = 2.0
    n_uae_samples: int = 20
    # --- Warm-start ---
    warm_start_n: int = 128
    # --- REINFORCE-specific ---
    reinforce_lr: float = 0.01
    reinforce_std_lr: Optional[float] = None
    reinforce_pop_size: int = 32
    reinforce_init_std: float = 0.3
    reinforce_entropy_coeff: float = 0.0
    reinforce_use_rank_transform: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for the experiment settings."""
    output_dir: str = "outputs"
    log_every: int = 10


EXAMPLE_CONFIG: List[Dict[str, Any]] = [
    {
        "benchmark": "sphere",
        "input_dim": 10,
        "method": "tpe",
        "num_iterations": 200,
        "eval_batch_size": 1,
        "seeds": 3,
        "log_every": 20,
        "output_dir": "outputs",
    },
    {
        "benchmark": "sphere",
        "input_dim": 10,
        "method": "cma_es",
        "num_iterations": 200,
        "eval_batch_size": 1,
        "sigma0": 0.5,
        "seeds": 3,
        "log_every": 20,
        "output_dir": "outputs",
    },
    {
        "benchmark": "sphere",
        "input_dim": 10,
        "method": "diffusion",
        "num_iterations": 200,
        "eval_batch_size": 1,
        "num_timesteps": 100,
        "beta_schedule": "linear",
        "hidden_dim": 256,
        "time_embed_dim": 64,
        "depth": 3,
        "batch_size": 128,
        "lr_diffusion": 1e-3,
        "lr_regressor": 1e-3,
        "train_steps": 200,
        "quantile": 0.9,
        "guidance_strength": 1.0,
        "use_regressor": True,
        "min_data": 0.1,
        "seeds": 3,
        "log_every": 20,
        "output_dir": "outputs",
    },
]

SUPPORTED_KEYS = {
    "benchmark",
    "method",
    "seeds",
    "input_dim",
    "ambient_dim",
    "intrinsic_dim",
    "base_function",
    "bounds",
    "bounds_low",
    "bounds_high",
    "invalid_penalty",
    "num_iterations",
    "eval_batch_size",
    "verbose",
    "sigma0",
    # diffusion / DDPMScheduler
    "num_timesteps",
    "beta_schedule",
    "beta_start",
    "beta_end",
    "prediction_type",
    "clip_sample",
    "clip_sample_range",
    # diffusion network / training
    "hidden_dim",
    "time_embed_dim",
    "depth",
    "batch_size",
    "lr_diffusion",
    "lr_regressor",
    "train_steps",
    "quantile",
    "guidance_strength",
    "use_regressor",
    "min_data",
    "device",
    "log_every",
    "output_dir",
    # diffusion_bbo specific
    "lr",
    "n_ensemble",
    "p_uncond",
    "guidance_scale",
    "n_uae_samples",
    # warm-start
    "warm_start_n",
    # REINFORCE-specific
    "reinforce_lr",
    "reinforce_std_lr",
    "reinforce_pop_size",
    "reinforce_init_std",
    "reinforce_entropy_coeff",
    "reinforce_use_rank_transform",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Black-box optimization experiments (JSON config list)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON experiment config list",
    )
    return parser.parse_args()


def _normalize_bound(value: Any, input_dim: int, label: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(input_dim, float(value))
    if isinstance(value, list):
        vals = [float(v) for v in value]
        if len(vals) == 1:
            return np.full(input_dim, vals[0])
        if len(vals) != input_dim:
            raise ValueError(
                f"{label} has {len(vals)} values but input_dim is {input_dim}"
            )
        return np.array(vals, dtype=np.float64)
    raise ValueError(f"Invalid {label} format: {value}")


def parse_bounds_from_config(
    config: Dict[str, Any],
    input_dim: Optional[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse bounds from config dict."""
    if "bounds" in config:
        bounds = config["bounds"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
        ):
            raise ValueError("bounds must be a list [lower, upper]")
        if input_dim is None:
            raise ValueError("input_dim must be set when providing bounds.")
        lower = _normalize_bound(bounds[0], input_dim, "bounds[0]")
        upper = _normalize_bound(bounds[1], input_dim, "bounds[1]")
        return (lower, upper)
    if "bounds_low" in config or "bounds_high" in config:
        if "bounds_low" not in config or "bounds_high" not in config:
            raise ValueError("Both bounds_low and bounds_high must be provided.")
        if input_dim is None:
            raise ValueError("input_dim must be set when providing bounds.")
        lower = _normalize_bound(config["bounds_low"], input_dim, "bounds_low")
        upper = _normalize_bound(config["bounds_high"], input_dim, "bounds_high")
        return (lower, upper)
    return None


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def instantiate_benchmark(cfg: BenchmarkConfig) -> BaseBenchmark:
    """Dynamically instantiate a benchmark from configuration.
    
    Args:
        cfg: Benchmark configuration.
        
    Returns:
        Instantiated benchmark object.
        
    Raises:
        ValueError: If benchmark name is not recognized.
    """
    # Map benchmark names to classes
    benchmark_classes = {
        "sphere": Sphere,
        "rosenbrock": Rosenbrock,
        "rastrigin": Rastrigin,
        "ackley": Ackley,
        "embedded": EmbeddedSubspaceBenchmark,
        "embedded_subspace": EmbeddedSubspaceBenchmark,
        "rotated": RotatedSubspaceBenchmark,
        "rotated_subspace": RotatedSubspaceBenchmark,
        # Design-Bench benchmarks (Diffusion-BBO, Wu et al. 2024)
        "tfbind": TFBIND,
        "tfbind8": TFBIND8,
        "tfbind10": TFBIND10,
        "supercon": SUPERCON,
        "superconductor": SUPERCON,
        "ant": ANT,
        "dkitty": DKITTY,
        "chembl": CHEMBL,
    }
    
    name_lower = cfg.name.lower()
    if name_lower not in benchmark_classes:
        raise ValueError(
            f"Unknown benchmark: {cfg.name}. "
            f"Available: {list(benchmark_classes.keys())}"
        )
    
    cls = benchmark_classes[name_lower]
    if name_lower in {"embedded", "embedded_subspace", "rotated", "rotated_subspace"}:
        if cfg.ambient_dim is None or cfg.intrinsic_dim is None:
            raise ValueError(
                "embedded/rotated subspace requires ambient_dim and intrinsic_dim"
            )
        return cls(
            base_function_name=cfg.base_function,
            ambient_dim=cfg.ambient_dim,
            intrinsic_dim=cfg.intrinsic_dim,
            seed=cfg.seed,
            bounds=cfg.bounds,
        )
    if name_lower in {"sphere", "rosenbrock", "rastrigin", "ackley"}:
        input_dim = cfg.input_dim or 2
        return cls(input_dim=input_dim, bounds=cfg.bounds)
    return cls(
        input_dim=cfg.input_dim,
        bounds=cfg.bounds,
        invalid_penalty=cfg.invalid_penalty,
    )


def instantiate_optimizer(
    cfg: MethodConfig,
    benchmark: BaseBenchmark,
) -> BaseOptimizer:
    """Dynamically instantiate an optimizer from configuration.
    
    Args:
        cfg: Method configuration.
        benchmark: The benchmark to optimize.
        
    Returns:
        Instantiated optimizer object.
        
    Raises:
        ValueError: If method name is not recognized.
    """
    # Get bounds from benchmark
    if benchmark.bounds is None:
        raise ValueError("Benchmark must have bounds defined for optimization")
    
    bounds = benchmark.bounds
    
    # Map method names to classes
    method_classes = {
        "tpe": TPE,
        "cma_es": CMAES,
        "diffusion": DiffusionOptimizer,
        "diffusion_bbo": DiffusionBBO,
        "reinforce": REINFORCE,
    }
    
    name_lower = cfg.name.lower()
    
    if name_lower == "random":
        raise NotImplementedError(
            "Random method not yet implemented. Use 'tpe', 'cma_es', 'diffusion', 'diffusion_bbo', or 'reinforce'."
        )
    elif name_lower not in method_classes:
        raise ValueError(
            f"Unknown method: {cfg.name}. "
            f"Available: {list(method_classes.keys())}"
        )
    
    cls = method_classes[name_lower]
    if name_lower == "diffusion":
        return cls(
            input_dim=benchmark.input_dim,
            bounds=bounds,
            num_timesteps=cfg.num_timesteps,
            beta_schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            prediction_type=cfg.prediction_type,
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            hidden_dim=cfg.hidden_dim,
            time_embed_dim=cfg.time_embed_dim,
            depth=cfg.depth,
            batch_size=cfg.batch_size,
            lr_diffusion=cfg.lr_diffusion,
            lr_regressor=cfg.lr_regressor,
            train_steps=cfg.train_steps,
            quantile=cfg.quantile,
            guidance_strength=cfg.guidance_strength,
            use_regressor=cfg.use_regressor,
            seed=cfg.seed,
            verbose=cfg.verbose,
            min_data=cfg.min_data,
            device=cfg.device,
        )
    if name_lower == "diffusion_bbo":
        return cls(
            input_dim=benchmark.input_dim,
            bounds=bounds,
            num_timesteps=cfg.num_timesteps,
            beta_schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            prediction_type=cfg.prediction_type,
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            hidden_dim=cfg.hidden_dim,
            time_embed_dim=cfg.time_embed_dim,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            train_steps=cfg.train_steps,
            n_ensemble=cfg.n_ensemble,
            p_uncond=cfg.p_uncond,
            guidance_scale=cfg.guidance_scale,
            n_uae_samples=cfg.n_uae_samples,
            seed=cfg.seed,
            min_data=cfg.min_data,
            device=cfg.device,
        )
    if name_lower == "cma_es":
        return cls(
            input_dim=benchmark.input_dim,
            bounds=bounds,
            seed=cfg.seed,
            sigma0=cfg.sigma0,
            verbose=cfg.verbose,
        )
    if name_lower == "reinforce":
        return cls(
            input_dim=benchmark.input_dim,
            bounds=bounds,
            seed=cfg.seed,
            lr=cfg.reinforce_lr,
            std_lr=cfg.reinforce_std_lr,
            pop_size=cfg.reinforce_pop_size,
            init_std=cfg.reinforce_init_std,
            entropy_coeff=cfg.reinforce_entropy_coeff,
            use_rank_transform=cfg.reinforce_use_rank_transform,
        )
    return cls(
        input_dim=benchmark.input_dim,
        bounds=bounds,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )


def run_optimization(
    benchmark: BaseBenchmark,
    optimizer: BaseOptimizer,
    num_iterations: int,
    eval_batch_size: int,
    log_every: int,
    warm_start_n: int = 128,
) -> Dict[str, Any]:
    """Run a single optimization run.
    
    Args:
        benchmark: The benchmark to optimize.
        optimizer: The optimizer to use.
        num_iterations: Number of optimization iterations.
        eval_batch_size: Number of candidates per iteration.
        log_every: Log progress every N iterations.
        warm_start_n: Number of initial data points to sample from the
            benchmark for warm-starting the optimizer.  Set to 0 to disable.
        
    Returns:
        Dictionary with optimization results.
    """
    loop_evals = num_iterations * eval_batch_size
    warm_start_evals = 0

    print(f"\n{'='*60}")
    print(
        "Starting optimization: "
        f"{num_iterations} iterations, "
        f"{eval_batch_size} points/iter "
        f"({loop_evals} loop evaluations)"
    )
    print(f"{'='*60}")
    
    best_value = float('inf')
    best_candidate = None
    evals_done = 0

    # ------------------------------------------------------------------
    # Warm-start: evaluate initial data from the benchmark and feed to
    # the optimizer *before* the ask/tell loop begins.
    # ------------------------------------------------------------------
    if warm_start_n > 0:
        try:
            x_init = benchmark.sample_initial_data(warm_start_n)
            if x_init is not None and len(x_init) > 0:
                y_init = benchmark.evaluate(x_init)
                y_init = np.atleast_1d(y_init).flatten()
                warm_start_evals = len(y_init)
                evals_done += warm_start_evals

                # Track best from initial data
                min_idx = int(np.argmin(y_init))
                if y_init[min_idx] < best_value:
                    best_value = float(y_init[min_idx])
                    best_candidate = x_init[min_idx]

                # Let the optimizer incorporate initial observations
                optimizer.warm_start(x_init, y_init)
                print(
                    f"  Warm-start: {warm_start_evals} initial points, "
                    f"best initial value = {best_value:.6f}"
                )
        except Exception as exc:
            print(f"  Warm-start skipped: {exc}")

    total_evals = warm_start_evals + loop_evals

    for iter_idx in range(num_iterations):
        # Sample candidate(s)
        candidates = optimizer.ask(n=eval_batch_size)
        if candidates.size == 0:
            break
        
        # Evaluate benchmark
        values = benchmark.evaluate(candidates)
        values = np.atleast_1d(values).flatten()
        evals_done += len(values)
        
        # Provide feedback to optimizer
        optimizer.tell(candidates, values)
        
        # Track best
        min_idx = int(np.argmin(values))
        if values[min_idx] < best_value:
            best_value = float(values[min_idx])
            best_candidate = candidates[min_idx]
        
        # Log progress
        if (iter_idx + 1) % log_every == 0 or iter_idx == 0:
            print(
                f"Iter {iter_idx + 1:4d}/{num_iterations}: "
                f"best_value = {best_value:.6f} "
                f"(evals {evals_done}/{total_evals})"
            )
    
    # Final results
    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"Best value: {best_value:.6f}")
    print(f"Best candidate: {best_candidate}")
    print(f"{'='*60}\n")
    
    return {
        "best_candidate": best_candidate,
        "best_value": best_value,
        "num_evals": evals_done,
    }


def load_experiment_list(config_path: Path) -> List[Dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Config must be a JSON list of experiment dicts.")
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Config entry {idx} must be a dict.")
    return payload


def validate_experiment_item(item: Dict[str, Any], index: int) -> None:
    required = {"benchmark", "method", "seeds"}
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"Experiment {index} missing keys: {missing}")
    if not isinstance(item["seeds"], int) or item["seeds"] <= 0:
        raise ValueError(f"Experiment {index} seeds must be a positive integer.")
    if "num_iterations" not in item:
        raise ValueError(
            f"Experiment {index} missing 'num_iterations'. "
            "This is now a required field (max_evals has been removed)."
        )
    unknown_keys = set(item.keys()) - SUPPORTED_KEYS
    if unknown_keys:
        raise ValueError(
            f"Experiment {index} has unsupported keys: {sorted(unknown_keys)}"
        )


def build_benchmark_config(item: Dict[str, Any], seed: int) -> BenchmarkConfig:
    name = str(item["benchmark"])
    base_function = str(item.get("base_function", "sphere"))
    input_dim = item.get("input_dim")
    ambient_dim = item.get("ambient_dim")
    intrinsic_dim = item.get("intrinsic_dim")
    invalid_penalty = float(item.get("invalid_penalty", 1e6))

    name_lower = name.lower()
    if name_lower in {"sphere", "rosenbrock", "rastrigin", "ackley"}:
        if input_dim is None:
            raise ValueError("input_dim must be set for analytical benchmarks.")
    if name_lower in {"embedded", "embedded_subspace", "rotated", "rotated_subspace"}:
        if ambient_dim is None or intrinsic_dim is None:
            raise ValueError("ambient_dim and intrinsic_dim must be set.")

    bounds_dim = ambient_dim if name_lower in {"embedded", "embedded_subspace", "rotated", "rotated_subspace"} else input_dim
    bounds = parse_bounds_from_config(item, bounds_dim)

    return BenchmarkConfig(
        name=name,
        input_dim=input_dim,
        bounds=bounds,
        base_function=base_function,
        ambient_dim=ambient_dim,
        intrinsic_dim=intrinsic_dim,
        seed=seed,
        invalid_penalty=invalid_penalty,
    )


def build_method_config(item: Dict[str, Any], seed: int) -> MethodConfig:
    return MethodConfig(
        name=str(item["method"]),
        num_iterations=int(item["num_iterations"]),
        eval_batch_size=int(item.get("eval_batch_size", 1)),
        seed=seed,
        verbose=bool(item.get("verbose", False)),
        # DDPMScheduler params
        num_timesteps=int(item.get("num_timesteps", 100)),
        beta_schedule=str(item.get("beta_schedule", "linear")),
        beta_start=float(item.get("beta_start", 1e-4)),
        beta_end=float(item.get("beta_end", 2e-2)),
        prediction_type=str(item.get("prediction_type", "epsilon")),
        clip_sample=bool(item.get("clip_sample", False)),
        clip_sample_range=float(item.get("clip_sample_range", 1.0)),
        # Network / training params
        hidden_dim=int(item.get("hidden_dim", 256)),
        time_embed_dim=int(item.get("time_embed_dim", 64)),
        depth=int(item.get("depth", 3)),
        batch_size=int(item.get("batch_size", 128)),
        lr_diffusion=float(item.get("lr_diffusion", 1e-3)),
        lr_regressor=float(item.get("lr_regressor", 1e-3)),
        train_steps=int(item.get("train_steps", 200)),
        quantile=item.get("quantile", 0.9),
        guidance_strength=float(item.get("guidance_strength", 1.0)),
        use_regressor=bool(item.get("use_regressor", True)),
        min_data=float(item.get("min_data", 0.1)),
        device=item.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        sigma0=float(item.get("sigma0", 0.5)),
        # DiffusionBBO-specific
        lr=float(item.get("lr", 1e-3)),
        n_ensemble=int(item.get("n_ensemble", 5)),
        p_uncond=float(item.get("p_uncond", 0.15)),
        guidance_scale=float(item.get("guidance_scale", 2.0)),
        n_uae_samples=int(item.get("n_uae_samples", 20)),
        # Warm-start
        warm_start_n=int(item.get("warm_start_n", 128)),
        # REINFORCE-specific
        reinforce_lr=float(item.get("reinforce_lr", 0.01)),
        reinforce_std_lr=(
            float(item["reinforce_std_lr"])
            if "reinforce_std_lr" in item
            else None
        ),
        reinforce_pop_size=int(item.get("reinforce_pop_size", 32)),
        reinforce_init_std=float(item.get("reinforce_init_std", 0.3)),
        reinforce_entropy_coeff=float(item.get("reinforce_entropy_coeff", 0.0)),
        reinforce_use_rank_transform=bool(
            item.get("reinforce_use_rank_transform", True)
        ),
    )


def resolve_evaluation_schedule(cfg: MethodConfig) -> Tuple[int, int]:
    """Resolve evaluation schedule from config.

    Returns:
        Tuple of ``(num_iterations, eval_batch_size)``.
    """
    eval_batch_size = max(1, int(cfg.eval_batch_size))
    num_iterations = int(cfg.num_iterations)
    if num_iterations <= 0:
        raise ValueError("num_iterations must be a positive integer.")
    return num_iterations, eval_batch_size


def stringify_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, (dict,)):
        return json.dumps(value)
    return str(value)


def collect_fieldnames(rows: Iterable[Dict[str, Any]]) -> List[str]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return [str(key) for key in sorted(keys, key=str)]


def resolve_log_directory(output_dir: str) -> Path:
    base_dir = Path(output_dir) / "logs"
    moscow = ZoneInfo("Europe/Moscow")
    day_str = datetime.now(moscow).strftime("%d.%m.%Y")
    log_dir = base_dir / day_str
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_csv(rows: List[Dict[str, Any]], output_dir: str) -> Path:
    csv_path = Path(output_dir) / "results.csv"
    fieldnames = collect_fieldnames(rows)
    normalized_rows: List[Dict[str, Any]] = []

    def normalize_cell(value: Any) -> Any:
        if value is None:
            return np.nan
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        return str(value)

    for row in rows:
        normalized: Dict[str, Any] = {}
        for key, value in row.items():
            normalized[str(key)] = normalize_cell(value)
        normalized_rows.append(normalized)

    df = pd.DataFrame(normalized_rows)
    df = df.reindex(columns=fieldnames)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="ignore")
    df.to_csv(csv_path, index=False)
    return csv_path


def build_experiment_dir(
    output_dir: str,
    experiments: List[Dict[str, Any]],
) -> Path:
    log_dir = resolve_log_directory(output_dir)
    benchmarks = sorted({str(item.get("benchmark", "unknown")) for item in experiments})
    methods = sorted({str(item.get("method", "unknown")) for item in experiments})
    time_str = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%H%M%S")

    def sanitize(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
        return cleaned.strip("_") or "unknown"

    bench_part = sanitize("_".join(benchmarks))
    method_part = sanitize("_".join(methods))
    dir_name = f"{bench_part}__{method_part}__{time_str}"
    exp_dir = log_dir / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def write_config_copy(config_path: Path, target_dir: Path) -> Path:
    config_copy_path = target_dir / "config.json"
    config_copy_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return config_copy_path

def main(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    experiments = load_experiment_list(config_path)

    all_rows: List[Dict[str, Any]] = []
    csv_output_dir: Optional[str] = None
    experiment_dir: Optional[Path] = None

    print("\n" + "=" * 60)
    print("Black-Box Optimization Experiments")
    print("=" * 60)
    print(f"Loaded {len(experiments)} experiments from {config_path}")

    for exp_idx, item in enumerate(experiments):
        validate_experiment_item(item, exp_idx)
        seed_count = int(item["seeds"])
        log_every = int(item.get("log_every", 10))
        output_dir = str(item.get("output_dir", "outputs"))
        if csv_output_dir is None:
            csv_output_dir = output_dir
            experiment_dir = build_experiment_dir(output_dir, experiments)
            write_config_copy(config_path, experiment_dir)
        elif csv_output_dir != output_dir:
            raise ValueError(
                "All experiments must share the same output_dir to write a single CSV."
            )

        for seed in range(seed_count):
            set_global_seed(seed)
            print(f"\n{'#' * 60}")
            print(f"Experiment {exp_idx + 1}/{len(experiments)} | Seed {seed}")
            print(f"{'#' * 60}")

            benchmark_cfg = build_benchmark_config(item, seed)
            method_cfg = build_method_config(item, seed)
            num_iterations, eval_batch_size = resolve_evaluation_schedule(method_cfg)
            method_cfg.num_iterations = num_iterations
            method_cfg.eval_batch_size = eval_batch_size
            experiment_cfg = ExperimentConfig(
                output_dir=output_dir,
                log_every=log_every,
            )

            benchmark = instantiate_benchmark(benchmark_cfg)
            optimizer = instantiate_optimizer(method_cfg, benchmark)
            if optimizer.verbose and experiment_dir is not None:
                optimizer.set_verbose_log_dir(experiment_dir / "verbose_logs")

            results = run_optimization(
                benchmark=benchmark,
                optimizer=optimizer,
                num_iterations=method_cfg.num_iterations,
                eval_batch_size=method_cfg.eval_batch_size,
                log_every=experiment_cfg.log_every,
                warm_start_n=method_cfg.warm_start_n,
            )

            if optimizer.verbose:
                optimizer.plot_convergence(
                    title=f"{method_cfg.name} on {benchmark_cfg.name} "
                          f"(dim={benchmark_cfg.input_dim}, seed={seed})",
                )

            row: Dict[str, Any] = {
                "experiment_idx": exp_idx,
                "seed": seed,
                "benchmark": benchmark_cfg.name,
                "method": method_cfg.name,
                "seeds": seed_count,
                "verbose": method_cfg.verbose,
                "input_dim": benchmark_cfg.input_dim,
                "ambient_dim": benchmark_cfg.ambient_dim,
                "intrinsic_dim": benchmark_cfg.intrinsic_dim,
                "base_function": benchmark_cfg.base_function,
                "invalid_penalty": benchmark_cfg.invalid_penalty,
                "num_iterations": method_cfg.num_iterations,
                "eval_batch_size": method_cfg.eval_batch_size,
                "log_every": experiment_cfg.log_every,
                "output_dir": experiment_cfg.output_dir,
                # diffusion / DDPMScheduler
                "num_timesteps": method_cfg.num_timesteps,
                "beta_schedule": method_cfg.beta_schedule,
                "beta_start": method_cfg.beta_start,
                "beta_end": method_cfg.beta_end,
                "prediction_type": method_cfg.prediction_type,
                "clip_sample": method_cfg.clip_sample,
                "clip_sample_range": method_cfg.clip_sample_range,
                # diffusion network / training
                "hidden_dim": method_cfg.hidden_dim,
                "time_embed_dim": method_cfg.time_embed_dim,
                "depth": method_cfg.depth,
                "batch_size": method_cfg.batch_size,
                "lr_diffusion": method_cfg.lr_diffusion,
                "lr_regressor": method_cfg.lr_regressor,
                "train_steps": method_cfg.train_steps,
                "quantile": method_cfg.quantile,
                "guidance_strength": method_cfg.guidance_strength,
                "use_regressor": method_cfg.use_regressor,
                "min_data": method_cfg.min_data,
                "device": method_cfg.device,
                "sigma0": method_cfg.sigma0,
                # DiffusionBBO-specific
                "lr": method_cfg.lr,
                "n_ensemble": method_cfg.n_ensemble,
                "p_uncond": method_cfg.p_uncond,
                "guidance_scale": method_cfg.guidance_scale,
                "n_uae_samples": method_cfg.n_uae_samples,
                # REINFORCE-specific
                "reinforce_lr": method_cfg.reinforce_lr,
                "reinforce_std_lr": method_cfg.reinforce_std_lr,
                "reinforce_pop_size": method_cfg.reinforce_pop_size,
                "reinforce_init_std": method_cfg.reinforce_init_std,
                "reinforce_entropy_coeff": method_cfg.reinforce_entropy_coeff,
                "reinforce_use_rank_transform": method_cfg.reinforce_use_rank_transform,
                "best_value": results["best_value"],
                "num_evals": results["num_evals"],
            }
            all_rows.append(row)

    if all_rows:
        if experiment_dir is None:
            experiment_dir = build_experiment_dir(csv_output_dir or "outputs", experiments)
            write_config_copy(config_path, experiment_dir)
        csv_path = write_csv(all_rows, str(experiment_dir))
        print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
