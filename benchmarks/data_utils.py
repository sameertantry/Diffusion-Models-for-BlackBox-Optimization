"""Utilities for loading external benchmark datasets."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


DEFAULT_DATA_ENV = "BBOPT_DATA_DIR"
DEFAULT_DATA_DIRNAME = "data"


def resolve_data_path(
    env_var: Optional[str],
    default_relpath: str,
) -> Path:
    """Resolve a dataset path from env vars or default data dir.

    Args:
        env_var: Optional env var pointing directly to a dataset file.
        default_relpath: Relative path under the data directory.

    Returns:
        Resolved Path to the dataset.

    Raises:
        FileNotFoundError: If neither env var nor data dir is available.
    """
    if env_var and env_var in os.environ:
        return Path(os.environ[env_var]).expanduser()
    data_dir = os.environ.get(DEFAULT_DATA_ENV)
    if not data_dir:
        raise FileNotFoundError(
            f"Dataset path not found. Set {env_var} or {DEFAULT_DATA_ENV}."
        )
    return Path(data_dir).expanduser() / default_relpath


def get_repo_root() -> Path:
    """Resolve repository root from this file location."""
    return Path(__file__).resolve().parents[1]


def get_benchmark_data_dir(benchmark_name: str) -> Path:
    """Get the data directory for a benchmark."""
    return get_repo_root() / DEFAULT_DATA_DIRNAME / benchmark_name


def load_array_dataset(
    path: Path,
    x_key: str = "x",
    y_key: str = "y",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load datasets from npz or h5 files.

    Expected formats:
    - .npz: arrays stored under keys x/y
    - .h5/.hdf5: datasets stored under keys x/y
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        if x_key not in data or y_key not in data:
            raise KeyError(f"Expected keys '{x_key}' and '{y_key}' in {path}")
        return np.asarray(data[x_key]), np.asarray(data[y_key])
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'h5py' required to read .h5 datasets."
            ) from exc
        with h5py.File(path, "r") as handle:
            if x_key not in handle or y_key not in handle:
                raise KeyError(f"Expected keys '{x_key}' and '{y_key}' in {path}")
            return np.asarray(handle[x_key]), np.asarray(handle[y_key])
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_metadata(path: Path) -> Dict[str, object]:
    """Load metadata JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found at: {path}")
    with path.open("r") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    """Load a CSV file as dict rows."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure inputs are 2D arrays."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    return x


def flatten_one_hot(x: np.ndarray) -> np.ndarray:
    """Flatten one-hot arrays to (n_samples, dim)."""
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    return x


def to_2d_column(y: np.ndarray) -> np.ndarray:
    """Ensure outputs are 2D (n_samples, 1)."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2 and y.shape[1] == 1:
        return y
    if y.ndim == 2 and y.shape[0] == 1:
        return y.T
    if y.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array for y, got {y.shape}")
    return y

