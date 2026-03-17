"""Design-Bench scientific discovery benchmarks for online black-box optimization.

Implements the 6 benchmarks from the Diffusion-BBO paper (Wu et al., 2024,
arXiv:2407.00610):

    1. TFBind8        – Transcription factor binding (8-mer DNA sequences)
    2. TFBind10       – Transcription factor binding (10-mer DNA sequences)
    3. Superconductor – Superconducting material critical temperature prediction
    4. Ant            – Ant morphology design (MuJoCo locomotion)
    5. D'Kitty        – D'Kitty morphology design (MuJoCo locomotion)
    6. ChEMBL         – Molecular activity prediction (drug discovery)

These benchmarks wrap the ``design_bench`` package (Trabucco et al., ICML 2022)
and adapt it for *online* BBO with the ask/tell interface used in this codebase.

Two evaluation modes (selected automatically):

* **Oracle mode** (default when ``design_bench`` is installed):
  Uses the actual oracle from ``design_bench`` for evaluation.
* **Lookup mode** (fallback when oracle unavailable):
  Uses nearest-neighbour lookup on the downloaded offline dataset.

Scores are **negated** so that the standard minimisation loop in ``main.py``
effectively *maximises* the Design-Bench objective.

Setup
-----
1. ``pip install design-bench``
2. ``python scripts/download_data.py --benchmarks all``

References
----------
- Wu et al., "Diffusion-BBO: Diffusion-Based Inverse Modeling for
  Online Black-Box Optimization", arXiv:2407.00610, 2024.
- Trabucco et al., "Design-Bench: Benchmarks for Data-Driven Offline
  Model-Based Optimization", ICML 2022.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# NumPy compatibility shim — design_bench uses deprecated aliases
# (np.bool, np.int, np.float, np.object, np.str, np.complex) which were
# removed in NumPy 1.24+.  Restore them before design_bench is imported.
_NP_COMPAT_MAP = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "long": int,
}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _alias, _builtin in _NP_COMPAT_MAP.items():
        if not hasattr(np, _alias):
            setattr(np, _alias, _builtin)  # type: ignore[attr-defined]

# np.loads (pickle-based array deserialiser) was removed in NumPy 1.20.
# design_bench still uses it internally for some oracles.
if not hasattr(np, "loads"):
    import pickle as _pickle
    np.loads = _pickle.loads  # type: ignore[attr-defined]

from benchmarks.base import BaseBenchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Resolve the repository root (parent of benchmarks/)."""
    return Path(__file__).resolve().parents[1]


def _data_dir_for(benchmark_name: str) -> Path:
    """Return ``<repo>/data/<benchmark_name>``."""
    return _repo_root() / "data" / benchmark_name


def _load_local_data(
    benchmark_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Load ``X.npy``, ``y.npy``, and ``metadata.json`` from the data dir.

    Returns ``(x, y, metadata)`` or ``(None, None, None)`` when files are
    missing.
    """
    data_dir = _data_dir_for(benchmark_name)
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    meta_path = data_dir / "metadata.json"

    if not (x_path.exists() and y_path.exists()):
        return None, None, None

    x = np.load(str(x_path)).astype(np.float64)
    y = np.load(str(y_path)).astype(np.float64).ravel()

    metadata: Optional[Dict[str, Any]] = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    return x, y, metadata


def _ensure_smiles_vocab() -> None:
    """Place the SMILES vocab file where ``design_bench`` expects it.

    ``design_bench`` eagerly registers ChEMBL tasks on import, which requires
    a ``smiles_vocab.txt`` file.  The original GCS download URL is dead, so we
    ship a local copy in ``data/smiles_vocab.txt`` and copy it into the
    installed package at runtime if it is missing.
    """
    import shutil
    import site

    # Locate design_bench_data dir — it is a *sibling* of the design_bench
    # package directory (i.e. <site-packages>/design_bench_data/), NOT inside it.
    db_data: Optional[Path] = None
    for base in (site.getsitepackages() if hasattr(site, "getsitepackages") else []):
        pkg = Path(base) / "design_bench"
        if pkg.exists():
            db_data = Path(base) / "design_bench_data"
            break
    if db_data is None:
        return  # design_bench not installed

    vocab_dest = db_data / "smiles_vocab.txt"
    if vocab_dest.exists():
        return

    # Try local bundled copy
    local_vocab = _repo_root() / "data" / "smiles_vocab.txt"
    if local_vocab.exists():
        db_data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_vocab), str(vocab_dest))
        return

    # Try downloading from rxnfp GitHub
    try:
        import urllib.request
        url = (
            "https://raw.githubusercontent.com/rxn4chemistry/rxnfp/"
            "master/rxnfp/models/transformers/bert_ft/vocab.txt"
        )
        db_data.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(vocab_dest))
    except Exception:  # noqa: BLE001
        pass  # best-effort; design_bench import may still fail


def _ensure_design_bench_importable() -> None:
    """Install mock modules for optional MuJoCo dependencies.

    ``design_bench`` eagerly imports **all** task modules at startup,
    including ``morphing_agents`` / ``robel`` (needed only for Ant and
    DKitty MuJoCo tasks).  When those packages are absent, the entire
    ``import design_bench`` fails — even for tasks that do not need them
    (TFBind, Superconductor, ChEMBL).

    This function installs a lightweight meta-path finder that provides
    mock modules for the missing packages so that ``import design_bench``
    succeeds.  Ant / DKitty tasks will report errors at *task creation*
    time rather than blocking every other task.
    """
    import sys
    import types as _types

    _OPTIONAL_PKGS = ("robel", "morphing_agents")

    # ------------------------------------------------------------------
    # Detect which packages are truly unusable.  We catch *any* exception
    # (not just ImportError) because a package might be installed yet
    # broken at runtime (e.g. morphing_agents installed without robel
    # causes TypeError deep inside its sub-modules).
    # ------------------------------------------------------------------
    missing: list[str] = []
    for pkg in _OPTIONAL_PKGS:
        if pkg in sys.modules:
            continue
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)

    # morphing_agents depends on robel; if robel is missing/broken we
    # must also mock morphing_agents even if its top-level import works.
    if "robel" in missing and "morphing_agents" not in missing:
        missing.append("morphing_agents")

    if not missing:
        return

    # Purge any partially-loaded modules so our mock can replace them.
    for pkg in missing:
        to_remove = [k for k in sys.modules
                     if k == pkg or k.startswith(pkg + ".")]
        for k in to_remove:
            del sys.modules[k]

    # ------------------------------------------------------------------
    # _MockAttr is a plain class (not a ModuleType!) so that it can
    # safely be used as a base class, a callable, a value, etc. inside
    # design_bench's dataset/oracle definitions.
    class _MockAttr:
        """Universal stand-in returned for any attribute of a mock module."""

        def __init__(self, *args, **kwargs):
            pass

        def __init_subclass__(cls, **kwargs):
            pass  # allow ``class Foo(_MockAttr): ...``

        def __mro_entries__(self, bases):
            # When a _MockAttr *instance* ends up in a class's base list
            # (e.g. via a mocked decorator that replaces a real class),
            # Python 3.7+ calls __mro_entries__.  Returning () removes
            # the mock from the bases (the class inherits from object).
            return ()

        def __getattr__(self, name):
            return _MockAttr

        def __call__(self, *args, **kwargs):
            return _MockAttr()

        def __iter__(self):
            return iter([])

        def __len__(self) -> int:
            return 0

        def __bool__(self) -> bool:
            return True

        def __repr__(self) -> str:
            return "<mock>"

    class _MockModule(_types.ModuleType):
        """Module mock whose attributes are :class:`_MockAttr` objects."""

        def __getattr__(self, name: str):  # type: ignore[override]
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _MockAttr

    class _MockFinder:
        """Meta-path finder/loader for missing optional packages."""

        def __init__(self, prefixes: list[str]):
            self.prefixes = tuple(prefixes)

        def find_module(self, fullname: str, path=None):  # noqa: D401
            for pfx in self.prefixes:
                if fullname == pfx or fullname.startswith(pfx + "."):
                    return self
            return None

        def load_module(self, fullname: str):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = _MockModule(fullname)
            mod.__path__ = []  # type: ignore[attr-defined]
            mod.__loader__ = self  # type: ignore[assignment]
            mod.__file__ = f"<mock {fullname}>"
            mod.__package__ = fullname
            sys.modules[fullname] = mod
            return mod

    # Install once
    if not any(isinstance(f, _MockFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MockFinder(missing))


def _try_make_task(task_name: str) -> Any:
    """Try to create a ``design_bench`` task object.

    Returns the task or ``None`` if ``design_bench`` is unavailable.
    """
    # Bootstrap the SMILES vocab and mock missing deps before importing
    _ensure_smiles_vocab()
    _ensure_design_bench_importable()
    try:
        import design_bench  # type: ignore
        return design_bench.make(task_name)
    except Exception:  # noqa: BLE001 – broad on purpose
        return None


def _load_data_from_task(task: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``(x, y)`` arrays from a ``design_bench`` task object."""
    x: np.ndarray = np.asarray(task.x, dtype=np.float64)
    y: np.ndarray = np.asarray(task.y, dtype=np.float64).ravel()
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x, y


def _nearest_neighbour_lookup(
    queries: np.ndarray,
    database: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Return ``values`` of the nearest database row for each query.

    Uses vectorised NumPy with chunking to limit memory usage.
    """
    n_query = queries.shape[0]
    scores = np.empty(n_query, dtype=np.float64)
    chunk = 256
    for start in range(0, n_query, chunk):
        end = min(start + chunk, n_query)
        # (chunk, 1, dim) - (1, n_db, dim) -> (chunk, n_db)
        diffs = queries[start:end, None, :] - database[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        scores[start:end] = values[nearest_idx]
    return scores


# ---------------------------------------------------------------------------
# Base Design-Bench benchmark
# ---------------------------------------------------------------------------

class DesignBenchBenchmark(BaseBenchmark):
    """Abstract base wrapper for Design-Bench benchmarks.

    Concrete subclasses only need to set the class-level constants
    ``BENCHMARK_NAME``, ``DESIGN_BENCH_TASK``, and ``IS_DISCRETE``.
    """

    # ---- Override in subclasses -------------------------------------------
    BENCHMARK_NAME: str = ""
    DESIGN_BENCH_TASK: str = ""
    IS_DISCRETE: bool = False
    ALPHABET_SIZE: int = 4  # only for discrete tasks
    # Legacy data directory names for backward compatibility
    _LEGACY_DATA_DIRS: Tuple[str, ...] = ()

    def __init__(
        self,
        input_dim: Optional[int] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        invalid_penalty: float = 1e6,
    ) -> None:
        self._invalid_penalty = float(invalid_penalty)
        # Allow subclasses (e.g. CHEMBL) to pre-set the oracle before
        # calling super().__init__().
        if not hasattr(self, "_oracle"):
            self._oracle: Any = None
        self._x_data: Optional[np.ndarray] = None
        self._y_data: Optional[np.ndarray] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._seq_length: Optional[int] = None

        # 1. Try loading cached local data (check primary + legacy dirs)
        x_local, y_local, meta_local = _load_local_data(self.BENCHMARK_NAME)
        if x_local is None:
            for legacy_name in self._LEGACY_DATA_DIRS:
                x_local, y_local, meta_local = _load_local_data(legacy_name)
                if x_local is not None:
                    break
        if x_local is not None and y_local is not None:
            self._x_data = x_local
            self._y_data = y_local
            self._metadata = meta_local

        # 2. Try setting up the design_bench oracle (skip if subclass
        #    already resolved it).
        if self._oracle is None:
            self._oracle = _try_make_task(self.DESIGN_BENCH_TASK)

        # 3. If no local data, try getting it from the oracle object
        if self._x_data is None and self._oracle is not None:
            self._x_data, self._y_data = _load_data_from_task(self._oracle)

        # 4. Check at least *some* data source is available
        if self._x_data is None:
            raise RuntimeError(
                f"Cannot initialise benchmark '{self.BENCHMARK_NAME}'.  "
                f"Either install design-bench (pip install design-bench) "
                f"or download data first:\n"
                f"  python scripts/download_data.py "
                f"--benchmarks {self.BENCHMARK_NAME}"
            )

        # --- Discrete-task bookkeeping ---
        if self.IS_DISCRETE:
            # Always detect encoding from the actual data to avoid stale
            # or incorrect metadata (e.g. sequence_length computed from
            # integer-encoded data as shape[1]//ALPHABET_SIZE).
            self._seq_length = self._detect_seq_length(self._x_data)
            expected_dim = self._seq_length * self.ALPHABET_SIZE
            if self._x_data.shape[1] != expected_dim:
                # Data is integer-encoded → convert to one-hot
                self._x_data = self._integers_to_onehot(self._x_data)

        # --- Determine actual input dimensionality ---
        actual_dim = self._x_data.shape[1]
        if input_dim is not None and input_dim != actual_dim:
            warnings.warn(
                f"[{self.BENCHMARK_NAME}] Provided input_dim={input_dim} "
                f"differs from task dimension {actual_dim}.  "
                f"Using {actual_dim}.",
                stacklevel=2,
            )
        input_dim = actual_dim

        # --- Compute bounds ---
        if bounds is None:
            if self.IS_DISCRETE:
                lower = np.zeros(input_dim, dtype=np.float64)
                upper = np.ones(input_dim, dtype=np.float64)
            else:
                data_min = self._x_data.min(axis=0)
                data_max = self._x_data.max(axis=0)
                data_range = data_max - data_min
                padding = np.maximum(0.1 * data_range, 1e-6)
                lower = data_min - padding
                upper = data_max + padding
            bounds = (lower, upper)

        # --- Dataset statistics (convenience) ---
        self.y_min: float = float(self._y_data.min())
        self.y_max: float = float(self._y_data.max())
        self.y_mean: float = float(self._y_data.mean())
        self.y_std: float = float(self._y_data.std())
        self.dataset_size: int = len(self._y_data)

        # --- Validate oracle ---
        # Test the oracle on a known data point to ensure it actually works.
        # Some oracles are created successfully (e.g. via mock dependencies)
        # but fail at prediction time (e.g. Ant/DKitty without MuJoCo).
        if self._oracle is not None:
            self._oracle = self._validate_oracle(self._oracle)

        super().__init__(
            name=self.BENCHMARK_NAME,
            input_dim=input_dim,
            output_dim=1,
            bounds=bounds,
        )

        mode = "oracle" if self._oracle is not None else "lookup"
        print(
            f"[{self.BENCHMARK_NAME}] dim={input_dim}, "
            f"n_data={self.dataset_size}, "
            f"y∈[{self.y_min:.4f}, {self.y_max:.4f}], "
            f"mode={mode}"
        )

    # ------------------------------------------------------------------
    # Oracle validation
    # ------------------------------------------------------------------

    def _validate_oracle(self, oracle: Any) -> Any:
        """Test the oracle on known data points to verify it works.

        Returns the oracle if validation passes, or ``None`` if the
        oracle is broken (triggering fallback to lookup mode).

        Checks performed:
        1. The oracle can actually call ``predict()`` without exceptions.
        2. Predictions are finite numbers (not NaN/inf).
        3. Predictions are within a reasonable range of the training data
           (catches wildly extrapolating surrogate models on mock data).
        """
        if self._x_data is None or self._y_data is None:
            return oracle

        # Pick a small sample of known data points for testing
        n_test = min(5, len(self._x_data))
        test_indices = np.linspace(
            0, len(self._x_data) - 1, n_test, dtype=int,
        )
        x_test = self._x_data[test_indices].copy()
        y_expected = self._y_data[test_indices].copy()

        # For discrete tasks, convert one-hot → integers for the oracle
        if self.IS_DISCRETE and self._seq_length is not None:
            try:
                x_eval = self._onehot_to_integers(x_test)
            except Exception:
                x_eval = x_test.astype(np.float32)
        else:
            x_eval = x_test.astype(np.float32)

        try:
            preds = np.asarray(
                oracle.predict(x_eval), dtype=np.float64,
            ).ravel()
        except Exception as exc:
            warnings.warn(
                f"[{self.BENCHMARK_NAME}] Oracle validation failed "
                f"({type(exc).__name__}: {exc}).  "
                f"Falling back to lookup mode.",
                stacklevel=2,
            )
            return None

        # Check for non-finite predictions
        if not np.all(np.isfinite(preds)):
            warnings.warn(
                f"[{self.BENCHMARK_NAME}] Oracle returned non-finite "
                f"predictions.  Falling back to lookup mode.",
                stacklevel=2,
            )
            return None

        # Check predictions are in a reasonable range.  Allow a generous
        # margin (10x the data range) to accommodate different oracle
        # scales, but catch completely broken oracles (e.g. all zeros
        # when data is non-zero, or values millions of times larger).
        y_range = self.y_max - self.y_min
        margin = max(10.0 * y_range, 1.0)
        if np.any(preds < self.y_min - margin) or np.any(
            preds > self.y_max + margin
        ):
            warnings.warn(
                f"[{self.BENCHMARK_NAME}] Oracle predictions on known "
                f"data are wildly out of range "
                f"(pred range [{preds.min():.4f}, {preds.max():.4f}] "
                f"vs data range [{self.y_min:.4f}, {self.y_max:.4f}]).  "
                f"Falling back to lookup mode.",
                stacklevel=2,
            )
            return None

        # Extra check: if all predictions are identical (e.g. all zeros)
        # but the expected values vary, the oracle is likely broken.
        if np.std(preds) < 1e-10 and np.std(y_expected) > 1e-6:
            warnings.warn(
                f"[{self.BENCHMARK_NAME}] Oracle returns constant "
                f"predictions ({preds[0]:.6f}) on data with varying "
                f"targets.  Falling back to lookup mode.",
                stacklevel=2,
            )
            return None

        return oracle

    # ------------------------------------------------------------------
    # Discrete ↔ continuous helpers
    # ------------------------------------------------------------------

    def _detect_seq_length(self, x: np.ndarray) -> int:
        """Infer the sequence length by detecting integer vs one-hot encoding.

        One-hot data contains only 0s and 1s and has
        ``shape[1] == seq_length * ALPHABET_SIZE``.
        Integer data contains values in ``{0, …, ALPHABET_SIZE-1}`` and has
        ``shape[1] == seq_length``.

        A sample of up to 1 000 rows is inspected to keep cost low.
        """
        sample = x[: min(1000, len(x))]
        unique_vals = set(np.unique(sample).tolist())
        is_onehot = (
            unique_vals.issubset({0.0, 1.0})
            and sample.shape[1] % self.ALPHABET_SIZE == 0
        )
        if is_onehot:
            # Verify structure: each group of ALPHABET_SIZE columns sums to 1
            candidate_seq = sample.shape[1] // self.ALPHABET_SIZE
            groups = sample.reshape(sample.shape[0], candidate_seq, self.ALPHABET_SIZE)
            if np.allclose(groups.sum(axis=2), 1.0):
                return candidate_seq
        # Integer-encoded: each column is one sequence position
        return x.shape[1]

    def _integers_to_onehot(self, x_int: np.ndarray) -> np.ndarray:
        """Convert integer-encoded sequences to flattened one-hot vectors.

        Args:
            x_int: Shape ``(n, seq_length)`` with values in
                   ``{0, …, ALPHABET_SIZE-1}``.

        Returns:
            One-hot array of shape ``(n, seq_length * ALPHABET_SIZE)``.
        """
        x = np.asarray(x_int, dtype=np.int64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n, seq_len = x.shape
        one_hot = np.zeros((n, seq_len, self.ALPHABET_SIZE), dtype=np.float64)
        rows = np.arange(n)[:, None]
        cols = np.arange(seq_len)[None, :]
        one_hot[rows, cols, x] = 1.0
        return one_hot.reshape(n, -1)

    def _onehot_to_integers(self, x_oh: np.ndarray) -> np.ndarray:
        """Convert flattened one-hot vectors to integer-encoded sequences.

        Args:
            x_oh: Shape ``(n, seq_length * ALPHABET_SIZE)``.

        Returns:
            Integer array of shape ``(n, seq_length)``.
        """
        x = np.asarray(x_oh, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x_3d = x.reshape(n, self._seq_length, self.ALPHABET_SIZE)
        return np.argmax(x_3d, axis=-1).astype(np.int64)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the black-box function.

        Scores are **negated** so that the minimisation framework in
        ``main.py`` effectively *maximises* the Design-Bench objective.

        Args:
            x: Input array – ``(n, input_dim)`` or ``(input_dim,)``.
               For discrete tasks this should be in one-hot (continuous)
               representation.

        Returns:
            Negated oracle scores: ``(n, 1)`` or ``(1,)`` for single input.
        """
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, got {x.shape[1]}"
            )

        if self._oracle is not None:
            scores = self._evaluate_oracle(x)
        else:
            scores = self._evaluate_lookup(x)

        # Negate for minimisation
        values = -scores

        if single:
            return values[:1]  # shape (1,)
        return values.reshape(-1, 1)  # shape (n, 1)

    def _evaluate_oracle(self, x: np.ndarray) -> np.ndarray:
        """Evaluate using the ``design_bench`` oracle.

        Predictions are clamped to a reasonable range around the training
        data statistics to prevent wildly extrapolating surrogate models
        from producing meaningless scores on out-of-distribution inputs.
        """
        if self.IS_DISCRETE:
            x_eval = self._onehot_to_integers(x)
        else:
            x_eval = x.astype(np.float32)

        try:
            scores = np.asarray(
                self._oracle.predict(x_eval), dtype=np.float64,
            ).ravel()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[{self.BENCHMARK_NAME}] Oracle evaluation failed "
                f"({type(exc).__name__}: {exc}). "
                f"Falling back to lookup for {x.shape[0]} queries.",
                stacklevel=2,
            )
            return self._evaluate_lookup(x)

        # Clamp predictions to a generous but finite range around the
        # training data to catch surrogate-model extrapolation.
        y_range = self.y_max - self.y_min
        clamp_margin = max(2.0 * y_range, 1.0)
        clamp_lo = self.y_min - clamp_margin
        clamp_hi = self.y_max + clamp_margin
        scores = np.clip(scores, clamp_lo, clamp_hi)

        return scores

    def _evaluate_lookup(self, x: np.ndarray) -> np.ndarray:
        """Evaluate using nearest-neighbour lookup on the offline dataset."""
        return _nearest_neighbour_lookup(x, self._x_data, self._y_data)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_random(self, n: int) -> np.ndarray:
        """Sample *n* random valid inputs from the design space.

        For discrete tasks, random one-hot vectors are generated (each
        position has exactly one active category).
        """
        if self.IS_DISCRETE:
            random_ints = np.random.randint(
                0, self.ALPHABET_SIZE, (n, self._seq_length),
            )
            return self._integers_to_onehot(random_ints)

        lower, upper = self.bounds
        return np.random.uniform(lower, upper, size=(n, self.input_dim))

    def sample_initial_data(self, n: int) -> np.ndarray:
        """Sample *n* initial designs from the offline dataset.

        This provides a warm-start for online optimisation, replicating the
        experimental setup of the Diffusion-BBO paper.
        """
        dataset_size = len(self._x_data)
        n_actual = min(n, dataset_size)
        indices = np.random.choice(dataset_size, n_actual, replace=False)
        return self._x_data[indices].copy()

    # ------------------------------------------------------------------
    # Validity checking
    # ------------------------------------------------------------------

    def is_valid(self, x: np.ndarray) -> np.ndarray:
        """Check whether inputs are finite and within bounds."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        finite = np.all(np.isfinite(x), axis=1)
        if self.bounds is not None:
            lower, upper = self.bounds
            within = np.all((x >= lower) & (x <= upper), axis=1)
            valid = finite & within
        else:
            valid = finite
        return valid[0] if len(valid) == 1 else valid

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the full offline dataset ``(x, y)`` in continuous form."""
        return self._x_data.copy(), self._y_data.copy()

    def __repr__(self) -> str:
        mode = "oracle" if self._oracle is not None else "lookup"
        return (
            f"{type(self).__name__}("
            f"dim={self.input_dim}, "
            f"n_data={self.dataset_size}, "
            f"mode={mode})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Concrete task implementations
# ═══════════════════════════════════════════════════════════════════════════

class TFBIND8(DesignBenchBenchmark):
    """TFBind8 – Transcription factor binding for 8-mer DNA sequences.

    The design space consists of DNA sequences of length 8 over the
    alphabet {A, C, G, T} (4 nucleotides).  The optimizer works in a
    continuous one-hot representation of dimension 8 × 4 = **32**.

    The oracle is an *exact* lookup table over all 4^8 = 65 536 possible
    sequences.

    Design-Bench task: ``TFBind8-Exact-v0``
    """

    BENCHMARK_NAME = "tfbind8"
    DESIGN_BENCH_TASK = "TFBind8-Exact-v0"
    IS_DISCRETE = True
    ALPHABET_SIZE = 4
    _LEGACY_DATA_DIRS = ("tfbind",)  # backward compat with older downloads


class TFBIND10(DesignBenchBenchmark):
    """TFBind10 – Transcription factor binding for 10-mer DNA sequences.

    Same as :class:`TFBIND8` but for sequences of length 10.
    Continuous one-hot dimension: 10 × 4 = **40**.

    Design-Bench task: ``TFBind10-Exact-v0``
    """

    BENCHMARK_NAME = "tfbind10"
    DESIGN_BENCH_TASK = "TFBind10-Exact-v0"
    IS_DISCRETE = True
    ALPHABET_SIZE = 4


class SUPERCON(DesignBenchBenchmark):
    """Superconductor – Critical temperature prediction.

    The design space consists of 86-dimensional material composition
    descriptors.  The oracle is a Random Forest model trained on the UCI
    Superconductor dataset.

    Design-Bench task: ``Superconductor-RandomForest-v0``
    """

    BENCHMARK_NAME = "superconductor"
    DESIGN_BENCH_TASK = "Superconductor-RandomForest-v0"
    IS_DISCRETE = False


class ANT(DesignBenchBenchmark):
    """Ant Morphology – Designing ant robot morphology for locomotion.

    The design space consists of 60-dimensional morphology parameters that
    define the body of an Ant robot in MuJoCo.  The oracle evaluates the
    locomotion performance via simulation.

    Design-Bench task: ``AntMorphology-Exact-v0``
    """

    BENCHMARK_NAME = "ant"
    DESIGN_BENCH_TASK = "AntMorphology-Exact-v0"
    IS_DISCRETE = False


class DKITTY(DesignBenchBenchmark):
    """D'Kitty Morphology – Designing D'Kitty robot morphology.

    The design space consists of 56-dimensional morphology parameters.
    The oracle evaluates locomotion performance via MuJoCo simulation.

    Design-Bench task: ``DKittyMorphology-Exact-v0``
    """

    BENCHMARK_NAME = "dkitty"
    DESIGN_BENCH_TASK = "DKittyMorphology-Exact-v0"
    IS_DISCRETE = False


class CHEMBL(DesignBenchBenchmark):
    """ChEMBL – Molecular activity prediction (drug discovery).

    The design space consists of Morgan fingerprint vectors representing
    molecules.  The oracle is a Random Forest model trained on ChEMBL
    bioactivity data for assay CHEMBL3885882.

    Design-Bench task:
        ``ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0``

    .. note::

       If the default assay is unavailable, the benchmark falls back to
       the shorter alias ``ChEMBL-ResNet-v0`` which some ``design_bench``
       versions provide.  You can also download data manually with
       ``python scripts/download_data.py --benchmarks chembl``.
    """

    BENCHMARK_NAME = "chembl"
    # Primary task name (standard Design-Bench registry)
    DESIGN_BENCH_TASK = (
        "ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0"
    )
    IS_DISCRETE = False

    # Allow fallback names for different design_bench versions
    _FALLBACK_TASK_NAMES = (
        "ChEMBL-ResNet-v0",
    )

    def __init__(
        self,
        input_dim: Optional[int] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        invalid_penalty: float = 1e6,
    ) -> None:
        # Try primary task name first, then fallbacks.
        # Store the oracle on self so the parent __init__ skips redundant
        # creation (it checks ``hasattr(self, '_oracle')``).
        original_task = self.DESIGN_BENCH_TASK
        self._oracle = _try_make_task(self.DESIGN_BENCH_TASK)
        if self._oracle is None:
            for fallback in self._FALLBACK_TASK_NAMES:
                self._oracle = _try_make_task(fallback)
                if self._oracle is not None:
                    self.DESIGN_BENCH_TASK = fallback
                    break

        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            invalid_penalty=invalid_penalty,
        )
        # Restore original task name for repr
        self.DESIGN_BENCH_TASK = original_task


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible aliases
# ═══════════════════════════════════════════════════════════════════════════

#: Legacy alias – defaults to the 8-mer variant used in most papers.
TFBIND = TFBIND8


# Convenience mapping for programmatic access
DESIGN_BENCH_REGISTRY: Dict[str, type] = {
    "tfbind8": TFBIND8,
    "tfbind10": TFBIND10,
    "tfbind": TFBIND8,
    "superconductor": SUPERCON,
    "supercon": SUPERCON,
    "ant": ANT,
    "dkitty": DKITTY,
    "chembl": CHEMBL,
}

__all__ = [
    "DesignBenchBenchmark",
    "TFBIND8",
    "TFBIND10",
    "TFBIND",
    "SUPERCON",
    "ANT",
    "DKITTY",
    "CHEMBL",
    "DESIGN_BENCH_REGISTRY",
]
