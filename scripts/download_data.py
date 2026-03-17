"""Download and prepare offline benchmark datasets.

This script explicitly downloads Design-Bench datasets and stores them under
data/<benchmark_name>/ with X.npy, y.npy, and metadata.json for offline usage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# NumPy compatibility shim — design_bench uses deprecated aliases
# (np.bool, np.int, np.float, np.object, np.str, np.complex) which were
# removed in NumPy 1.24+.  Restore them before design_bench is imported.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# SMILES vocabulary bootstrap
# ---------------------------------------------------------------------------
# design_bench cannot even be *imported* without a SMILES vocab file because
# it eagerly registers ChEMBL tasks that depend on the MorganFingerprint
# feature extractor.  The original GCS download URL is dead, so we ship a
# local copy and copy it into the design_bench package directory at runtime.
# ---------------------------------------------------------------------------

_VOCAB_DOWNLOAD_URLS: Tuple[str, ...] = (
    # rxnfp GitHub (primary – verified working)
    "https://raw.githubusercontent.com/rxn4chemistry/rxnfp/"
    "master/rxnfp/models/transformers/bert_ft/vocab.txt",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _local_vocab_path() -> Path:
    """Return the path to the repo-bundled SMILES vocab file."""
    return _repo_root() / "data" / "smiles_vocab.txt"


def _find_design_bench_data_dir() -> Path:
    """Locate the ``design_bench_data`` directory used by design_bench.

    design_bench stores ``DATA_DIR`` as
    ``os.path.join(os.path.dirname(os.path.dirname(__file__)), 'design_bench_data')``
    which resolves to ``<site-packages>/design_bench_data/`` (a *sibling* of the
    ``design_bench`` package directory, **not** inside it).
    """
    import site

    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    for base in candidates:
        if base and Path(base).exists():
            pkg_dir = Path(base) / "design_bench"
            if pkg_dir.exists():
                # DATA_DIR is a sibling of the package dir, not inside it
                return Path(base) / "design_bench_data"
    raise FileNotFoundError("Could not locate design_bench package directory.")


def _ensure_smiles_vocab() -> None:
    """Make sure ``design_bench_data/smiles_vocab.txt`` exists.

    Strategy:
      1.  If the file already exists → nothing to do.
      2.  Copy the local repo-bundled copy if available.
      3.  Download from rxnfp GitHub as a last resort.
    """
    try:
        data_dir = _find_design_bench_data_dir()
    except FileNotFoundError:
        return  # design_bench not installed – nothing we can do here

    data_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = data_dir / "smiles_vocab.txt"
    if vocab_path.exists():
        return

    # --- Try local bundled copy ---
    local = _local_vocab_path()
    if local.exists():
        shutil.copy2(str(local), str(vocab_path))
        print(f"[vocab] Copied bundled smiles_vocab.txt → {vocab_path}")
        return

    # --- Try downloading ---
    import urllib.request

    for url in _VOCAB_DOWNLOAD_URLS:
        try:
            urllib.request.urlretrieve(url, str(vocab_path))
            print(f"[vocab] Downloaded smiles_vocab.txt → {vocab_path}")
            return
        except Exception:
            continue

    print(
        "[vocab] WARNING: Could not obtain smiles_vocab.txt. "
        "design_bench import may fail."
    )


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
    # ------------------------------------------------------------------
    class _MockAttr:
        """Universal stand-in returned for any attribute of a mock module."""

        def __init__(self, *args, **kwargs):
            pass

        def __init_subclass__(cls, **kwargs):
            pass

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

        def __getattr__(self, name: str):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _MockAttr

    class _MockFinder:
        def __init__(self, prefixes: list[str]):
            self.prefixes = tuple(prefixes)

        def find_module(self, fullname: str, path=None):
            for pfx in self.prefixes:
                if fullname == pfx or fullname.startswith(pfx + "."):
                    return self
            return None

        def load_module(self, fullname: str):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = _MockModule(fullname)
            mod.__path__ = []
            mod.__loader__ = self
            mod.__file__ = f"<mock {fullname}>"
            mod.__package__ = fullname
            sys.modules[fullname] = mod
            return mod

    if not any(isinstance(f, _MockFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MockFinder(missing))


# ---------------------------------------------------------------------------
# Task map
# ---------------------------------------------------------------------------

TASK_MAP: Dict[str, Tuple[str, str]] = {
    "tfbind": ("TFBind8-Exact-v0", "TFBind8 protein sequence design"),
    "tfbind8": ("TFBind8-Exact-v0", "TFBind8 protein sequence design (8-mer)"),
    "tfbind10": ("TFBind10-Exact-v0", "TFBind10 protein sequence design (10-mer)"),
    "chembl": (
        "ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0",
        "ChEMBL drug discovery (assay CHEMBL3885882)",
    ),
    "superconductor": ("Superconductor-RandomForest-v0", "Superconductor Tc prediction"),
    "ant": ("AntMorphology-Exact-v0", "Ant morphology design"),
    "dkitty": ("DKittyMorphology-Exact-v0", "DKitty morphology design"),
}

# Fallback ChEMBL task names for different design_bench versions
_CHEMBL_FALLBACKS: Tuple[str, ...] = (
    "ChEMBL-ResNet-v0",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download offline benchmark datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
        help=(
            "Comma-separated list: "
            "analytical,tfbind,tfbind8,tfbind10,chembl,superconductor,ant,dkitty,all"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing data if present",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return _repo_root()


def data_dir_for(name: str) -> Path:
    return repo_root() / "data" / name


def ensure_atomic_numpy_save(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
    tmp_path.replace(path)


def ensure_atomic_json_save(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def select_benchmarks(arg: str) -> Iterable[str]:
    items = [item.strip().lower() for item in arg.split(",") if item.strip()]
    if "all" in items:
        return ["analytical"] + list(TASK_MAP.keys())
    return items


# ---------------------------------------------------------------------------
# design_bench loader
# ---------------------------------------------------------------------------

def load_design_bench(task_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a Design-Bench task and return (x, y) arrays.

    Handles the SMILES vocab bootstrapping, MuJoCo mock injection, and
    ChEMBL fallbacks.
    """
    # Ensure the vocab file exists and optional deps are mocked BEFORE
    # importing design_bench.
    _ensure_smiles_vocab()
    _ensure_design_bench_importable()

    try:
        import design_bench as db  # type: ignore
    except ImportError as exc:
        msg = str(exc)
        # Distinguish "design-bench not installed" from "a dependency is
        # missing" (e.g. robel, morphing_agents).
        if "design_bench" in msg or "design-bench" in msg:
            raise ImportError(
                "Missing dependency 'design-bench'. Install it to download "
                "datasets:\n  pip install design-bench"
            ) from exc
        raise ImportError(
            f"design-bench import failed due to a missing dependency:\n"
            f"  {exc}\n"
            f"Install the missing package and try again."
        ) from exc
    except Exception as exc:
        if "WordPiece" in str(exc) or "vocab" in str(exc):
            raise RuntimeError(
                "design_bench import failed due to missing SMILES vocabulary.\n"
                "The bundled vocab file could not be placed correctly.\n"
                "Try manually running:\n"
                "  python -c \"from scripts.download_data import _ensure_smiles_vocab; "
                "_ensure_smiles_vocab()\"\n"
                f"Original error: {exc}"
            ) from exc
        raise

    # Create the task
    try:
        task = db.make(task_name)
    except Exception as make_exc:
        # Try fallback names (e.g. for ChEMBL across design_bench versions)
        if "ChEMBL" in task_name:
            for fallback in _CHEMBL_FALLBACKS:
                try:
                    task = db.make(fallback)
                    break
                except Exception:
                    continue
            else:
                raise make_exc
        else:
            raise

    # Extract arrays
    if hasattr(task, "x") and hasattr(task, "y"):
        x = np.asarray(task.x)
        y = np.asarray(task.y)
    elif hasattr(task, "dataset") and hasattr(task.dataset, "x") and hasattr(task.dataset, "y"):
        x = np.asarray(task.dataset.x)
        y = np.asarray(task.dataset.y)
    elif hasattr(task, "get_dataset"):
        dataset = task.get_dataset()
        x = np.asarray(dataset.x)
        y = np.asarray(dataset.y)
    else:
        raise RuntimeError(f"Unable to access data for task: {task_name}")

    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if y.ndim > 1:
        y = y.reshape(y.shape[0], -1)[:, 0]

    return x.astype(np.float64), y.astype(np.float64)


# ---------------------------------------------------------------------------
# Metadata / persistence
# ---------------------------------------------------------------------------

def build_metadata(
    name: str,
    description: str,
    source: str,
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, object]:
    lower = np.min(x, axis=0)
    upper = np.max(x, axis=0)
    metadata: Dict[str, object] = {
        "name": name,
        "description": description,
        "source": source,
        "input_dim": int(x.shape[1]),
        "output_dim": 1,
        "bounds": [lower.tolist(), upper.tolist()],
    }
    if name in ("tfbind", "tfbind8", "tfbind10"):
        alphabet_size = 4
        # Detect whether data is integer-encoded or already one-hot.
        # One-hot: only values {0, 1}, shape[1] divisible by alphabet_size,
        #          and each group of alphabet_size columns sums to 1.
        # Integer: values in {0, …, alphabet_size-1}, shape[1] == seq_length.
        sample = x[: min(1000, len(x))]
        unique_vals = set(np.unique(sample).tolist())
        is_onehot = (
            unique_vals.issubset({0.0, 1.0})
            and sample.shape[1] % alphabet_size == 0
        )
        if is_onehot:
            candidate_seq = sample.shape[1] // alphabet_size
            groups = sample.reshape(sample.shape[0], candidate_seq, alphabet_size)
            if not np.allclose(groups.sum(axis=2), 1.0):
                is_onehot = False

        if is_onehot:
            metadata["representation"] = "one_hot"
            metadata["sequence_length"] = int(x.shape[1] // alphabet_size)
        else:
            metadata["representation"] = "integer"
            metadata["sequence_length"] = int(x.shape[1])
        metadata["alphabet_size"] = alphabet_size
    return metadata


def write_offline_dataset(
    name: str,
    description: str,
    x: np.ndarray,
    y: np.ndarray,
    force: bool,
) -> None:
    data_dir = data_dir_for(name)
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    meta_path = data_dir / "metadata.json"

    if not force and x_path.exists() and y_path.exists() and meta_path.exists():
        print(f"[cache] Using cached data for {name} at {data_dir}")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(name, description, "Design-Bench", x, y)

    ensure_atomic_numpy_save(x_path, x)
    ensure_atomic_numpy_save(y_path, y)
    ensure_atomic_json_save(meta_path, metadata)
    print(f"[ok] Saved {name} data to {data_dir}")


def write_analytical_metadata(force: bool) -> None:
    name = "analytical"
    data_dir = data_dir_for(name)
    meta_path = data_dir / "metadata.json"

    if not force and meta_path.exists():
        print(f"[cache] Using cached metadata for {name} at {data_dir}")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "name": name,
        "description": "Analytical benchmark suite (no dataset required)",
        "source": "Built-in",
        "input_dim": None,
        "output_dim": 1,
        "bounds": None,
    }
    ensure_atomic_json_save(meta_path, metadata)
    print(f"[ok] Saved analytical metadata to {data_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    benchmarks = list(select_benchmarks(args.benchmarks))

    if "analytical" in benchmarks:
        write_analytical_metadata(force=args.force)

    _MUJOCO_TASKS = {"ant", "dkitty"}

    for name in benchmarks:
        if name == "analytical":
            continue
        if name not in TASK_MAP:
            raise ValueError(f"Unknown benchmark: {name}")
        task_name, description = TASK_MAP[name]
        print(f"[download] Fetching {name} from {task_name}")
        try:
            x, y = load_design_bench(task_name)
        except Exception as exc:
            print(f"[ERROR] Failed to load {name}: {exc}")
            if name in _MUJOCO_TASKS:
                print(
                    f"[hint]  '{name}' requires MuJoCo packages that are not "
                    f"installed.\n"
                    f"        To enable it, run:\n"
                    f"          pip install morphing-agents mujoco"
                )
            print(f"[skip]  Skipping {name} – continuing with remaining benchmarks.")
            continue
        write_offline_dataset(name, description, x, y, force=args.force)


if __name__ == "__main__":
    main()
