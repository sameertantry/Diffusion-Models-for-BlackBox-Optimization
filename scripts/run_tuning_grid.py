#!/usr/bin/env python
"""Expand a config with a ``_grid`` section into all parameter combinations
and run each as a separate COCO experiment.

Usage::

    python scripts/run_tuning_grid.py configs/tuning/A_percentile_annealing.json
    python scripts/run_tuning_grid.py configs/tuning/A_percentile_annealing.json --dry-run
    python scripts/run_tuning_grid.py configs/tuning/*.json

The ``_grid`` key in the config must be a dict mapping **dotted paths**
(e.g. ``"optimizer.conditioning.p_low"``) to lists of values to try.
The script computes the Cartesian product, patches a deep copy of the
config for each combination, and calls :func:`run_from_config`.

Metadata keys starting with ``_`` (``_grid``, ``_grid_notes``,
``_description``) are stripped before passing the config to the runner.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.coco_wrapper import run_from_config  # noqa: E402


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value inside a nested dict using a dotted path."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _get_nested(d: dict, dotted_key: str) -> Any:
    keys = dotted_key.split(".")
    for k in keys:
        d = d[k]
    return d


def _strip_meta(config: dict) -> dict:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def expand_grid(config: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return ``(patched_config, overrides_dict)`` for every grid point."""
    grid = config.get("_grid", {})
    if not grid:
        return [(_strip_meta(config), {})]

    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for combo in itertools.product(*value_lists):
        patched = _strip_meta(copy.deepcopy(config))
        overrides = {}
        for key, val in zip(keys, combo):
            _set_nested(patched, key, val)
            overrides[key] = val
        results.append((patched, overrides))
    return results


def _combo_suffix(overrides: Dict[str, Any]) -> str:
    """Human-readable short suffix for an override combination."""
    parts: List[str] = []
    for key, val in overrides.items():
        short = key.rsplit(".", 1)[-1]
        if isinstance(val, float):
            parts.append(f"{short}{val:g}")
        else:
            parts.append(f"{short}{val}")
    return "_".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Expand _grid configs and run COCO experiments.",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Path(s) to JSON config file(s) with optional _grid section.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid expansions without running experiments.",
    )
    args = parser.parse_args(argv)

    for cfg_path in args.configs:
        path = Path(cfg_path)
        with path.open("r") as fh:
            config = json.load(fh)

        grid_points = expand_grid(config)
        desc = config.get("_description", path.stem)
        print(f"\n{'='*72}")
        print(f"Config: {path.name}  —  {desc}")
        print(f"Grid points: {len(grid_points)}")
        print(f"{'='*72}")

        for i, (patched, overrides) in enumerate(grid_points, 1):
            suffix = _combo_suffix(overrides)
            base_folder = patched.get("output_folder", "DiffusionV2/tuning")
            patched["output_folder"] = f"{base_folder}/{suffix}" if suffix else base_folder

            print(f"\n[{i}/{len(grid_points)}] {suffix or 'default'}")
            for k, v in overrides.items():
                print(f"  {k} = {v}")

            if args.dry_run:
                print(f"  -> output_folder: {patched['output_folder']}")
                continue

            t0 = time.time()
            try:
                result = run_from_config(patched)
                elapsed = time.time() - t0
                print(f"  -> Done in {elapsed:.1f}s  |  results: {result}")
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  -> FAILED after {elapsed:.1f}s: {exc}")

    if args.dry_run:
        print("\n(dry-run: no experiments were executed)")


if __name__ == "__main__":
    main()
