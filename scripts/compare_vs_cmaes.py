#!/usr/bin/env python3
"""Compare all complete experiments in a folder against CMA-ES.

Scans *folder* (recursively) for ``experiment_results.csv`` files that
contain exactly 144 rows (24 functions × 2 instances × 3 dimensions).
For each (function, dimension) pair the script picks the best diffusion
config (lowest median gap across instances) and compares it to CMA-ES.

Usage::

    python scripts/compare_vs_cmaes.py exdata/rm/optimal/stage-1
    python scripts/compare_vs_cmaes.py exdata/rm/optimal/stage-1 --cmaes exdata/predefence/CMA-ES
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BBOB_FUNCS = list(range(1, 25))
DIMS = [2, 5, 10]
EXPECTED_ROWS = 144  # 24 * 2 * 3

BBOB_GROUPS: Dict[int, str] = {}
for _fid in range(1, 6):
    BBOB_GROUPS[_fid] = "Sep"
for _fid in range(6, 10):
    BBOB_GROUPS[_fid] = "Mod"
for _fid in range(10, 15):
    BBOB_GROUPS[_fid] = "Ill"
for _fid in range(15, 20):
    BBOB_GROUPS[_fid] = "MM"
for _fid in range(20, 25):
    BBOB_GROUPS[_fid] = "Weak"

# ANSI colours for terminal output
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

# ---------------------------------------------------------------------------
# CSV loader (no pandas dependency)
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> List[Dict[str, str]]:
    """Load a CSV file into a list of dicts (stdlib only)."""
    import csv
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _median_gap(
    rows: List[Dict[str, str]],
    fid: int,
    dim: int,
) -> float:
    """Median gap across instances for a (function, dimension) pair."""
    vals = [
        float(r["gap"])
        for r in rows
        if int(r["function_id"]) == fid and int(r["dimension"]) == dim
    ]
    if not vals:
        return float("inf")
    return float(np.median(vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare diffusion experiments against CMA-ES.",
    )
    parser.add_argument(
        "folder",
        help="Root folder to scan for experiment_results.csv files.",
    )
    parser.add_argument(
        "--cmaes",
        default="exdata/predefence/CMA-ES",
        help="Path to CMA-ES experiment folder.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output in markdown table format (for reports).",
    )
    args = parser.parse_args(argv)

    root = Path(args.folder)
    cmaes_csv = Path(args.cmaes) / "experiment_results.csv"
    if not cmaes_csv.exists():
        sys.exit(f"CMA-ES results not found: {cmaes_csv}")

    cmaes_rows = _load_csv(cmaes_csv)

    # -- Discover complete experiments ------------------------------------
    experiments: Dict[str, List[Dict[str, str]]] = {}
    for csv_path in sorted(root.rglob("experiment_results.csv")):
        rows = _load_csv(csv_path)
        if len(rows) != EXPECTED_ROWS:
            continue
        # Derive short name from path relative to root
        rel = csv_path.parent.relative_to(root)
        name = str(rel).replace("/", " / ")
        experiments[name] = rows

    if not experiments:
        sys.exit(f"No complete experiments (144 rows) found under {root}")

    exp_names = sorted(experiments.keys())
    print(f"Found {len(exp_names)} complete experiments\n")

    # -- Build comparison matrix ------------------------------------------
    # matrix[fid][dim] = (best_name, best_gap, cma_gap)
    matrix: Dict[int, Dict[int, Tuple[str, float, float]]] = {}

    for fid in BBOB_FUNCS:
        matrix[fid] = {}
        for dim in DIMS:
            cma_gap = _median_gap(cmaes_rows, fid, dim)

            best_name = ""
            best_gap = float("inf")
            for name in exp_names:
                g = _median_gap(experiments[name], fid, dim)
                if g < best_gap:
                    best_gap = g
                    best_name = name

            matrix[fid][dim] = (best_name, best_gap, cma_gap)

    # -- Print results ----------------------------------------------------
    md = args.markdown

    # Header
    if md:
        print("| Func | Group | dim=2 | | dim=5 | | dim=10 | |")
        print("|------|-------|-------|-------|-------|-------|--------|-------|")
    else:
        hdr = f"{'Func':>5s} {'Grp':>4s}"
        for dim in DIMS:
            hdr += f" │ {'Best config':>30s} {'Gap':>8s} {'CMA':>8s} {'Ratio':>6s}"
        print(hdr)
        print("─" * len(hdr))

    wins_total = {d: 0 for d in DIMS}
    losses_total = {d: 0 for d in DIMS}

    for fid in BBOB_FUNCS:
        grp = BBOB_GROUPS[fid]

        if md:
            cells = [f"| f{fid} | {grp} "]
        else:
            line = f"f{fid:3d}  {grp:>4s}"

        for dim in DIMS:
            best_name, best_gap, cma_gap = matrix[fid][dim]
            ratio = best_gap / max(cma_gap, 1e-10)
            is_win = best_gap < cma_gap

            if is_win:
                wins_total[dim] += 1
            else:
                losses_total[dim] += 1

            # Shorten name for display
            short = best_name.replace("baseline_ema / ", "ema/").replace(
                "baseline / ", "base/"
            ).replace("ddim / ", "").replace("steps_", "s").replace(
                "epochs_", "e"
            )
            if len(short) > 30:
                short = short[:27] + "..."

            if md:
                winner_tag = "**DIFF**" if is_win else "CMA"
                cells.append(
                    f"| {short} | {winner_tag} gap={best_gap:.3f} vs {cma_gap:.3f} (×{ratio:.2f}) "
                )
            else:
                if is_win:
                    tag = f"{_GREEN}DIFF{_RESET}"
                else:
                    tag = f"{_RED}CMA {_RESET}"
                line += f" │ {short:>30s} {best_gap:8.3f} {cma_gap:8.3f} {ratio:6.2f} {tag}"

        if md:
            print("".join(cells) + "|")
        else:
            print(line)

    # -- Summary ----------------------------------------------------------
    print()
    if md:
        print("### Win/Loss Summary\n")
        print("| Dimension | Wins | Losses |")
        print("|-----------|------|--------|")
        for dim in DIMS:
            print(f"| dim={dim} | {wins_total[dim]}/24 | {losses_total[dim]}/24 |")
    else:
        print(f"{_BOLD}Win/Loss Summary{_RESET}")
        for dim in DIMS:
            w = wins_total[dim]
            l = losses_total[dim]
            print(f"  dim={dim:2d}:  {_GREEN}{w:2d} wins{_RESET}  /  {_RED}{l:2d} losses{_RESET}")

    # -- Best config per group summary ------------------------------------
    print()
    if md:
        print("### Best Config per Group\n")
        print("| Group | dim=2 best | dim=5 best | dim=10 best |")
        print("|-------|-----------|-----------|------------|")

    group_names = ["Sep", "Mod", "Ill", "MM", "Weak"]
    group_funcs = {
        "Sep": range(1, 6), "Mod": range(6, 10), "Ill": range(10, 15),
        "MM": range(15, 20), "Weak": range(20, 25),
    }

    for grp in group_names:
        if md:
            cells = [f"| {grp} "]
        else:
            line = f"  {grp:>4s}:"

        for dim in DIMS:
            # Count wins per config in this group
            config_wins: Dict[str, int] = {}
            for fid in group_funcs[grp]:
                best_name, best_gap, cma_gap = matrix[fid][dim]
                if best_name not in config_wins:
                    config_wins[best_name] = 0
                config_wins[best_name] += 1

            top_config = max(config_wins, key=config_wins.get)  # type: ignore[arg-type]
            short = top_config.replace("baseline_ema / ", "ema/").replace(
                "baseline / ", "base/"
            ).replace("ddim / ", "").replace("steps_", "s").replace(
                "epochs_", "e"
            )
            if len(short) > 25:
                short = short[:22] + "..."

            if md:
                cells.append(f"| {short} ({config_wins[top_config]}/{len(list(group_funcs[grp]))}) ")
            else:
                line += f"  d={dim}: {short} ({config_wins[top_config]}/{len(list(group_funcs[grp]))})"

        if md:
            print("".join(cells) + "|")
        else:
            print(line)


if __name__ == "__main__":
    main()
