#!/usr/bin/env python3
"""Build an interactive dashboard for comparing COCO/BBOB experiment folders.

The script recursively scans one or more roots (e.g. ``exdata``), discovers
experiment directories, loads per-(function, dimension, instance) records, and
generates a single HTML dashboard served via a local HTTP server for easy
access from Windows hosts.

Key comparison rule:
Only shared triplets across selected experiments are compared.
"""

from __future__ import annotations

import argparse
import base64
import csv
import http.server
import json
import math
import os
import re
import socketserver
import struct
import threading
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse, parse_qs


Key = Tuple[int, int, int]  # (function_id, dimension, instance_id)


@dataclass
class Record:
    function_id: int
    dimension: int
    instance_id: int
    gap: Optional[float]
    evaluations: Optional[float]
    converged: Optional[bool]
    final_target_hit: Optional[bool]

    @property
    def key(self) -> Key:
        return (self.function_id, self.dimension, self.instance_id)


@dataclass
class VerboseTrace:
    filename: str
    title: str
    function_id: Optional[int]
    dimension: Optional[int]
    instance_id: Optional[int]
    values: List[float]
    running_best: List[float]
    iter_indices: List[int]
    iter_best: List[float]
    iter_mean: List[float]
    iter_median: List[float]
    iter_std: List[float]
    iter_count: List[int]
    taus: List[Optional[float]]
    iter_taus: List[Optional[float]]


@dataclass
class Experiment:
    exp_id: str
    name: str
    display_name: str
    rel_path: str
    abs_path: Path
    source: str
    records: Dict[Key, Record]
    verbose_traces: List[VerboseTrace] = field(default_factory=list)


CSV_NAME = "experiment_results.csv"
INFO_GLOB = "bbobexp_f*.info"


def _to_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    low = value.lower()
    if low in {"nan", "none", "null"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _to_bool(value: str) -> Optional[bool]:
    value = (value or "").strip().lower()
    if not value:
        return None
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    return None


def discover_experiment_dirs(roots: Iterable[Path]) -> List[Path]:
    discovered: Set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for current, dirnames, filenames in os.walk(root):
            cur_path = Path(current)
            has_csv = CSV_NAME in filenames
            has_info = any(
                name.startswith("bbobexp_f") and name.endswith(".info")
                for name in filenames
            )
            if has_csv or has_info:
                discovered.add(cur_path)
                dirnames[:] = []
    return sorted(discovered)


def load_from_csv(csv_path: Path) -> Tuple[str, Dict[Key, Record]]:
    records: Dict[Key, Record] = {}
    method_name = ""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method_name = method_name or (row.get("method_name") or "").strip()
            f_id = _to_int(row.get("function_id", ""))
            dim = _to_int(row.get("dimension", ""))
            inst = _to_int(row.get("instance_id", ""))
            if f_id is None or dim is None or inst is None:
                continue
            rec = Record(
                function_id=f_id,
                dimension=dim,
                instance_id=inst,
                gap=_to_float(row.get("gap", "")),
                evaluations=_to_float(row.get("evaluations", "")),
                converged=_to_bool(row.get("converged", "")),
                final_target_hit=_to_bool(row.get("final_target_hit", "")),
            )
            records[rec.key] = rec
    return method_name, records


META_FUNC_RE = re.compile(r"funcId\s*=\s*(\d+)")
META_DIM_RE = re.compile(r"DIM\s*=\s*(\d+)")
META_PREC_RE = re.compile(r"Precision\s*=\s*([0-9eE.+-]+)")
META_ALG_RE = re.compile(r"algId\s*=\s*'([^']*)'")
RUN_TOKEN_RE = re.compile(r"^\s*(\d+)\s*:\s*([0-9eE.+-]+)\s*\|\s*([0-9eE.+-]+)\s*$")


def load_from_info_files(exp_dir: Path) -> Tuple[str, Dict[Key, Record]]:
    records: Dict[Key, Record] = {}
    method_name = ""
    info_files = sorted(exp_dir.glob(INFO_GLOB))
    for info_file in info_files:
        with info_file.open("r", encoding="utf-8") as handle:
            current_func: Optional[int] = None
            current_dim: Optional[int] = None
            current_precision: Optional[float] = None
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                if "suite" in line and "funcId" in line and "DIM" in line:
                    match_f = META_FUNC_RE.search(line)
                    match_d = META_DIM_RE.search(line)
                    match_p = META_PREC_RE.search(line)
                    match_a = META_ALG_RE.search(line)
                    current_func = int(match_f.group(1)) if match_f else None
                    current_dim = int(match_d.group(1)) if match_d else None
                    current_precision = (
                        _to_float(match_p.group(1)) if match_p else None
                    )
                    if match_a:
                        method_name = method_name or match_a.group(1).strip()
                    continue

                if ".dat" not in line or current_func is None or current_dim is None:
                    continue

                chunks = [part.strip() for part in line.split(",")]
                if len(chunks) < 2:
                    continue

                for token in chunks[1:]:
                    match_run = RUN_TOKEN_RE.match(token)
                    if not match_run:
                        continue
                    instance_id = int(match_run.group(1))
                    evaluations = _to_float(match_run.group(2))
                    gap = _to_float(match_run.group(3))
                    converged = (
                        (gap is not None and current_precision is not None and gap <= current_precision)
                        if current_precision is not None
                        else None
                    )
                    rec = Record(
                        function_id=current_func,
                        dimension=current_dim,
                        instance_id=instance_id,
                        gap=gap,
                        evaluations=evaluations,
                        converged=converged,
                        final_target_hit=converged,
                    )
                    records[rec.key] = rec
    return method_name, records


_FDI_RE = re.compile(r"f(\d+)_d(\d+)_i(\d+)")


def _load_npz_simple(path: Path) -> Optional[dict]:
    """Load arrays from an .npz file without requiring numpy at build time."""
    try:
        import numpy as np
        d = np.load(str(path))
        return {k: d[k].tolist() for k in d.keys()}
    except Exception:
        return None


def _parse_fdi_from_stem(stem: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract (function_id, dimension, instance_id) from filename stem."""
    m = _FDI_RE.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def _flatten(arr: list) -> list:
    return [v[0] if isinstance(v, list) else v for v in arr]


def _scan_verbose_metadata(
    exp_dir: Path,
    records: Dict[Key, Record],
) -> List[VerboseTrace]:
    """Scan verbose_logs dir for .npz files — only extract metadata, no heavy data."""
    vlog_dir = exp_dir / "verbose_logs"
    if not vlog_dir.is_dir():
        return []
    npz_files = sorted(vlog_dir.glob("*.npz"))
    if not npz_files:
        return []

    csv_keys: List[Optional[Key]] = []
    csv_path = exp_dir / CSV_NAME
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    fid = _to_int(row.get("function_id", ""))
                    dim = _to_int(row.get("dimension", ""))
                    iid = _to_int(row.get("instance_id", ""))
                    if fid is not None and dim is not None and iid is not None:
                        csv_keys.append((fid, dim, iid))
                    else:
                        csv_keys.append(None)
        except Exception:
            csv_keys = []

    traces: List[VerboseTrace] = []
    for file_idx, npz_path in enumerate(npz_files):
        fid, dim, iid = _parse_fdi_from_stem(npz_path.stem)
        if fid is None and file_idx < len(csv_keys) and csv_keys[file_idx] is not None:
            fid, dim, iid = csv_keys[file_idx]
        title = f"f{fid} d{dim} i{iid}" if fid is not None else npz_path.stem
        traces.append(VerboseTrace(
            filename=npz_path.name, title=title,
            function_id=fid, dimension=dim, instance_id=iid,
            values=[], running_best=[],
            iter_indices=[], iter_best=[], iter_mean=[], iter_median=[],
            iter_std=[], iter_count=[], taus=[], iter_taus=[],
        ))
    return traces


def _process_single_npz(npz_path: Path) -> Optional[dict]:
    """Load and fully process one .npz file into a JSON-serialisable dict."""
    data = _load_npz_simple(npz_path)
    if data is None or "values" not in data:
        return None

    values_flat = [float(v) for v in _flatten(data["values"])
                   if isinstance(v, (int, float)) and math.isfinite(v)]
    if not values_flat:
        return None

    running_best: List[float] = []
    best_so_far = float("inf")
    for v in values_flat:
        best_so_far = min(best_so_far, v)
        running_best.append(best_so_far)

    idx_flat = [int(v) for v in _flatten(data.get("indexes", []))
                if isinstance(v, (int, float))]
    tau_flat = _flatten(data.get("taus", []))
    tau_flat = [float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
                for v in tau_flat]

    iter_vals: Dict[int, List[float]] = {}
    iter_tau_map: Dict[int, Optional[float]] = {}
    for j, (idx_val, fval) in enumerate(zip(idx_flat, values_flat)):
        iter_vals.setdefault(idx_val, []).append(fval)
        if j < len(tau_flat) and tau_flat[j] is not None:
            iter_tau_map[idx_val] = tau_flat[j]

    sorted_iters = sorted(iter_vals.keys())

    def _median(lst):
        s = sorted(lst)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    def _std(lst):
        if len(lst) < 2:
            return 0.0
        mu = sum(lst) / len(lst)
        return (sum((v - mu) ** 2 for v in lst) / len(lst)) ** 0.5

    def _c(v, d=4):
        r = round(v, d)
        ir = int(r)
        return ir if ir == r else r

    return {
        "values": [_c(v, 1) for v in values_flat],
        "running_best": [_c(v, 2) for v in running_best],
        "iter_indices": sorted_iters,
        "iter_best": [_c(min(iter_vals[i]), 2) for i in sorted_iters],
        "iter_mean": [_c(sum(iter_vals[i]) / len(iter_vals[i]), 2) for i in sorted_iters],
        "iter_median": [_c(_median(iter_vals[i]), 2) for i in sorted_iters],
        "iter_std": [_c(_std(iter_vals[i]), 2) for i in sorted_iters],
        "iter_count": [len(iter_vals[i]) for i in sorted_iters],
        "iter_taus": [_c(iter_tau_map[i], 2) if iter_tau_map.get(i) is not None else None
                      for i in sorted_iters],
    }


def load_experiment(exp_dir: Path, project_root: Path, exp_index: int) -> Optional[Experiment]:
    csv_path = exp_dir / CSV_NAME
    if csv_path.exists():
        method_name, records = load_from_csv(csv_path)
        source = "csv"
    else:
        method_name, records = load_from_info_files(exp_dir)
        source = "info"

    if not records:
        return None

    rel_path = str(exp_dir.relative_to(project_root))
    name = method_name or exp_dir.name
    exp_id = f"exp_{exp_index:04d}"
    verbose_traces = _scan_verbose_metadata(exp_dir, records)
    return Experiment(
        exp_id=exp_id,
        name=name,
        display_name=name,
        rel_path=rel_path,
        abs_path=exp_dir,
        source=source,
        records=records,
        verbose_traces=verbose_traces,
    )


def choose_preselected(experiments: List[Experiment], explicit_paths: List[Path]) -> List[str]:
    if explicit_paths:
        normalized = {str(path.resolve()) for path in explicit_paths}
        selected = [
            exp.exp_id
            for exp in experiments
            if str(exp.abs_path.resolve()) in normalized
        ]
        return selected
    return [exp.exp_id for exp in experiments[:3]]


def assign_unique_display_names(experiments: List[Experiment]) -> None:
    by_name: Dict[str, List[Experiment]] = {}
    for exp in experiments:
        by_name.setdefault(exp.name, []).append(exp)
    for name, group in by_name.items():
        if len(group) == 1:
            group[0].display_name = name
            continue
        for exp in group:
            exp.display_name = f"{name} [{exp.rel_path}]"


def build_tree_structure(experiments: List[Experiment]) -> dict:
    """Build a nested tree from experiment relative paths for the tree UI."""
    tree: dict = {}
    for exp in experiments:
        parts = Path(exp.rel_path).parts
        node = tree
        for part in parts[:-1]:
            if part not in node:
                node[part] = {"__children": {}}
            elif "__children" not in node[part]:
                node[part]["__children"] = {}
            node = node[part]["__children"]
        leaf_name = parts[-1]
        if leaf_name not in node:
            node[leaf_name] = {}
        node[leaf_name]["__exp"] = {
            "id": exp.exp_id,
            "name": exp.name,
            "display_name": exp.display_name,
            "path": exp.rel_path,
            "source": exp.source,
            "total_records": len(exp.records),
        }
    return tree


def to_json_payload(experiments: List[Experiment], preselected_ids: List[str]) -> dict:
    tree = build_tree_structure(experiments)
    payload = {
        "experiments": [],
        "records": [],
        "preselectedExperimentIds": preselected_ids,
        "tree": tree,
    }
    for exp in experiments:
        exp_entry: Dict[str, Any] = {
            "id": exp.exp_id,
            "name": exp.name,
            "display_name": exp.display_name,
            "path": exp.rel_path,
            "source": exp.source,
            "total_records": len(exp.records),
            "has_verbose": len(exp.verbose_traces) > 0,
            "verbose_count": len(exp.verbose_traces),
        }
        if exp.verbose_traces:
            exp_entry["verbose_traces"] = [
                {
                    "title": t.title,
                    "function_id": t.function_id,
                    "dimension": t.dimension,
                    "instance_id": t.instance_id,
                    "filename": t.filename,
                }
                for t in exp.verbose_traces
            ]
        payload["experiments"].append(exp_entry)
        for rec in exp.records.values():
            payload["records"].append(
                {
                    "experiment_id": exp.exp_id,
                    "function_id": rec.function_id,
                    "dimension": rec.dimension,
                    "instance_id": rec.instance_id,
                    "gap": rec.gap,
                    "evaluations": rec.evaluations,
                    "converged": rec.converged,
                    "final_target_hit": rec.final_target_hit,
                }
            )
    return payload


BBOB_FUNCTION_NAMES = {
    1: "Sphere", 2: "Ellipsoidal (separable)", 3: "Rastrigin (separable)",
    4: "Büche-Rastrigin", 5: "Linear Slope", 6: "Attractive Sector",
    7: "Step Ellipsoidal", 8: "Rosenbrock (original)", 9: "Rosenbrock (rotated)",
    10: "Ellipsoidal (rotated)", 11: "Discus", 12: "Bent Cigar",
    13: "Sharp Ridge", 14: "Different Powers", 15: "Rastrigin (rotated)",
    16: "Weierstrass", 17: "Schaffers F7", 18: "Schaffers F7 (ill-cond.)",
    19: "Composite Griewank-Rosenbrock F8F2", 20: "Schwefel",
    21: "Gallagher's Gaussian 101-me Peaks", 22: "Gallagher's Gaussian 21-hi Peaks",
    23: "Katsuura", 24: "Lunacek bi-Rastrigin",
}


def render_dashboard_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=True)
    bbob_names_json = json.dumps(BBOB_FUNCTION_NAMES, ensure_ascii=True)
    # The HTML is a single self-contained file with embedded JS/CSS
    return _DASHBOARD_TEMPLATE.replace("__DATA_PAYLOAD__", data_json).replace("__BBOB_NAMES__", bbob_names_json)


_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>COCO/BBOB Experiment Comparison Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/tabulator-tables@6.3.0/dist/css/tabulator.min.css" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/tabulator-tables@6.3.0/dist/js/tabulator.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #f5f7fb;
      --surface: #ffffff;
      --surface2: #f0f2f8;
      --text: #1a1d2e;
      --text2: #555b72;
      --accent: #4361ee;
      --accent2: #3a56d4;
      --success: #10b981;
      --warn: #f59e0b;
      --danger: #ef4444;
      --border: #e2e5ef;
      --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
      --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
      --radius: 10px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.5;
    }

    .header {
      background: linear-gradient(135deg, #4361ee 0%, #3a56d4 50%, #2d46b9 100%);
      color: #fff;
      padding: 24px 32px;
    }
    .header h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.3px; }
    .header p { font-size: 13px; opacity: 0.85; margin-top: 4px; }

    .container { max-width: 1800px; margin: 0 auto; padding: 20px 24px 40px; }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }
    .card-header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
      border-radius: var(--radius) var(--radius) 0 0;
    }
    .card-header h2 { font-size: 14px; font-weight: 600; color: var(--text); }
    .card-body { padding: 16px; }

    .controls-grid {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
    }
    @media (max-width: 900px) { .controls-grid { grid-template-columns: 1fr; } }

    .tree-container {
      max-height: 400px;
      overflow-y: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      padding: 8px;
    }
    .tree-node { padding-left: 0; }
    .tree-folder {
      cursor: pointer;
      display: flex; align-items: center; gap: 4px;
      padding: 3px 6px;
      border-radius: 4px;
      font-size: 13px; font-weight: 500;
      color: var(--text2);
      user-select: none;
    }
    .tree-folder:hover { background: var(--surface2); }
    .tree-folder .arrow { display: inline-block; width: 16px; text-align: center; font-size: 10px; transition: transform 0.15s; }
    .tree-folder .arrow.collapsed { transform: rotate(-90deg); }
    .tree-folder .folder-icon { font-size: 14px; }
    .tree-children { padding-left: 18px; }
    .tree-children.hidden { display: none; }
    .tree-leaf {
      display: flex; align-items: center; gap: 6px;
      padding: 4px 6px;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
    }
    .tree-leaf:hover { background: #eef1fb; }
    .tree-leaf input { accent-color: var(--accent); cursor: pointer; }
    .tree-leaf .leaf-name { font-weight: 500; color: var(--text); }
    .tree-leaf .leaf-meta { color: var(--text2); font-size: 11px; }
    .tree-leaf .leaf-badge {
      display: inline-block; padding: 1px 5px;
      border-radius: 3px; font-size: 10px; font-weight: 600;
      background: #eef1fb; color: var(--accent);
    }

    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
    input[type="text"], select {
      font-family: inherit; font-size: 12px;
      padding: 7px 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
      color: var(--text);
      outline: none;
      transition: border-color 0.15s;
    }
    input[type="text"]:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(67,97,238,0.15); }
    .btn {
      font-family: inherit; font-size: 12px; font-weight: 500;
      padding: 7px 14px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
      color: var(--text);
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn:hover { border-color: var(--accent); color: var(--accent); }
    .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .btn-primary:hover { background: var(--accent2); border-color: var(--accent2); color: #fff; }
    .btn-sm { padding: 4px 10px; font-size: 11px; }
    .btn-icon {
      display: inline-flex; align-items: center; justify-content: center;
      width: 28px; height: 28px; padding: 0;
      border-radius: 6px; font-size: 14px;
    }

    .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }
    .kpi-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 16px;
      box-shadow: var(--shadow);
    }
    .kpi-label { font-size: 11px; color: var(--text2); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 26px; font-weight: 700; color: var(--text); margin-top: 2px; }

    .chart-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
      margin-bottom: 16px;
    }
    @media (max-width: 1000px) { .chart-grid { grid-template-columns: 1fr; } }
    .chart-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .chart-card-header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
      border-radius: var(--radius) var(--radius) 0 0;
    }
    .chart-card-header h3 { font-size: 13px; font-weight: 600; }
    .chart-card-actions { display: flex; gap: 4px; align-items: center; }
    .chart-area { height: 380px; }
    .chart-controls {
      display: flex; gap: 8px; align-items: center; padding: 6px 14px;
      border-bottom: 1px solid var(--border);
      background: #f8f9fd;
      flex-wrap: wrap;
    }
    .chart-controls label { font-size: 11px; font-weight: 500; color: var(--text2); }
    .chart-controls select {
      font-size: 11px; padding: 3px 6px; border-radius: 4px;
      max-width: 200px;
    }

    .chart-full .chart-area { height: 420px; }

    .table-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      margin-bottom: 16px;
      overflow: visible;
    }
    .table-card > div:last-child { padding: 0; }
    .table-card-header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
      border-radius: var(--radius) var(--radius) 0 0;
    }
    .table-card-header h3 { font-size: 13px; font-weight: 600; }
    .table-card-actions { display: flex; gap: 4px; }

    .tabulator { font-size: 12px; border: none !important; }
    .tabulator .tabulator-header { background: var(--surface2) !important; border-bottom: 2px solid var(--border) !important; }
    .tabulator .tabulator-header .tabulator-col { border-right: 1px solid var(--border) !important; }
    .tabulator .tabulator-tableHolder { min-height: 100px; }
    .tabulator .tabulator-row { border-bottom: 1px solid #f0f0f0 !important; }
    .tabulator .tabulator-row:hover { background: #f8f9fd !important; }
    .tabulator .tabulator-footer { background: var(--surface2) !important; border-top: 1px solid var(--border) !important; }
    .tabulator .tabulator-page.active { background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important; }

    .status-msg { font-size: 12px; color: var(--text2); padding: 6px 0; min-height: 24px; }
    .status-msg.error { color: var(--danger); }

    .modal-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.3); z-index: 9999;
      justify-content: center; align-items: center;
    }
    .modal-overlay.active { display: flex; }
    .modal-box {
      background: var(--surface);
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.2);
      max-width: 560px; width: 90%;
      max-height: 80vh;
      overflow-y: auto;
      padding: 24px;
    }
    .modal-box h3 { font-size: 16px; font-weight: 700; margin-bottom: 12px; }
    .modal-box p, .modal-box li { font-size: 13px; line-height: 1.7; color: var(--text2); margin-bottom: 8px; }
    .modal-box .lang-toggle {
      display: inline-flex; gap: 0; margin-bottom: 16px;
      border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
    }
    .modal-box .lang-btn {
      padding: 5px 14px; font-size: 12px; font-weight: 500;
      cursor: pointer; border: none; background: var(--surface);
      color: var(--text2); transition: all 0.15s;
    }
    .modal-box .lang-btn.active { background: var(--accent); color: #fff; }
    .modal-close {
      position: absolute; top: 12px; right: 14px;
      background: none; border: none; font-size: 18px;
      cursor: pointer; color: var(--text2);
    }
    .modal-content { position: relative; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #c5c9d8; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #a0a5b8; }

    .filter-group { display: flex; flex-direction: column; gap: 4px; }
    .filter-group label { font-size: 11px; font-weight: 500; color: var(--text2); }

    .palette-bar {
      display: flex; align-items: center; gap: 10px; padding: 8px 16px;
      background: var(--surface); border-bottom: 1px solid var(--border);
    }
    .palette-bar label { font-size: 11px; font-weight: 600; color: var(--text2); white-space: nowrap; }
    .palette-swatch {
      display: flex; gap: 2px; cursor: pointer; padding: 3px 6px;
      border: 2px solid transparent; border-radius: 6px; transition: all 0.15s;
    }
    .palette-swatch:hover { border-color: var(--text2); }
    .palette-swatch.active { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(67,97,238,0.15); }
    .palette-swatch span { width: 14px; height: 14px; border-radius: 3px; display: block; }

    .func-group-chips { display: flex; flex-wrap: wrap; gap: 4px; }
    .func-group-chip {
      font-size: 11px; padding: 3px 10px; border-radius: 12px;
      cursor: pointer; border: 1px solid var(--border); background: var(--surface);
      color: var(--text2); transition: all 0.15s; user-select: none;
    }
    .func-group-chip:hover { border-color: var(--accent); color: var(--accent); }
    .func-group-chip.active { background: var(--accent); color: #fff; border-color: var(--accent); }

    .verbose-filter-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 4px; }
    .verbose-filter-row .func-group-chip { font-size: 10px; padding: 2px 8px; }
  </style>
</head>
<body>
  <div class="header">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <h1>COCO/BBOB Experiment Comparison Dashboard</h1>
        <p>Compare optimization experiments on shared (function, dimension, instance) triplets. Only overlapping configurations are shown.</p>
      </div>
      <button class="btn btn-sm" id="btnReloadExperiments" style="background:rgba(255,255,255,0.2);color:#fff;border:1px solid rgba(255,255,255,0.3);font-size:12px;" title="Re-scan exdata for new experiments">&#x21bb; Reload experiments</button>
    </div>
  </div>

  <div class="palette-bar" id="paletteBar">
    <label>Color palette:</label>
    <div id="paletteOptions"></div>
  </div>

  <div class="container">
    <!-- Controls -->
    <div class="card">
      <div class="card-header">
        <h2>Experiment Selection & Filters</h2>
        <div style="display:flex;gap:6px;">
          <button class="btn btn-sm" id="btnSelectAll">Select All</button>
          <button class="btn btn-sm" id="btnClearAll">Clear</button>
          <button class="btn btn-primary btn-sm" id="btnRefresh">Compare Selected</button>
        </div>
      </div>
      <div class="card-body">
        <div class="controls-grid">
          <div>
            <div class="toolbar" style="margin-bottom:8px;">
              <input type="text" id="searchExperiments" placeholder="Search experiments..." style="flex:1;" />
            </div>
            <div id="experimentTree" class="tree-container"></div>
          </div>
          <div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px;">
              <div class="filter-group">
                <label>Metric</label>
                <select id="metricSelect">
                  <option value="gap">Gap (lower is better)</option>
                  <option value="evaluations">Evaluations (lower is better)</option>
                  <option value="success">Success rate (higher is better)</option>
                </select>
              </div>
              <div class="filter-group">
                <label>Function filter</label>
                <input type="text" id="functionFilter" placeholder="e.g. 1,2,5-8" />
              </div>
              <div class="filter-group">
                <label>Dimension filter</label>
                <input type="text" id="dimensionFilter" placeholder="e.g. 2,3,5" />
              </div>
              <div class="filter-group">
                <label>Instance filter</label>
                <input type="text" id="instanceFilter" placeholder="e.g. 1-5" />
              </div>
            </div>
            <div style="margin-bottom:12px;">
              <label style="font-size:11px;font-weight:500;color:var(--text2);display:block;margin-bottom:4px;">BBOB function groups (click to toggle, hover for description):
                <button class="btn btn-icon" title="Group descriptions" data-info="bbobGroups" style="font-size:10px;width:18px;height:18px;vertical-align:middle;">?</button>
              </label>
              <div class="func-group-chips" id="funcGroupChips"></div>
            </div>
            <!-- KPIs -->
            <div class="kpi-row">
              <div class="kpi-card"><div class="kpi-label">Selected Experiments</div><div class="kpi-value" id="kpiSelected">0</div></div>
              <div class="kpi-card"><div class="kpi-label">Shared Triplets</div><div class="kpi-value" id="kpiShared">0</div></div>
              <div class="kpi-card"><div class="kpi-label">Shared Functions</div><div class="kpi-value" id="kpiFunctions">0</div></div>
              <div class="kpi-card"><div class="kpi-label">Shared Dimensions</div><div class="kpi-value" id="kpiDims">0</div></div>
              <div class="kpi-card"><div class="kpi-label">Shared Instances</div><div class="kpi-value" id="kpiInstances">0</div></div>
            </div>
            <div class="status-msg" id="statusMessage"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Charts row 1 -->
    <div class="chart-grid" id="chartsRow1">
      <div class="chart-card">
        <div class="chart-card-header">
          <h3>Win Rate by Experiment</h3>
          <div class="chart-card-actions">
            <button class="btn btn-icon" title="Info" data-info="winRate">?</button>
            <button class="btn btn-icon" title="Save" data-export-plot="plotWins">&#128190;</button>
          </div>
        </div>
        <div id="plotWins" class="chart-area"></div>
      </div>
      <div class="chart-card">
        <div class="chart-card-header">
          <h3>Metric Distribution (log&#8321;&#8320; ratio to best)</h3>
          <div class="chart-card-actions">
            <button class="btn btn-icon" title="Info" data-info="relativeDistribution">?</button>
            <button class="btn btn-icon" title="Save" data-export-plot="plotRelative">&#128190;</button>
          </div>
        </div>
        <div id="plotRelative" class="chart-area"></div>
      </div>
    </div>

    <!-- Charts row 2: Dimension Scaling with per-function filter, Function Rank Heatmap with per-dimension filter -->
    <div class="chart-grid" id="chartsRow2">
      <div class="chart-card">
        <div class="chart-card-header">
          <h3>Dimension Scaling</h3>
          <div class="chart-card-actions">
            <button class="btn btn-icon" title="Info" data-info="dimensionScaling">?</button>
            <button class="btn btn-icon" title="Save" data-export-plot="plotDimension">&#128190;</button>
          </div>
        </div>
        <div class="chart-controls">
          <label>Function:</label>
          <select id="dimScalingFuncFilter"><option value="all">All functions</option></select>
        </div>
        <div id="plotDimension" class="chart-area"></div>
      </div>
      <div class="chart-card">
        <div class="chart-card-header">
          <h3>Function-wise Mean Rank Heatmap</h3>
          <div class="chart-card-actions">
            <button class="btn btn-icon" title="Info" data-info="functionRank">?</button>
            <button class="btn btn-icon" title="Save" data-export-plot="plotFunctionRank">&#128190;</button>
          </div>
        </div>
        <div class="chart-controls">
          <label>Dimension:</label>
          <select id="funcRankDimFilter"><option value="all">All dimensions</option></select>
        </div>
        <div id="plotFunctionRank" class="chart-area"></div>
      </div>
    </div>

    <!-- Charts row 3: ECDF with per-function and per-dimension filter, Convergence Rate -->
    <div class="chart-grid" id="chartsRow3">
      <div class="chart-card">
        <div class="chart-card-header">
          <h3>ECDF: Fraction of Problems Solved vs Gap Threshold</h3>
          <div class="chart-card-actions">
            <button class="btn btn-icon" title="Info" data-info="ecdf">?</button>
            <button class="btn btn-icon" title="Save" data-export-plot="plotECDF">&#128190;</button>
          </div>
        </div>
        <div class="chart-controls">
          <label>Function:</label>
          <select id="ecdfFuncFilter"><option value="all">All functions</option></select>
          <label>Dimension:</label>
          <select id="ecdfDimFilter"><option value="all">All dimensions</option></select>
        </div>
        <div id="plotECDF" class="chart-area"></div>
      </div>
      <div class="chart-card">
        <div class="chart-card-header">
          <h3>Convergence Rate by Dimension</h3>
          <div class="chart-card-actions">
            <button class="btn btn-icon" title="Info" data-info="convergenceRate">?</button>
            <button class="btn btn-icon" title="Save" data-export-plot="plotConvergence">&#128190;</button>
          </div>
        </div>
        <div class="chart-controls">
          <label>Function:</label>
          <select id="convFuncFilter"><option value="all">All functions</option></select>
        </div>
        <div id="plotConvergence" class="chart-area"></div>
      </div>
    </div>

    <!-- Charts row 4: Median Gap with per-dimension filter, Pairwise -->
    <div class="chart-grid" id="chartsRow4">
      <div class="chart-card">
        <div class="chart-card-header">
          <h3>Median Gap by Function (log scale)</h3>
          <div class="chart-card-actions">
            <button class="btn btn-icon" title="Info" data-info="medianGapByFunction">?</button>
            <button class="btn btn-icon" title="Save" data-export-plot="plotMedianGap">&#128190;</button>
          </div>
        </div>
        <div class="chart-controls">
          <label>Dimension:</label>
          <select id="medianGapDimFilter"><option value="all">All dimensions</option></select>
        </div>
        <div id="plotMedianGap" class="chart-area"></div>
      </div>
      <div class="chart-card">
        <div class="chart-card-header">
          <h3>Pairwise Head-to-Head Wins</h3>
          <div class="chart-card-actions">
            <button class="btn btn-icon" title="Info" data-info="pairwiseWins">?</button>
            <button class="btn btn-icon" title="Save" data-export-plot="plotPairwise">&#128190;</button>
          </div>
        </div>
        <div id="plotPairwise" class="chart-area"></div>
      </div>
    </div>

    <!-- Tables -->
    <div class="table-card" id="aggregateTableWrap">
      <div class="table-card-header">
        <h3>Aggregate Ranking Table</h3>
        <div class="table-card-actions">
          <button class="btn btn-icon" title="Info" data-info="aggregateTable">?</button>
          <button class="btn btn-sm" data-export-plot="plotAggTableFallback">Save as PNG</button>
        </div>
      </div>
      <div id="aggregateTable" style="min-height:120px;"></div>
    </div>

    <div class="table-card" id="functionSummaryTableWrap">
      <div class="table-card-header">
        <h3>Per-Function Summary</h3>
        <div class="table-card-actions">
          <button class="btn btn-icon" title="Info" data-info="functionSummaryTable">?</button>
          <button class="btn btn-sm" data-export-plot="plotFuncTableFallback">Save as PNG</button>
        </div>
      </div>
      <div id="functionSummaryTable" style="min-height:120px;"></div>
    </div>

    <div class="table-card" id="pairwiseTableWrap">
      <div class="table-card-header">
        <h3>Pairwise Win Matrix</h3>
        <div class="table-card-actions">
          <button class="btn btn-icon" title="Info" data-info="pairwiseTable">?</button>
          <button class="btn btn-sm" data-export-plot="plotPairTableFallback">Save as PNG</button>
        </div>
      </div>
      <div id="pairwiseTable" style="min-height:120px;"></div>
    </div>

    <div class="table-card" id="sharedTableWrap">
      <div class="table-card-header">
        <h3>Detailed Shared Keys Table</h3>
        <div class="table-card-actions">
          <button class="btn btn-icon" title="Info" data-info="detailedTable">?</button>
          <button class="btn btn-sm" data-export-plot="plotSharedTableFallback">Save as PNG</button>
        </div>
      </div>
      <div id="sharedTable" style="min-height:120px;"></div>
    </div>

    <!-- Verbose Convergence Logs -->
    <div class="card" id="verboseSection" style="display:none;">
      <div class="card-header">
        <h2>Convergence Traces (Verbose Logs)</h2>
        <button class="btn btn-icon" title="Info" data-info="verboseLogs">?</button>
      </div>
      <div class="card-body">
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
          <div class="filter-group">
            <label>Experiment</label>
            <select id="verboseExpSelect" style="min-width:200px;"></select>
          </div>
          <div class="filter-group">
            <label>Filter by function</label>
            <select id="verboseFuncFilter" style="min-width:120px;"><option value="all">All</option></select>
          </div>
          <div class="filter-group">
            <label>Filter by dimension</label>
            <select id="verboseDimFilter" style="min-width:100px;"><option value="all">All</option></select>
          </div>
        </div>
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px;">
          <div class="filter-group" style="min-width:320px;flex:1;">
            <label>Trace</label>
            <select id="verboseTraceSelect" style="width:100%;"></select>
          </div>
          <button class="btn btn-sm" id="verbosePrev" type="button">&larr; Prev</button>
          <button class="btn btn-sm" id="verboseNext" type="button">Next &rarr;</button>
          <span id="verboseCounter" style="font-size:12px;color:var(--text2);"></span>
        </div>
        <!-- Row 1: convergence + per-iteration -->
        <div class="chart-grid">
          <div class="chart-card">
            <div class="chart-card-header">
              <h3>Convergence (all evaluations)</h3>
              <div class="chart-card-actions">
                <button class="btn btn-icon" title="Info" data-info="verboseConvergence">?</button>
                <button class="btn btn-icon" title="Save" data-export-plot="plotVerboseAll">&#128190;</button>
              </div>
            </div>
            <div id="plotVerboseAll" class="chart-area"></div>
          </div>
          <div class="chart-card">
            <div class="chart-card-header">
              <h3>Per-Iteration Best / Mean / Median</h3>
              <div class="chart-card-actions">
                <button class="btn btn-icon" title="Info" data-info="verboseIterStats">?</button>
                <button class="btn btn-icon" title="Save" data-export-plot="plotVerboseIter">&#128190;</button>
              </div>
            </div>
            <div id="plotVerboseIter" class="chart-area"></div>
          </div>
        </div>
        <!-- Row 2: tau evolution + sample spread -->
        <div class="chart-grid">
          <div class="chart-card">
            <div class="chart-card-header">
              <h3>Tau (elite floor) &amp; Std per iteration</h3>
              <div class="chart-card-actions">
                <button class="btn btn-icon" title="Info" data-info="verboseTauStd">?</button>
                <button class="btn btn-icon" title="Save" data-export-plot="plotVerboseTau">&#128190;</button>
              </div>
            </div>
            <div id="plotVerboseTau" class="chart-area"></div>
          </div>
          <div class="chart-card">
            <div class="chart-card-header">
              <h3>Samples per iteration &amp; Best-value drop</h3>
              <div class="chart-card-actions">
                <button class="btn btn-icon" title="Info" data-info="verboseSamplesImprovement">?</button>
                <button class="btn btn-icon" title="Save" data-export-plot="plotVerboseSamples">&#128190;</button>
              </div>
            </div>
            <div id="plotVerboseSamples" class="chart-area"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Info Modal -->
  <div class="modal-overlay" id="infoModal">
    <div class="modal-box">
      <div class="modal-content">
        <button class="modal-close" id="modalClose">&times;</button>
        <div class="lang-toggle">
          <button class="lang-btn active" data-lang="en">English</button>
          <button class="lang-btn" data-lang="ru">Русский</button>
        </div>
        <h3 id="modalTitle"></h3>
        <div id="modalBody"></div>
      </div>
    </div>
  </div>

  <script>
    // ===== DATA =====
    const RAW = __DATA_PAYLOAD__;
    const EXPERIMENTS = RAW.experiments;
    const RECORDS = RAW.records;
    const PRESELECTED = new Set(RAW.preselectedExperimentIds || []);
    const TREE = RAW.tree;
    const BBOB_NAMES = __BBOB_NAMES__;

    // BBOB function groups
    const BBOB_GROUPS = {
      'Separable': { funcs: [1, 2, 3, 4, 5],
        en: 'Functions where each variable can be optimised independently. No variable interactions — the global optimum is found by solving D one-dimensional problems. Easy for methods that adapt per-coordinate, hard for methods that rely on covariance (e.g. vanilla CMA-ES with full matrix). Includes: f1 Sphere, f2 Ellipsoidal-separable, f3 Rastrigin-separable, f4 Büche-Rastrigin, f5 Linear slope.',
        ru: 'Функции, где каждая переменная может быть оптимизирована независимо. Нет взаимодействий между переменными — глобальный оптимум находится решением D одномерных задач. Легко для методов с покоординатной адаптацией, сложно для методов с ковариационной матрицей. Включает: f1 Сфера, f2 Эллипсоидальная, f3 Растригина, f4 Бюхе-Растригина, f5 Линейный уклон.' },
      'Low/moderate conditioning': { funcs: [6, 7, 8, 9],
        en: 'Unimodal functions with moderate ill-conditioning (condition number ~10–1000). Variables interact, but the landscape has a single funnel toward the optimum. Tests basic covariance adaptation and step-size control. Includes: f6 Attractive sector, f7 Step-ellipsoidal, f8 Rosenbrock (original), f9 Rosenbrock (rotated).',
        ru: 'Унимодальные функции с умеренной обусловленностью (число обусловленности ~10–1000). Переменные взаимодействуют, но ландшафт имеет одну воронку к оптимуму. Тестирует базовую адаптацию ковариации и контроль размера шага. Включает: f6 Притягивающий сектор, f7 Ступенчато-эллипсоидальная, f8 Розенброк (оригинал), f9 Розенброк (повёрнутая).' },
      'High conditioning & unimodal': { funcs: [10, 11, 12, 13, 14],
        en: 'Unimodal functions with very high ill-conditioning (condition number up to 10⁶). The landscape is still a single valley, but axes are extremely stretched — naive search follows narrow, elongated ridges. Tests whether the optimizer can learn and exploit the full covariance structure efficiently. Includes: f10 Ellipsoidal (high-cond), f11 Discus, f12 Bent cigar, f13 Sharp ridge, f14 Different powers.',
        ru: 'Унимодальные функции с очень высокой плохой обусловленностью (число обусловленности до 10⁶). Ландшафт — одна долина, но оси сильно растянуты — наивный поиск следует по узким хребтам. Тестирует способность оптимизатора обучать и использовать полную ковариационную структуру. Включает: f10 Эллипсоидальная (высокая обусл.), f11 Диск, f12 Изогнутая сигара, f13 Острый хребет, f14 Разные степени.' },
      'Multi-modal (adequate structure)': { funcs: [15, 16, 17, 18, 19],
        en: 'Multi-modal functions that retain enough global structure for well-tuned methods to find the global optimum. Multiple local optima exist, but the landscape has gradients that guide search toward the best basin. Tests the balance between exploration and exploitation. Includes: f15 Rastrigin (rotated), f16 Weierstrass, f17 Schaffers F7, f18 Schaffers F7 (ill-cond), f19 Griewank-Rosenbrock (composite).',
        ru: 'Мультимодальные функции, сохраняющие достаточно глобальной структуры, чтобы настроенные методы нашли глобальный оптимум. Множество локальных оптимумов, но ландшафт имеет градиенты, направляющие поиск к лучшему бассейну. Тестирует баланс между исследованием и эксплуатацией. Включает: f15 Растригина (повёрнутая), f16 Вейерштрасс, f17 Шафферс F7, f18 Шафферс F7 (обусл.), f19 Гриванк-Розенброк.' },
      'Multi-modal (weak structure)': { funcs: [20, 21, 22, 23, 24],
        en: 'Multi-modal functions with very weak or deceptive global structure. Many local optima of similar quality; the global optimum offers little basin-of-attraction advantage. Essentially a needle-in-a-haystack scenario — tests pure exploration capability and robustness to deceptive gradients. Includes: f20 Schwefel, f21 Gallagher 101 peaks, f22 Gallagher 21 peaks, f23 Katsuura, f24 Lunacek bi-Rastrigin.',
        ru: 'Мультимодальные функции с очень слабой или обманчивой глобальной структурой. Множество локальных оптимумов похожего качества; глобальный оптимум не имеет преимущества по бассейну притяжения. «Иголка в стоге сена» — тестирует способность к чистому исследованию и устойчивость к обманчивым градиентам. Включает: f20 Швефеля, f21 Галлагер 101 пик, f22 Галлагер 21 пик, f23 Кацуура, f24 Лунацек би-Растригина.' },
    };

    // Color palettes (chart-only colors, not page background)
    const COLOR_PALETTES = {
      'Default': ['#4361ee','#e63946','#2a9d8f','#f77f00','#6a4c93','#1d3557','#457b9d','#e9c46a','#264653','#ef476f'],
      'Vibrant': ['#ff6b6b','#4ecdc4','#45b7d1','#f7dc6f','#bb8fce','#82e0aa','#f0b27a','#85c1e9','#d7bde2','#f9e79f'],
      'Earth': ['#8b4513','#556b2f','#2f4f4f','#b8860b','#708090','#a0522d','#6b8e23','#4682b4','#cd853f','#696969'],
      'Pastel': ['#a8d8ea','#aa96da','#fcbad3','#ffffd2','#b5ead7','#c7ceea','#ffdac1','#e2f0cb','#ff9aa2','#d4a5a5'],
      'Neon': ['#00ff87','#ff00ff','#00e5ff','#ffea00','#ff3d00','#d500f9','#00e676','#ff9100','#2979ff','#76ff03'],
      'Ocean': ['#0077b6','#00b4d8','#90e0ef','#023e8a','#48cae4','#0096c7','#caf0f8','#03045e','#ade8f4','#0466c8'],
      'Sunset': ['#f94144','#f3722c','#f8961e','#f9844a','#f9c74f','#90be6d','#43aa8b','#4d908e','#577590','#277da1'],
    };
    var currentPalette = 'Default';

    function getPaletteColor(i) { var p = COLOR_PALETTES[currentPalette]; return p[i % p.length]; }

    // Build palette selector UI
    (function buildPaletteUI() {
      var container = document.getElementById('paletteOptions');
      for (var name in COLOR_PALETTES) {
        var swatch = document.createElement('div');
        swatch.className = 'palette-swatch' + (name === currentPalette ? ' active' : '');
        swatch.dataset.palette = name;
        swatch.title = name;
        for (var c = 0; c < 5; c++) {
          var s = document.createElement('span');
          s.style.background = COLOR_PALETTES[name][c];
          swatch.appendChild(s);
        }
        swatch.addEventListener('click', function() {
          currentPalette = this.dataset.palette;
          document.querySelectorAll('.palette-swatch').forEach(function(el) { el.classList.remove('active'); });
          this.classList.add('active');
          if (lastResult) { renderPlots(lastResult); renderTables(lastResult); }
          renderVerbosePlot();
        });
        container.appendChild(swatch);
      }
    })();

    // Build BBOB function group chips
    (function buildFuncGroupChips() {
      var container = document.getElementById('funcGroupChips');
      for (var group in BBOB_GROUPS) {
        var chip = document.createElement('span');
        chip.className = 'func-group-chip';
        var gd = BBOB_GROUPS[group];
        chip.textContent = group + ' (f' + gd.funcs[0] + '-f' + gd.funcs[gd.funcs.length-1] + ')';
        chip.dataset.group = group;
        chip.title = gd.en;
        chip.addEventListener('click', function() {
          this.classList.toggle('active');
          applyFuncGroupFilter();
        });
        container.appendChild(chip);
      }
    })();

    function applyFuncGroupFilter() {
      var active = document.querySelectorAll('.func-group-chip.active');
      if (!active.length) {
        document.getElementById('functionFilter').value = '';
        return;
      }
      var funcs = [];
      active.forEach(function(chip) {
        var g = BBOB_GROUPS[chip.dataset.group];
        if (g) funcs = funcs.concat(g.funcs);
      });
      funcs = [...new Set(funcs)].sort(function(a,b) { return a - b; });
      document.getElementById('functionFilter').value = funcs.join(',');
    }

    const PALETTE = ['#4361ee','#ef4444','#10b981','#f59e0b','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'];
    const PLOTLY_LAYOUT_BASE = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#fafbfe',
      font: { family: 'Inter, system-ui, sans-serif', size: 12, color: '#1a1d2e' },
      margin: { l: 60, r: 30, t: 40, b: 60 },
      xaxis: { gridcolor: '#eef0f6', zerolinecolor: '#dde0ea' },
      yaxis: { gridcolor: '#eef0f6', zerolinecolor: '#dde0ea' },
    };

    let aggregateTable = null, sharedTable = null, functionSummaryTab = null, pairwiseTab = null;
    let lastResult = null;

    // ===== INFO DESCRIPTIONS (EN/RU) =====
    const INFO = {
      winRate: {
        en: { title: 'Win Rate by Experiment', body: '<p>This bar chart shows the percentage of shared triplets (function, dimension, instance) where each experiment achieves the best value of the selected metric.</p><p>A "win" means the experiment had the lowest gap (or evaluations) or highest success rate on a specific triplet. Ties are split equally among winners.</p><p><strong>Interpretation:</strong> Higher bar = better overall performance. If one method dominates, it will show a much higher win rate.</p>' },
        ru: { title: 'Процент побед по эксперименту', body: '<p>Эта столбчатая диаграмма показывает процент общих триплетов (функция, размерность, экземпляр), в которых каждый эксперимент достигает лучшего значения выбранной метрики.</p><p>«Победа» означает, что эксперимент показал наименьший gap (или количество вычислений) или наибольший процент успеха на конкретном триплете. При ничьей победа делится поровну.</p><p><strong>Интерпретация:</strong> Более высокий столбец = лучшая общая производительность.</p>' }
      },
      relativeDistribution: {
        en: { title: 'Relative Metric Distribution', body: '<p>Box plot showing the distribution of log₁₀(metric / best_metric) for each experiment across all shared triplets.</p><p>For each triplet, the best experiment gets ratio = 1 (log₁₀ = 0). Worse experiments get positive values.</p><p><strong>Interpretation:</strong> Boxes closer to 0 indicate the method is consistently near-optimal. Long whiskers or outliers mean occasional poor performance. A narrow box near 0 is ideal.</p>' },
        ru: { title: 'Распределение относительной метрики', body: '<p>Box-plot показывает распределение log₁₀(метрика / лучшая_метрика) для каждого эксперимента по всем общим триплетам.</p><p>Для каждого триплета лучший эксперимент получает отношение = 1 (log₁₀ = 0). Худшие эксперименты получают положительные значения.</p><p><strong>Интерпретация:</strong> Боксы ближе к 0 означают, что метод стабильно близок к оптимуму. Длинные усы или выбросы означают случайные плохие результаты. Узкий бокс около 0 — идеально.</p>' }
      },
      dimensionScaling: {
        en: { title: 'Dimension Scaling', body: '<p>Line chart showing how the mean metric value changes as problem dimensionality increases, for each selected experiment.</p><p>This reveals how well each method scales to higher dimensions — a crucial property for black-box optimizers.</p><p><strong>Interpretation:</strong> Flatter lines indicate better scaling. Steep increases mean the method degrades quickly in high dimensions. Crossing lines reveal dimension-dependent trade-offs between methods.</p>' },
        ru: { title: 'Масштабирование по размерности', body: '<p>Линейная диаграмма показывает, как среднее значение метрики меняется с ростом размерности задачи для каждого выбранного эксперимента.</p><p>Это показывает, насколько хорошо метод масштабируется на большие размерности — ключевое свойство для оптимизаторов чёрного ящика.</p><p><strong>Интерпретация:</strong> Более плоские линии означают лучшее масштабирование. Резкий рост означает быструю деградацию в высоких размерностях. Пересечения линий показывают зависящие от размерности компромиссы.</p>' }
      },
      functionRank: {
        en: { title: 'Function-wise Mean Rank Heatmap', body: '<p>Heatmap where each cell shows the mean rank of an experiment on a specific BBOB function (averaged over all shared dimensions and instances).</p><p>Rank 1 = best on that function. BBOB functions test different problem properties: separability, multi-modality, ill-conditioning, etc.</p><p><strong>Interpretation:</strong> Dark (low rank) cells show strengths. Look for patterns: does a method excel on unimodal functions (f1-f5) but struggle on multi-modal ones (f15-f24)?</p>' },
        ru: { title: 'Тепловая карта рангов по функциям', body: '<p>Тепловая карта, где каждая ячейка показывает средний ранг эксперимента на конкретной функции BBOB (усреднённый по всем общим размерностям и экземплярам).</p><p>Ранг 1 = лучший на данной функции. Функции BBOB тестируют разные свойства: сепарабельность, мультимодальность, плохую обусловленность и т.д.</p><p><strong>Интерпретация:</strong> Тёмные (низкий ранг) ячейки показывают сильные стороны. Ищите закономерности: преуспевает ли метод на унимодальных функциях (f1-f5), но испытывает трудности на мультимодальных (f15-f24)?</p>' }
      },
      ecdf: {
        en: { title: 'ECDF: Empirical Cumulative Distribution Function', body: '<p>Shows the fraction of problems solved (gap below threshold) as a function of the gap threshold, plotted on a log scale.</p><p>This is a standard COCO benchmarking plot. Each curve shows how many triplets achieve a gap smaller than the x-axis value.</p><p><strong>Interpretation:</strong> Curves further to the left and higher are better. A curve that rises quickly means the method achieves small gaps on many problems. The vertical distance between curves at any threshold shows relative advantage.</p>' },
        ru: { title: 'ECDF: Эмпирическая кумулятивная функция распределения', body: '<p>Показывает долю решённых задач (gap ниже порога) как функцию от порога gap, на логарифмической шкале.</p><p>Это стандартный график бенчмаркинга COCO. Каждая кривая показывает, сколько триплетов достигают gap меньше значения на оси x.</p><p><strong>Интерпретация:</strong> Кривые левее и выше — лучше. Быстро растущая кривая означает, что метод достигает малых gap на многих задачах. Вертикальное расстояние между кривыми показывает относительное преимущество.</p>' }
      },
      convergenceRate: {
        en: { title: 'Convergence Rate by Dimension', body: '<p>Grouped bar chart showing the percentage of triplets where each experiment achieved convergence (gap below precision threshold), broken down by dimension.</p><p><strong>Interpretation:</strong> Higher bars mean better convergence. Compare how convergence rates change with dimension — a method that stays high even at large dimensions is robust. Methods that drop off sharply may need more evaluations in high dimensions.</p>' },
        ru: { title: 'Процент сходимости по размерности', body: '<p>Сгруппированная столбчатая диаграмма, показывающая процент триплетов, в которых каждый эксперимент достиг сходимости (gap ниже порога точности), с разбивкой по размерности.</p><p><strong>Интерпретация:</strong> Более высокие столбцы — лучшая сходимость. Сравните, как процент сходимости меняется с размерностью. Метод, который остаётся высоким даже при больших размерностях — робастный.</p>' }
      },
      medianGapByFunction: {
        en: { title: 'Median Gap by Function', body: '<p>Grouped bar chart showing the median gap value for each experiment on each BBOB function (across all shared dimensions and instances), on a log scale.</p><p>This gives a function-by-function comparison that is more robust to outliers than the mean.</p><p><strong>Interpretation:</strong> Lower bars are better. Compare bar heights within each function group to see which method performs best on specific problem types.</p>' },
        ru: { title: 'Медианный gap по функциям', body: '<p>Сгруппированная столбчатая диаграмма, показывающая медианное значение gap для каждого эксперимента на каждой функции BBOB (по всем общим размерностям и экземплярам), на логарифмической шкале.</p><p>Это даёт сравнение по функциям, которое более устойчиво к выбросам, чем среднее.</p><p><strong>Интерпретация:</strong> Более низкие столбцы — лучше. Сравните высоту столбцов внутри каждой группы функций.</p>' }
      },
      pairwiseWins: {
        en: { title: 'Pairwise Head-to-Head Wins', body: '<p>Heatmap showing pairwise win counts. Cell (row, col) shows how many shared triplets where the row experiment beat the column experiment on the selected metric.</p><p><strong>Interpretation:</strong> Read row-by-row. High values in a row mean that experiment frequently beats others. This reveals dominance relationships that aggregate statistics might hide.</p>' },
        ru: { title: 'Попарные победы', body: '<p>Тепловая карта попарных побед. Ячейка (строка, столбец) показывает, на скольких общих триплетах эксперимент в строке победил эксперимент в столбце по выбранной метрике.</p><p><strong>Интерпретация:</strong> Читайте построчно. Высокие значения в строке означают, что эксперимент часто побеждает остальных. Это выявляет отношения доминирования, которые агрегированная статистика может скрыть.</p>' }
      },
      aggregateTable: {
        en: { title: 'Aggregate Ranking Table', body: '<p>Summary table with one row per experiment. Key columns:</p><ul style="padding-left:20px;"><li><strong>Mean/Median Metric:</strong> Average and median values of the selected metric across shared triplets.</li><li><strong>Wins/Win Rate:</strong> Number and percentage of triplets where this experiment was best.</li><li><strong>Mean log₁₀ ratio:</strong> Average log₁₀(metric/best). Zero means always best; higher means worse.</li></ul><p>Sort any column by clicking its header. Use the header filters to search.</p>' },
        ru: { title: 'Агрегированная таблица ранжирования', body: '<p>Сводная таблица с одной строкой на эксперимент. Ключевые столбцы:</p><ul style="padding-left:20px;"><li><strong>Mean/Median Metric:</strong> Среднее и медианное значения выбранной метрики.</li><li><strong>Wins/Win Rate:</strong> Количество и процент триплетов, где эксперимент был лучшим.</li><li><strong>Mean log₁₀ ratio:</strong> Средний log₁₀(метрика/лучшая). Ноль = всегда лучший.</li></ul><p>Сортируйте любой столбец, нажав на заголовок. Используйте фильтры для поиска.</p>' }
      },
      functionSummaryTable: {
        en: { title: 'Per-Function Summary', body: '<p>One row per (experiment, function) combination, showing the mean and median metric, win count, and convergence rate for that function across all shared dimensions and instances.</p><p>Use the header filters to focus on specific functions or experiments. BBOB function names help identify problem types (separable, multimodal, ill-conditioned).</p>' },
        ru: { title: 'Сводка по функциям', body: '<p>Одна строка на комбинацию (эксперимент, функция), показывающая среднее и медианное значение метрики, количество побед и процент сходимости для данной функции.</p><p>Используйте фильтры заголовков для поиска конкретных функций или экспериментов. Названия функций BBOB помогают определить тип задачи (сепарабельная, мультимодальная, плохо обусловленная).</p>' }
      },
      pairwiseTable: {
        en: { title: 'Pairwise Win Matrix', body: '<p>Shows head-to-head comparison between every pair of experiments. For each pair: the number of wins for each side and ties.</p><p><strong>Interpretation:</strong> Useful for understanding dominance relationships. If method A beats method B on 80% of triplets, A is strongly preferred in direct comparison.</p>' },
        ru: { title: 'Матрица попарных побед', body: '<p>Показывает прямое сравнение между каждой парой экспериментов: количество побед каждой стороны и ничьих.</p><p><strong>Интерпретация:</strong> Полезно для понимания отношений доминирования. Если метод A побеждает метод B на 80% триплетов, A строго предпочтительнее.</p>' }
      },
      detailedTable: {
        en: { title: 'Detailed Shared Keys Table', body: '<p>Full per-triplet data. Each row is one (function, dimension, instance) triplet showing the metric value for each experiment and the winner.</p><p>Use column filters to drill down. Sort by any column to find specific results. This is the raw data underlying all other visualizations.</p>' },
        ru: { title: 'Детальная таблица общих ключей', body: '<p>Полные данные по триплетам. Каждая строка — один триплет (функция, размерность, экземпляр), показывающий значение метрики для каждого эксперимента и победителя.</p><p>Используйте фильтры столбцов для поиска. Это исходные данные, лежащие в основе всех визуализаций.</p>' }
      },
      verboseLogs: {
        en: { title: 'Convergence Traces (Verbose Logs)', body: '<p>This section shows detailed optimization traces from experiments that recorded verbose logs (<code>.npz</code> files). Each <code>.npz</code> file stores raw data from one optimizer run on a single (function, dimension, instance) triplet.</p><p>The data comes from the <em>ask-tell</em> loop: at each <strong>iteration</strong> the optimizer generates a batch of candidate solutions (samples), evaluates them via the objective function, and uses the results to update its internal model. The four plots below visualize different aspects of this process.</p><p>Use the dropdown filters above to select an experiment and navigate between traces (one trace = one optimizer run).</p>' },
        ru: { title: 'Трассы сходимости (Verbose Logs)', body: '<p>Этот раздел показывает подробные трассы оптимизации из экспериментов, которые записали verbose-логи (<code>.npz</code> файлы). Каждый <code>.npz</code> файл хранит необработанные данные одного запуска оптимизатора на одном триплете (функция, размерность, экземпляр).</p><p>Данные берутся из цикла <em>ask-tell</em>: на каждой <strong>итерации</strong> оптимизатор генерирует пакет кандидатов (samples), оценивает их целевой функцией и использует результаты для обновления модели. Четыре графика ниже визуализируют разные аспекты этого процесса.</p><p>Используйте фильтры выше для выбора эксперимента и навигации между трассами (одна трасса = один запуск оптимизатора).</p>' }
      },
      verboseConvergence: {
        en: { title: 'Convergence (all evaluations)', body: '<p>Shows <strong>every individual function evaluation</strong> performed during the optimizer run.</p><h4>Axes</h4><ul style="padding-left:18px;"><li><strong>X-axis — Cumulative evaluation:</strong> sequential number of the function evaluation (1, 2, 3, …). All samples from all iterations are laid out in chronological order.</li><li><strong>Y-axis — f(x):</strong> the objective function value returned for that candidate solution.</li></ul><h4>Traces</h4><ul style="padding-left:18px;"><li><strong>f(x) (scatter dots):</strong> each dot is one evaluation — the raw value returned by the objective function for one candidate. Spread shows how diverse each batch is.</li><li><strong>running best (line):</strong> the <em>cumulative minimum</em>: at evaluation <em>k</em>, it equals <code>min(f(x₁), f(x₂), …, f(xₖ))</code>. A flat running-best line means the optimizer has not found any improvement in that segment.</li></ul><h4>What to look for</h4><p>A healthy optimizer shows the scatter dots trending downward with the running-best line descending in clear steps. If the running best flattens early while dots remain spread out, the optimizer may be stuck in a local optimum.</p>' },
        ru: { title: 'Сходимость (все вычисления)', body: '<p>Показывает <strong>каждое отдельное вычисление целевой функции</strong>, выполненное за запуск оптимизатора.</p><h4>Оси</h4><ul style="padding-left:18px;"><li><strong>X — Кумулятивное вычисление:</strong> порядковый номер вычисления функции (1, 2, 3, …). Все сэмплы всех итераций расположены хронологически.</li><li><strong>Y — f(x):</strong> значение целевой функции для данного кандидата.</li></ul><h4>Трассы</h4><ul style="padding-left:18px;"><li><strong>f(x) (точки):</strong> каждая точка — одно вычисление. Разброс показывает разнообразие кандидатов.</li><li><strong>running best (линия):</strong> <em>кумулятивный минимум</em>: на вычислении <em>k</em> значение равно <code>min(f(x₁), …, f(xₖ))</code>. Горизонтальный участок означает отсутствие улучшения.</li></ul><h4>На что обратить внимание</h4><p>Здоровый оптимизатор показывает тренд точек вниз, а running-best снижается ступенями. Если running-best выходит на плато рано, а точки всё ещё разбросаны — оптимизатор застрял в локальном оптимуме.</p>' }
      },
      verboseIterStats: {
        en: { title: 'Per-Iteration Statistics', body: '<p>Aggregates objective values <strong>within each iteration</strong> (one ask-tell cycle = one batch of candidates).</p><h4>Axes</h4><ul style="padding-left:18px;"><li><strong>X-axis — Iteration:</strong> the iteration index (0, 1, 2, …). Each iteration corresponds to one batch of samples generated by the optimizer.</li><li><strong>Y-axis — f(x):</strong> aggregated objective value for that iteration.</li></ul><h4>Traces (all computed from the batch of samples at iteration <em>i</em>)</h4><ul style="padding-left:18px;"><li><strong>best:</strong> <code>min(f(x))</code> across all samples in iteration <em>i</em>. Shows the quality of the best candidate the optimizer produced in this batch.</li><li><strong>mean:</strong> <code>Σf(x) / n</code> — arithmetic mean of all sample values in iteration <em>i</em>. Reflects the average quality of the whole batch.</li><li><strong>median:</strong> the middle value when sorting all sample values in iteration <em>i</em>. Less sensitive to outliers than the mean.</li></ul><h4>What to look for</h4><p>If <em>best</em> decreases steadily, the optimizer finds better solutions each iteration. A large gap between <em>mean</em> and <em>best</em> means most samples are far from optimal — the model is still exploring. When all three lines converge to the same value, the model has collapsed to a narrow region.</p>' },
        ru: { title: 'Статистика по итерациям', body: '<p>Агрегирует значения целевой функции <strong>внутри каждой итерации</strong> (один цикл ask-tell = один пакет кандидатов).</p><h4>Оси</h4><ul style="padding-left:18px;"><li><strong>X — Итерация:</strong> индекс итерации (0, 1, 2, …). Каждая итерация — один пакет сэмплов от оптимизатора.</li><li><strong>Y — f(x):</strong> агрегированное значение целевой функции.</li></ul><h4>Трассы (вычисляются из пакета сэмплов итерации <em>i</em>)</h4><ul style="padding-left:18px;"><li><strong>best:</strong> <code>min(f(x))</code> среди всех сэмплов итерации <em>i</em>. Качество лучшего кандидата в пакете.</li><li><strong>mean:</strong> <code>Σf(x) / n</code> — среднее арифметическое всех значений в итерации <em>i</em>. Средний уровень качества пакета.</li><li><strong>median:</strong> серединное значение при сортировке всех значений итерации <em>i</em>. Менее чувствительно к выбросам, чем среднее.</li></ul><h4>На что обратить внимание</h4><p>Если <em>best</em> стабильно снижается — оптимизатор находит лучшие решения каждую итерацию. Большой разрыв между <em>mean</em> и <em>best</em> означает, что большинство сэмплов далеки от оптимума. Когда все три линии сходятся — модель коллапсировала в узкую область.</p>' }
      },
      verboseTauStd: {
        en: { title: 'Tau & Spread per Iteration', body: '<p>Two metrics that reveal how the diffusion model\'s training signal and sample diversity evolve.</p><h4>Axes</h4><ul style="padding-left:18px;"><li><strong>X-axis — Iteration:</strong> the iteration index.</li><li><strong>Left Y-axis — tau (elite floor):</strong> the threshold used to select "elite" samples for training. In quantile-based diffusion optimizers, tau is the <em>worst objective value among elite samples</em>. Only samples with <code>f(x) ≤ tau</code> are used as positive training data for the diffusion model.</li><li><strong>Right Y-axis — std(f(x)):</strong> the <em>population standard deviation</em> of all objective values in the batch: <code>sqrt(Σ(f(xⱼ) − mean)² / n)</code>. Measures how spread out the batch is.</li></ul><h4>What to look for</h4><ul style="padding-left:18px;"><li><strong>tau decreasing:</strong> the elite quality threshold is tightening — the model is learning to produce better solutions.</li><li><strong>tau flat:</strong> the elite pool is not improving. The model may need more iterations or the problem is near convergence.</li><li><strong>std decreasing:</strong> samples are becoming more concentrated. Moderate decrease is good (model focuses on promising regions). If std drops to near-zero, the model has <em>collapsed</em> — it only generates nearly identical candidates and cannot explore.</li><li><strong>std increasing:</strong> the model is diversifying, possibly because of noise or a reset mechanism.</li></ul>' },
        ru: { title: 'Tau и разброс по итерациям', body: '<p>Две метрики, показывающие эволюцию обучающего сигнала диффузионной модели и разнообразия сэмплов.</p><h4>Оси</h4><ul style="padding-left:18px;"><li><strong>X — Итерация:</strong> индекс итерации.</li><li><strong>Левая Y — tau (порог элиты):</strong> порог для отбора «элитных» сэмплов для обучения. В квантильных диффузионных оптимизаторах tau — это <em>худшее значение целевой функции среди элитных сэмплов</em>. Только сэмплы с <code>f(x) ≤ tau</code> используются как положительные обучающие данные для диффузионной модели.</li><li><strong>Правая Y — std(f(x)):</strong> <em>стандартное отклонение</em> всех значений в пакете: <code>sqrt(Σ(f(xⱼ) − mean)² / n)</code>. Измеряет разброс сэмплов.</li></ul><h4>На что обратить внимание</h4><ul style="padding-left:18px;"><li><strong>tau снижается:</strong> порог элиты ужесточается — модель учится генерировать лучшие решения.</li><li><strong>tau плоский:</strong> пул элиты не улучшается.</li><li><strong>std снижается:</strong> сэмплы концентрируются. Умеренное снижение — хорошо. Если std падает к нулю — модель <em>коллапсировала</em> и не может исследовать пространство.</li><li><strong>std растёт:</strong> модель диверсифицируется, возможно из-за шума или механизма перезапуска.</li></ul>' }
      },
      verboseSamplesImprovement: {
        en: { title: 'Samples per Iteration & Best-Value Drop', body: '<p>Combines the batch size with the <em>improvement signal</em> to show whether adding more evaluations actually helps.</p><h4>Axes</h4><ul style="padding-left:18px;"><li><strong>X-axis — Iteration:</strong> the iteration index.</li><li><strong>Left Y-axis — samples (bars):</strong> the number of candidate solutions generated and evaluated in this iteration (batch size). May vary between iterations if the optimizer uses an adaptive batch scheme.</li><li><strong>Right Y-axis — best-value drop (line):</strong> calculated as <code>iter_best[i−1] − iter_best[i]</code>, i.e. how much the per-iteration best improved compared to the previous iteration. Positive values mean the optimizer found a <em>better</em> solution than before; zero means no progress; negative values (rare) mean the best in this batch was worse than the previous batch\'s best (note: the <em>running</em> best still cannot increase).</li></ul><h4>What to look for</h4><ul style="padding-left:18px;"><li><strong>Large bars + positive drop:</strong> the optimizer is using many samples productively — good exploration.</li><li><strong>Large bars + zero drop:</strong> spending evaluations without progress — budget is being wasted.</li><li><strong>Drop spikes:</strong> iterations where a significant breakthrough happened. Useful for identifying when the optimizer discovered a new promising basin.</li><li><strong>Drop near zero everywhere:</strong> the optimizer converged (or stalled) very early.</li></ul>' },
        ru: { title: 'Сэмплы за итерацию и падение лучшего значения', body: '<p>Объединяет размер пакета с <em>сигналом улучшения</em>, показывая, приносят ли дополнительные вычисления пользу.</p><h4>Оси</h4><ul style="padding-left:18px;"><li><strong>X — Итерация:</strong> индекс итерации.</li><li><strong>Левая Y — samples (столбцы):</strong> число сгенерированных и оценённых кандидатов в этой итерации (размер пакета). Может меняться между итерациями при адаптивной схеме.</li><li><strong>Правая Y — падение лучшего значения (линия):</strong> вычисляется как <code>iter_best[i−1] − iter_best[i]</code>, т.е. насколько лучший результат итерации улучшился по сравнению с предыдущей. Положительные значения — оптимизатор нашёл <em>лучшее</em> решение; ноль — прогресса нет.</li></ul><h4>На что обратить внимание</h4><ul style="padding-left:18px;"><li><strong>Большие столбцы + положительное падение:</strong> оптимизатор продуктивно использует сэмплы.</li><li><strong>Большие столбцы + нулевое падение:</strong> бюджет тратится без прогресса.</li><li><strong>Пики падения:</strong> итерации с прорывом — оптимизатор обнаружил новый бассейн.</li><li><strong>Падение около нуля везде:</strong> оптимизатор сошёлся (или застрял) очень рано.</li></ul>' }
      },
      bbobGroups: {
        en: { title: 'BBOB Function Groups', body: '<p>The 24 BBOB (Black-Box Optimization Benchmarking) functions are organised into 5 groups by their <strong>landscape properties</strong>. Understanding these groups helps interpret why certain optimizers excel or struggle on specific functions.</p><h4>1. Separable (f1–f5)</h4><p>Each variable can be optimised independently — there are no interactions between dimensions. The global optimum is found by solving <em>D</em> independent 1-D problems. Easy for coordinate-wise methods; functions include Sphere, Ellipsoidal, Rastrigin (separable), Büche-Rastrigin, Linear slope.</p><h4>2. Low/moderate conditioning (f6–f9)</h4><p>Unimodal with moderate ill-conditioning (condition number ~10–1000). Variables interact, but there is a single funnel toward the optimum. Tests basic covariance adaptation and step-size control. Functions: Attractive sector, Step-ellipsoidal, Rosenbrock (original & rotated).</p><h4>3. High conditioning & unimodal (f10–f14)</h4><p>Unimodal with <strong>very high</strong> ill-conditioning (condition number up to 10⁶). The landscape is a single extremely stretched valley — naive search gets trapped on narrow ridges. Tests efficient learning of the full covariance structure. Functions: Ellipsoidal (high-cond), Discus, Bent cigar, Sharp ridge, Different powers.</p><h4>4. Multi-modal — adequate structure (f15–f19)</h4><p>Multiple local optima, but the landscape retains gradients that guide well-tuned search toward the best basin. Tests the exploration/exploitation balance. Functions: Rastrigin (rotated), Weierstrass, Schaffers F7, Schaffers F7 (ill-cond), Griewank-Rosenbrock.</p><h4>5. Multi-modal — weak structure (f20–f24)</h4><p>Many local optima of <em>similar quality</em>; the global optimum has little basin-of-attraction advantage — essentially needle-in-a-haystack. Tests pure exploration capability and robustness to deceptive gradients. Functions: Schwefel, Gallagher 101 peaks, Gallagher 21 peaks, Katsuura, Lunacek bi-Rastrigin.</p>' },
        ru: { title: 'Группы функций BBOB', body: '<p>24 функции BBOB (Black-Box Optimization Benchmarking) организованы в 5 групп по <strong>свойствам ландшафта</strong>. Понимание этих групп помогает интерпретировать, почему определённые оптимизаторы преуспевают или терпят неудачу на конкретных функциях.</p><h4>1. Сепарабельные (f1–f5)</h4><p>Каждая переменная оптимизируется независимо — нет взаимодействий. Глобальный оптимум находится решением <em>D</em> одномерных задач. Включает: Сфера, Эллипсоидальная, Растригина (сепарабельная), Бюхе-Растригина, Линейный уклон.</p><h4>2. Низкая/умеренная обусловленность (f6–f9)</h4><p>Унимодальные с умеренной плохой обусловленностью (число обусловленности ~10–1000). Переменные взаимодействуют, но ландшафт — одна воронка. Тестирует адаптацию ковариации. Включает: Притягивающий сектор, Ступенчатая, Розенброк.</p><h4>3. Высокая обусловленность, унимодальные (f10–f14)</h4><p>Унимодальные с <strong>очень высокой</strong> плохой обусловленностью (до 10⁶). Ландшафт — сильно растянутая долина. Тестирует обучение полной ковариационной структуры. Включает: Эллипсоидальная, Диск, Изогнутая сигара, Острый хребет, Разные степени.</p><h4>4. Мультимодальные — адекватная структура (f15–f19)</h4><p>Множество локальных оптимумов, но ландшафт сохраняет градиенты к лучшему бассейну. Тестирует баланс исследования и эксплуатации. Включает: Растригина (повёрнутая), Вейерштрасс, Шафферс F7, Гриванк-Розенброк.</p><h4>5. Мультимодальные — слабая структура (f20–f24)</h4><p>Множество оптимумов похожего качества. Глобальный оптимум не имеет преимущества. «Иголка в стоге сена». Включает: Швефеля, Галлагер 101/21 пик, Кацуура, Лунацек.</p>' }
      },
    };

    // ===== TREE RENDERING =====
    function renderTree(node, container, depth) {
      const sortedKeys = Object.keys(node).sort();
      for (const key of sortedKeys) {
        if (key === '__exp' || key === '__children') continue;
        const child = node[key];
        const hasExp = child.__exp;
        const hasChildren = child.__children && Object.keys(child.__children).length > 0;
        const isFolder = !hasExp && hasChildren;
        const isLeafWithChildren = hasExp && hasChildren;
        const isPureLeaf = hasExp && !hasChildren;

        if (isFolder) {
          const folderDiv = document.createElement('div');
          folderDiv.className = 'tree-node';
          const folderHeader = document.createElement('div');
          folderHeader.className = 'tree-folder';
          folderHeader.innerHTML = '<span class="arrow">&#9660;</span><span class="folder-icon">&#128193;</span> ' + key;
          folderDiv.appendChild(folderHeader);
          const childrenDiv = document.createElement('div');
          childrenDiv.className = 'tree-children';
          renderTree(child.__children, childrenDiv, depth + 1);
          folderDiv.appendChild(childrenDiv);
          folderHeader.addEventListener('click', () => {
            childrenDiv.classList.toggle('hidden');
            folderHeader.querySelector('.arrow').classList.toggle('collapsed');
          });
          container.appendChild(folderDiv);
        } else if (isPureLeaf || isLeafWithChildren) {
          if (isLeafWithChildren) {
            const folderDiv = document.createElement('div');
            folderDiv.className = 'tree-node';
            const folderHeader = document.createElement('div');
            folderHeader.className = 'tree-folder';
            folderHeader.innerHTML = '<span class="arrow">&#9660;</span><span class="folder-icon">&#128193;</span> ' + key;
            folderDiv.appendChild(folderHeader);
            const innerDiv = document.createElement('div');
            innerDiv.className = 'tree-children';
            addLeaf(innerDiv, child.__exp);
            renderTree(child.__children, innerDiv, depth + 1);
            folderDiv.appendChild(innerDiv);
            folderHeader.addEventListener('click', () => {
              innerDiv.classList.toggle('hidden');
              folderHeader.querySelector('.arrow').classList.toggle('collapsed');
            });
            container.appendChild(folderDiv);
          } else {
            addLeaf(container, child.__exp);
          }
        } else {
          const nonLeafChildren = Object.keys(child).filter(k => k !== '__exp' && k !== '__children');
          if (nonLeafChildren.length > 0) {
            const folderDiv = document.createElement('div');
            folderDiv.className = 'tree-node';
            const folderHeader = document.createElement('div');
            folderHeader.className = 'tree-folder';
            folderHeader.innerHTML = '<span class="arrow">&#9660;</span><span class="folder-icon">&#128193;</span> ' + key;
            folderDiv.appendChild(folderHeader);
            const childrenDiv = document.createElement('div');
            childrenDiv.className = 'tree-children';
            renderTree(child, childrenDiv, depth + 1);
            folderDiv.appendChild(childrenDiv);
            folderHeader.addEventListener('click', () => {
              childrenDiv.classList.toggle('hidden');
              folderHeader.querySelector('.arrow').classList.toggle('collapsed');
            });
            container.appendChild(folderDiv);
          }
        }
      }
    }

    function addLeaf(container, exp) {
      const leaf = document.createElement('label');
      leaf.className = 'tree-leaf';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'exp-checkbox';
      cb.value = exp.id;
      cb.checked = PRESELECTED.has(exp.id);
      cb.addEventListener('change', () => {
        if (cb.checked) PRESELECTED.add(exp.id);
        else PRESELECTED.delete(exp.id);
      });
      const info = document.createElement('div');
      info.innerHTML = '<span class="leaf-name">' + escapeHtml(exp.display_name || exp.name) + '</span> ' +
        '<span class="leaf-badge">' + exp.source + '</span> ' +
        '<span class="leaf-meta">' + exp.total_records + ' records</span>';
      leaf.appendChild(cb);
      leaf.appendChild(info);
      container.appendChild(leaf);
    }

    function escapeHtml(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    function buildTreeUI() {
      const container = document.getElementById('experimentTree');
      container.innerHTML = '';
      renderTree(TREE, container, 0);
    }

    function filterTree() {
      const q = document.getElementById('searchExperiments').value.toLowerCase().trim();
      const container = document.getElementById('experimentTree');
      if (!q) { buildTreeUI(); return; }
      container.innerHTML = '';
      const matching = EXPERIMENTS.filter(e => {
        const text = (e.display_name + ' ' + e.name + ' ' + e.path).toLowerCase();
        return text.includes(q);
      });
      for (const exp of matching) {
        addLeaf(container, exp);
      }
    }

    // ===== UTILITIES =====
    function parseNumberSet(input) {
      const value = (input || '').trim();
      if (!value) return null;
      const out = new Set();
      for (const chunk of value.split(',').map(s => s.trim()).filter(Boolean)) {
        if (chunk.includes('-')) {
          const [aRaw, bRaw] = chunk.split('-').map(s => s.trim());
          const a = Number(aRaw), b = Number(bRaw);
          if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
          for (let x = Math.min(a,b); x <= Math.max(a,b); x++) out.add(x);
        } else {
          const n = Number(chunk);
          if (Number.isFinite(n)) out.add(n);
        }
      }
      return out.size ? out : null;
    }

    function metricValue(rec, metric) {
      if (metric === 'gap') return rec.gap;
      if (metric === 'evaluations') return rec.evaluations;
      if (metric === 'success') return (rec.final_target_hit || rec.converged) ? 1 : 0;
      return null;
    }

    function direction(metric) { return metric === 'success' ? 'higher' : 'lower'; }
    function keyOf(rec) { return rec.function_id + '|' + rec.dimension + '|' + rec.instance_id; }
    function expName(id) { return (EXPERIMENTS.find(e => e.id === id) || {}).display_name || id; }
    function median(arr) {
      if (!arr.length) return null;
      const s = [...arr].sort((a,b) => a - b);
      const m = Math.floor(s.length / 2);
      return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
    }

    function collectSelectedIds() {
      return Array.from(PRESELECTED).filter(id => EXPERIMENTS.some(e => e.id === id));
    }

    // ===== COMPUTATION =====
    function computeComparison() {
      const metric = document.getElementById('metricSelect').value;
      const fnFilter = parseNumberSet(document.getElementById('functionFilter').value);
      const dimFilter = parseNumberSet(document.getElementById('dimensionFilter').value);
      const insFilter = parseNumberSet(document.getElementById('instanceFilter').value);
      const selected = collectSelectedIds();
      document.getElementById('kpiSelected').textContent = selected.length;

      if (selected.length < 2) return { error: 'Select at least 2 experiments for comparison.' };

      const byExp = new Map(selected.map(id => [id, []]));
      for (const rec of RECORDS) {
        if (!byExp.has(rec.experiment_id)) continue;
        if (fnFilter && !fnFilter.has(rec.function_id)) continue;
        if (dimFilter && !dimFilter.has(rec.dimension)) continue;
        if (insFilter && !insFilter.has(rec.instance_id)) continue;
        byExp.get(rec.experiment_id).push(rec);
      }

      const keySets = new Map();
      for (const [id, rows] of byExp.entries()) keySets.set(id, new Set(rows.map(keyOf)));

      const ids = Array.from(byExp.keys());
      let shared = null;
      for (const id of ids) {
        const keys = keySets.get(id);
        shared = shared === null ? new Set(keys) : new Set([...shared].filter(k => keys.has(k)));
      }
      shared = shared || new Set();

      if (!shared.size) return { error: 'No shared (function, dimension, instance) keys for selected filters.' };

      const expKeyRec = new Map();
      for (const [id, rows] of byExp.entries()) {
        const km = new Map();
        for (const rec of rows) { const key = keyOf(rec); if (shared.has(key)) km.set(key, rec); }
        expKeyRec.set(id, km);
      }

      const sharedRows = [], functionSet = new Set(), dimSet = new Set(), insSet = new Set();
      const wins = Object.fromEntries(ids.map(id => [id, 0]));
      const ties = Object.fromEntries(ids.map(id => [id, 0]));
      const relByExp = Object.fromEntries(ids.map(id => [id, []]));
      const metricByDim = new Map(), rankByFunction = new Map();
      const gapsByExpFunc = {};
      const pairwiseWinsMatrix = {};
      for (const a of ids) { pairwiseWinsMatrix[a] = {}; for (const b of ids) pairwiseWinsMatrix[a][b] = 0; }

      const convergenceByDimExp = {};

      for (const key of shared) {
        const [f, d, i] = key.split('|').map(Number);
        functionSet.add(f); dimSet.add(d); insSet.add(i);

        const values = [];
        for (const id of ids) {
          const rec = expKeyRec.get(id).get(key);
          const val = metricValue(rec, metric);
          if (val === null || !Number.isFinite(val)) continue;
          values.push({ id, val, rec });
        }
        if (values.length !== ids.length) continue;

        let bestVal = values[0].val;
        for (const item of values) {
          if (direction(metric) === 'lower') bestVal = Math.min(bestVal, item.val);
          else bestVal = Math.max(bestVal, item.val);
        }

        let winners = values.filter(v => v.val === bestVal).map(v => v.id);
        if (!winners.length) winners = [values[0].id];
        for (const wid of winners) { wins[wid] += 1/winners.length; if (winners.length > 1) ties[wid]++; }

        // Pairwise
        for (let a = 0; a < values.length; a++) {
          for (let b = 0; b < values.length; b++) {
            if (a === b) continue;
            const better = direction(metric) === 'lower' ? values[a].val < values[b].val : values[a].val > values[b].val;
            if (better) pairwiseWinsMatrix[values[a].id][values[b].id]++;
          }
        }

        for (const item of values) {
          const ratio = direction(metric) === 'lower'
            ? item.val / (bestVal || 1e-15)
            : (bestVal || 1e-15) / (item.val || 1e-15);
          relByExp[item.id].push(Math.log10(Math.max(ratio, 1e-15)));

          const gKey = item.id + '|' + f;
          if (!gapsByExpFunc[gKey]) gapsByExpFunc[gKey] = [];
          gapsByExpFunc[gKey].push(item.val);

          const cKey = item.id + '|' + d;
          if (!convergenceByDimExp[cKey]) convergenceByDimExp[cKey] = { total: 0, converged: 0 };
          convergenceByDimExp[cKey].total++;
          if (item.rec.converged || item.rec.final_target_hit) convergenceByDimExp[cKey].converged++;
        }

        if (!metricByDim.has(d)) metricByDim.set(d, []);
        metricByDim.get(d).push(values);

        const sorted = [...values].sort((a, b) => direction(metric) === 'lower' ? a.val - b.val : b.val - a.val);
        let pos = 1;
        const ranks = new Map();
        for (const item of sorted) { if (!ranks.has(item.id)) ranks.set(item.id, pos); pos++; }
        if (!rankByFunction.has(f)) rankByFunction.set(f, new Map());
        for (const id of ids) {
          if (!rankByFunction.get(f).has(id)) rankByFunction.get(f).set(id, []);
          rankByFunction.get(f).get(id).push(ranks.get(id));
        }

        const detail = { function_id: f, dimension: d, instance_id: i,
          function_name: BBOB_NAMES[f] || ('f' + f),
          winner: winners.map(w => expName(w)).join(', ') };
        for (const item of values) detail['metric__' + expName(item.id)] = item.val;
        sharedRows.push(detail);
      }

      const validKeyCount = sharedRows.length;
      if (!validKeyCount) return { error: 'Shared keys exist, but selected metric is missing on all shared keys.' };

      const aggregateRows = ids.map(id => {
        const exp = EXPERIMENTS.find(e => e.id === id);
        const rawVals = [];
        for (const row of sharedRows) {
          const col = 'metric__' + (exp.display_name || exp.name);
          const v = row[col];
          if (Number.isFinite(v)) rawVals.push(v);
        }
        rawVals.sort((a,b) => a - b);
        const mean = rawVals.length ? rawVals.reduce((a,b) => a+b, 0) / rawVals.length : null;
        const med = rawVals.length ? median(rawVals) : null;
        const relVals = relByExp[id] || [];
        return {
          experiment: exp.display_name || exp.name,
          path: exp.path,
          compared_keys: validKeyCount,
          mean_metric: mean !== null ? +mean.toFixed(6) : null,
          median_metric: med !== null ? +med.toFixed(6) : null,
          wins: +wins[id].toFixed(1),
          win_rate_pct: +(wins[id] * 100 / validKeyCount).toFixed(2),
          ties: ties[id],
          mean_log10_ratio: relVals.length ? +(relVals.reduce((a,b) => a+b, 0) / relVals.length).toFixed(6) : null,
        };
      });

      // function summary rows
      const functionSummaryRows = [];
      for (const id of ids) {
        for (const f of [...functionSet].sort((a,b) => a - b)) {
          const gKey = id + '|' + f;
          const vals = gapsByExpFunc[gKey] || [];
          if (!vals.length) continue;
          const mean = vals.reduce((a,b) => a+b, 0) / vals.length;
          const med = median(vals);
          const fnRanks = rankByFunction.get(f)?.get(id) || [];
          const winsOnF = fnRanks.filter(r => r === 1).length;
          functionSummaryRows.push({
            experiment: expName(id),
            function_id: f,
            function_name: BBOB_NAMES[f] || ('f' + f),
            mean_metric: +mean.toFixed(6),
            median_metric: med !== null ? +med.toFixed(6) : null,
            count: vals.length,
            wins: winsOnF,
            mean_rank: fnRanks.length ? +(fnRanks.reduce((a,b) => a+b, 0) / fnRanks.length).toFixed(2) : null,
          });
        }
      }

      // pairwise table rows
      const pairwiseRows = [];
      for (let a = 0; a < ids.length; a++) {
        for (let b = a+1; b < ids.length; b++) {
          const wA = pairwiseWinsMatrix[ids[a]][ids[b]];
          const wB = pairwiseWinsMatrix[ids[b]][ids[a]];
          const t = validKeyCount - wA - wB;
          pairwiseRows.push({
            experiment_a: expName(ids[a]),
            experiment_b: expName(ids[b]),
            wins_a: wA,
            wins_b: wB,
            ties: t,
            a_rate: +(wA * 100 / validKeyCount).toFixed(1),
            b_rate: +(wB * 100 / validKeyCount).toFixed(1),
          });
        }
      }

      return {
        metric, selected: ids, aggregateRows, sharedRows, functionSummaryRows, pairwiseRows,
        validKeyCount, functionIds: [...functionSet].sort((a,b) => a-b),
        dimensions: [...dimSet].sort((a,b) => a-b), instances: [...insSet].sort((a,b) => a-b),
        wins, relByExp, metricByDim, rankByFunction, gapsByExpFunc, pairwiseWinsMatrix,
        convergenceByDimExp,
      };
    }

    // ===== RENDERING =====
    function setStatus(text, isError) {
      const el = document.getElementById('statusMessage');
      el.textContent = text || '';
      el.className = 'status-msg' + (isError ? ' error' : '');
    }

    function updateKpis(r) {
      document.getElementById('kpiShared').textContent = r?.validKeyCount || 0;
      document.getElementById('kpiFunctions').textContent = r?.functionIds?.length || 0;
      document.getElementById('kpiDims').textContent = r?.dimensions?.length || 0;
      document.getElementById('kpiInstances').textContent = r?.instances?.length || 0;
    }

    function color(i) { return getPaletteColor(i); }
    function R(v, d) { if (v === null || v === undefined || !Number.isFinite(v)) return null; const p = Math.pow(10, d||4); return Math.round(v * p) / p; }
    function Rfmt(v) { if (v === null || v === undefined) return ''; const a = Math.abs(v); if (a === 0) return '0'; if (a >= 1000) return v.toPrecision(4); if (a >= 1) return v.toFixed(2); if (a >= 0.01) return v.toFixed(4); return v.toExponential(2); }

    function populateChartFilters(r) {
      const funcOpts = '<option value="all">All functions</option>' +
        r.functionIds.map(f => '<option value="'+f+'">f'+f+' '+escapeHtml((BBOB_NAMES[f]||'').substring(0,20))+'</option>').join('');
      const dimOpts = '<option value="all">All dimensions</option>' +
        r.dimensions.map(d => '<option value="'+d+'">d='+d+'</option>').join('');
      ['dimScalingFuncFilter','ecdfFuncFilter','convFuncFilter'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { const old = el.value; el.innerHTML = funcOpts; if ([...el.options].some(o=>o.value===old)) el.value = old; }
      });
      ['funcRankDimFilter','ecdfDimFilter','medianGapDimFilter'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { const old = el.value; el.innerHTML = dimOpts; if ([...el.options].some(o=>o.value===old)) el.value = old; }
      });
    }

    function renderPlots(r) {
      const metric = r.metric, ids = r.selected;

      // 1) Win Rate
      const expNames = ids.map(expName);
      const winY = ids.map(id => R(r.wins[id] * 100 / r.validKeyCount, 1));
      Plotly.newPlot('plotWins', [{
        type: 'bar', x: expNames, y: winY,
        marker: { color: ids.map((_, i) => color(i)), line: { width: 0 } },
        text: winY.map(v => v + '%'), textposition: 'outside',
      }], { ...PLOTLY_LAYOUT_BASE, title: { text: 'Win rate (%) on ' + r.validKeyCount + ' shared triplets', font: { size: 14 } },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Win rate (%)', tickformat: '.1f', rangemode: 'tozero' },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, categoryorder: 'array', categoryarray: expNames },
        margin: { l: 50, r: 20, t: 50, b: 80 }, bargap: 0.3,
      }, { responsive: true });

      // 2) Relative distribution
      Plotly.newPlot('plotRelative', ids.map((id, i) => ({
        type: 'box', name: expName(id), y: r.relByExp[id].map(v => R(v,4)), boxpoints: 'outliers',
        marker: { color: color(i), opacity: 0.7 }, line: { color: color(i) },
      })), { ...PLOTLY_LAYOUT_BASE, title: { text: 'Relative ' + metric + ' distribution (log\u2081\u2080 scale)', font: { size: 14 } },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'log\u2081\u2080(ratio to best)', tickformat: '.2f' },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, categoryorder: 'array', categoryarray: expNames },
        margin: { l: 60, r: 20, t: 50, b: 80 }, showlegend: false,
      }, { responsive: true });

      renderDimensionScaling(r);
      renderFunctionRankHeatmap(r);
      renderECDF(r);
      renderConvergenceRate(r);
      renderMedianGap(r);

      // 8) Pairwise heatmap
      if (ids.length >= 2) {
        const pwNames = ids.map(expName);
        const pwZ = ids.map(a => ids.map(b => a === b ? null : r.pairwiseWinsMatrix[a][b]));
        Plotly.newPlot('plotPairwise', [{
          type: 'heatmap', x: pwNames, y: pwNames, z: pwZ,
          colorscale: [[0,'#f5f7fb'],[1,color(0)]],
          colorbar: { title: 'Wins', titleside: 'right', thickness: 12 },
          hovertemplate: '%{y} beats %{x}: %{z} times<extra></extra>',
        }], { ...PLOTLY_LAYOUT_BASE, title: { text: 'Pairwise wins (row beats column)', font: { size: 14 } },
          margin: { l: 140, r: 80, t: 50, b: 100 },
          xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, tickangle: -45, categoryorder: 'array', categoryarray: pwNames },
          yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, categoryorder: 'array', categoryarray: pwNames },
        }, { responsive: true });
      }
    }

    // ===== CHART WITH LOCAL FILTERS =====

    function renderDimensionScaling(r) {
      const metric = r.metric, ids = r.selected;
      const funcVal = document.getElementById('dimScalingFuncFilter').value;
      const filterFunc = funcVal !== 'all' ? Number(funcVal) : null;
      const dims = [...r.metricByDim.keys()].sort((a,b) => a-b);

      Plotly.newPlot('plotDimension', ids.map((id, i) => ({
        type: 'scatter', mode: 'lines+markers', name: expName(id),
        x: dims, y: dims.map(d => {
          const vals = [];
          for (const vl of (r.metricByDim.get(d)||[])) {
            if (filterFunc !== null) {
              const sample = vl[0]?.rec;
              if (sample && sample.function_id !== filterFunc) continue;
            }
            const it = vl.find(v=>v.id===id);
            if (it && Number.isFinite(it.val)) vals.push(it.val);
          }
          return vals.length ? R(vals.reduce((a,b)=>a+b,0)/vals.length, 4) : null;
        }),
        line: { color: color(i), width: 2 }, marker: { size: 6, color: color(i) },
        hovertemplate: '%{x}<br>%{y}<extra>'+expName(id)+'</extra>',
      })), { ...PLOTLY_LAYOUT_BASE,
        title: { text: 'Mean ' + metric + ' by dimension' + (filterFunc ? ' (f' + filterFunc + ')' : ''), font: { size: 14 } },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Dimension', type: 'category' },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Mean ' + metric, type: metric === 'gap' ? 'log' : 'linear',
          tickformat: metric === 'gap' ? '.2e' : '.2f' },
        margin: { l: 70, r: 20, t: 50, b: 60 }, legend: { orientation: 'h', y: -0.2 },
      }, { responsive: true });
    }

    function renderFunctionRankHeatmap(r) {
      const ids = r.selected;
      const dimVal = document.getElementById('funcRankDimFilter').value;
      const filterDim = dimVal !== 'all' ? Number(dimVal) : null;
      const funcs = [...r.rankByFunction.keys()].sort((a,b) => a-b);

      const z = ids.map(id => funcs.map(f => {
        const allRanks = r.rankByFunction.get(f)?.get(id) || [];
        if (filterDim === null) {
          return allRanks.length ? R(allRanks.reduce((a,b)=>a+b,0)/allRanks.length, 2) : null;
        }
        const filtered = [];
        const fRows = r.sharedRows.filter(row => row.function_id === f && row.dimension === filterDim);
        if (!fRows.length) return null;
        const valuesForRanking = fRows.map(row => {
          const vals = ids.map(eid => ({ id: eid, val: row['metric__' + expName(eid)] })).filter(v => Number.isFinite(v.val));
          vals.sort((a,b) => direction(r.metric)==='lower' ? a.val-b.val : b.val-a.val);
          const rank = {};
          vals.forEach((v,idx) => rank[v.id] = idx+1);
          return rank[id] || null;
        }).filter(v => v !== null);
        return valuesForRanking.length ? R(valuesForRanking.reduce((a,b)=>a+b,0)/valuesForRanking.length, 2) : null;
      }));

      const frExpNames = ids.map(expName);
      const frFuncLabels = funcs.map(f => 'f' + f + ' ' + (BBOB_NAMES[f]||'').substring(0,12));
      Plotly.newPlot('plotFunctionRank', [{
        type: 'heatmap',
        x: frFuncLabels,
        y: frExpNames, z,
        colorscale: [[0,'#4361ee'],[0.5,'#f5f7fb'],[1,'#ef4444']],
        colorbar: { title: 'Rank', titleside: 'right', thickness: 12 },
        hovertemplate: 'Function: %{x}<br>Experiment: %{y}<br>Mean Rank: %{z:.2f}<extra></extra>',
      }], { ...PLOTLY_LAYOUT_BASE,
        title: { text: 'Mean rank by function' + (filterDim ? ' (d=' + filterDim + ')' : '') + ' (lower = better)', font: { size: 14 } },
        margin: { l: 140, r: 80, t: 50, b: 100 },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, tickangle: -45, categoryorder: 'array', categoryarray: frFuncLabels },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, categoryorder: 'array', categoryarray: frExpNames },
      }, { responsive: true });
    }

    function renderECDF(r) {
      const metric = r.metric, ids = r.selected;
      if (metric !== 'gap' && metric !== 'evaluations') {
        Plotly.purge('plotECDF');
        document.getElementById('plotECDF').innerHTML = '<div style="padding:20px;text-align:center;color:#888;">ECDF not applicable for success rate metric.</div>';
        return;
      }
      const funcVal = document.getElementById('ecdfFuncFilter').value;
      const dimVal = document.getElementById('ecdfDimFilter').value;
      const filterFunc = funcVal !== 'all' ? Number(funcVal) : null;
      const filterDim = dimVal !== 'all' ? Number(dimVal) : null;

      const ecdfTraces = ids.map((id, i) => {
        const vals = [];
        for (const row of r.sharedRows) {
          if (filterFunc !== null && row.function_id !== filterFunc) continue;
          if (filterDim !== null && row.dimension !== filterDim) continue;
          const col = 'metric__' + expName(id);
          if (Number.isFinite(row[col]) && row[col] > 0) vals.push(row[col]);
        }
        vals.sort((a,b) => a - b);
        const n = vals.length || 1;
        return {
          type: 'scatter', mode: 'lines', name: expName(id),
          x: vals, y: vals.map((_, j) => R((j + 1) / n, 4)),
          line: { color: color(i), width: 2 },
        };
      });
      let subtitle = '';
      if (filterFunc) subtitle += ' f' + filterFunc;
      if (filterDim) subtitle += ' d=' + filterDim;

      Plotly.newPlot('plotECDF', ecdfTraces, { ...PLOTLY_LAYOUT_BASE,
        title: { text: 'ECDF: fraction of problems vs ' + metric + ' threshold' + subtitle, font: { size: 14 } },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, type: 'log', title: metric + ' threshold', exponentformat: 'e' },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Fraction of problems', range: [0, 1.05], tickformat: '.1f' },
        margin: { l: 60, r: 20, t: 50, b: 60 }, legend: { orientation: 'h', y: -0.2 },
      }, { responsive: true });
    }

    function renderConvergenceRate(r) {
      const ids = r.selected;
      const funcVal = document.getElementById('convFuncFilter').value;
      const filterFunc = funcVal !== 'all' ? Number(funcVal) : null;

      if (filterFunc === null) {
        const dimTraces = ids.map((id, i) => ({
          type: 'bar', name: expName(id),
          x: r.dimensions.map(d => 'd=' + d),
          y: r.dimensions.map(d => {
            const ck = id + '|' + d;
            const c = r.convergenceByDimExp[ck];
            return c ? R(c.converged * 100 / c.total, 1) : 0;
          }),
          marker: { color: color(i) },
        }));
        Plotly.newPlot('plotConvergence', dimTraces, { ...PLOTLY_LAYOUT_BASE,
          title: { text: 'Convergence rate by dimension', font: { size: 14 } },
          barmode: 'group', bargap: 0.2, bargroupgap: 0.05,
          xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Dimension' },
          yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Convergence rate (%)', range: [0, 105], tickformat: '.0f' },
          margin: { l: 50, r: 20, t: 50, b: 60 }, legend: { orientation: 'h', y: -0.2 },
        }, { responsive: true });
      } else {
        const dimTraces = ids.map((id, i) => ({
          type: 'bar', name: expName(id),
          x: r.dimensions.map(d => 'd=' + d),
          y: r.dimensions.map(d => {
            const matching = r.sharedRows.filter(row => row.function_id === filterFunc && row.dimension === d);
            if (!matching.length) return 0;
            const expRecs = RECORDS.filter(rec => rec.experiment_id === id && rec.function_id === filterFunc && rec.dimension === d);
            const conv = expRecs.filter(rec => rec.converged || rec.final_target_hit).length;
            return matching.length ? R(conv * 100 / matching.length, 1) : 0;
          }),
          marker: { color: color(i) },
        }));
        Plotly.newPlot('plotConvergence', dimTraces, { ...PLOTLY_LAYOUT_BASE,
          title: { text: 'Convergence rate by dimension (f' + filterFunc + ')', font: { size: 14 } },
          barmode: 'group', bargap: 0.2, bargroupgap: 0.05,
          xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Dimension' },
          yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Convergence rate (%)', range: [0, 105], tickformat: '.0f' },
          margin: { l: 50, r: 20, t: 50, b: 60 }, legend: { orientation: 'h', y: -0.2 },
        }, { responsive: true });
      }
    }

    function renderMedianGap(r) {
      const metric = r.metric, ids = r.selected;
      if (metric !== 'gap') {
        Plotly.purge('plotMedianGap');
        document.getElementById('plotMedianGap').innerHTML = '<div style="padding:20px;text-align:center;color:#888;">Median gap plot only available with gap metric.</div>';
        return;
      }
      const dimVal = document.getElementById('medianGapDimFilter').value;
      const filterDim = dimVal !== 'all' ? Number(dimVal) : null;

      const fTraces = ids.map((id, i) => ({
        type: 'bar', name: expName(id),
        x: r.functionIds.map(f => 'f' + f),
        y: r.functionIds.map(f => {
          let vals;
          if (filterDim === null) {
            const gKey = id + '|' + f;
            vals = r.gapsByExpFunc[gKey] || [];
          } else {
            vals = r.sharedRows
              .filter(row => row.function_id === f && row.dimension === filterDim)
              .map(row => row['metric__' + expName(id)])
              .filter(v => Number.isFinite(v));
          }
          return vals.length ? R(median(vals), 4) : null;
        }),
        marker: { color: color(i) },
      }));
      Plotly.newPlot('plotMedianGap', fTraces, { ...PLOTLY_LAYOUT_BASE,
        title: { text: 'Median gap by function' + (filterDim ? ' (d=' + filterDim + ')' : '') + ' (log scale)', font: { size: 14 } },
        barmode: 'group', bargap: 0.15,
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Function', tickangle: -45 },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Median gap', type: 'log', exponentformat: 'e' },
        margin: { l: 70, r: 20, t: 50, b: 60 }, legend: { orientation: 'h', y: -0.2 },
      }, { responsive: true });
    }

    function destroyTable(t) { try { if (t) t.destroy(); } catch(e) {} return null; }

    function renderTables(r) {
      aggregateTable = destroyTable(aggregateTable);
      sharedTable = destroyTable(sharedTable);
      functionSummaryTab = destroyTable(functionSummaryTab);
      pairwiseTab = destroyTable(pairwiseTab);

      console.log('[Dashboard] renderTables: aggregate=' + r.aggregateRows.length +
        ' funcSummary=' + r.functionSummaryRows.length +
        ' pairwise=' + r.pairwiseRows.length +
        ' shared=' + r.sharedRows.length);

      var numFmt = function(cell) { var v = cell.getValue(); return Rfmt(v); };
      var pctFmt = function(cell) { var v = cell.getValue(); return R(v,1) + '%'; };

      function initTable(elId, config) {
        try {
          var el = document.getElementById(elId);
          if (!el) { console.error('[Dashboard] Element #' + elId + ' not found'); return null; }
          el.innerHTML = '';
          var t = new Tabulator('#' + elId, config);
          console.log('[Dashboard] Table #' + elId + ' initialized, rows=' + (config.data ? config.data.length : 0));
          return t;
        } catch(e) {
          console.error('[Dashboard] Tabulator init error for #' + elId + ':', e);
          var el2 = document.getElementById(elId);
          if (el2) el2.innerHTML = '<div style="padding:16px;color:red;">Table error: ' + e.message + '</div>';
          return null;
        }
      }

      aggregateTable = initTable('aggregateTable', {
        data: r.aggregateRows.map(function(row) { return Object.assign({}, row); }),
        layout: 'fitDataStretch',
        movableColumns: true,
        maxHeight: '400px',
        columns: [
          { title: 'Experiment', field: 'experiment', headerFilter: 'input', minWidth: 150 },
          { title: 'Path', field: 'path', headerFilter: 'input', minWidth: 150 },
          { title: 'Keys', field: 'compared_keys', sorter: 'number', hozAlign: 'right' },
          { title: 'Mean Metric', field: 'mean_metric', sorter: 'number', formatter: numFmt, hozAlign: 'right' },
          { title: 'Median Metric', field: 'median_metric', sorter: 'number', formatter: numFmt, hozAlign: 'right' },
          { title: 'Wins', field: 'wins', sorter: 'number', hozAlign: 'right' },
          { title: 'Win Rate %', field: 'win_rate_pct', sorter: 'number', hozAlign: 'right', formatter: pctFmt },
          { title: 'Ties', field: 'ties', sorter: 'number', hozAlign: 'right' },
          { title: 'Mean log10 Ratio', field: 'mean_log10_ratio', sorter: 'number', formatter: numFmt, hozAlign: 'right' },
        ],
        initialSort: [{ column: 'win_rate_pct', dir: 'desc' }],
      });

      functionSummaryTab = initTable('functionSummaryTable', {
        data: r.functionSummaryRows.map(function(row) { return Object.assign({}, row); }),
        layout: 'fitDataStretch',
        movableColumns: true,
        maxHeight: '500px',
        pagination: 'local', paginationSize: 25,
        columns: [
          { title: 'Experiment', field: 'experiment', headerFilter: 'input', minWidth: 130 },
          { title: 'Func', field: 'function_id', sorter: 'number', headerFilter: 'number', hozAlign: 'center', width: 65 },
          { title: 'Function Name', field: 'function_name', headerFilter: 'input', minWidth: 140 },
          { title: 'Mean Metric', field: 'mean_metric', sorter: 'number', formatter: numFmt, hozAlign: 'right' },
          { title: 'Median Metric', field: 'median_metric', sorter: 'number', formatter: numFmt, hozAlign: 'right' },
          { title: 'Count', field: 'count', sorter: 'number', hozAlign: 'right', width: 65 },
          { title: 'Wins', field: 'wins', sorter: 'number', hozAlign: 'right', width: 65 },
          { title: 'Mean Rank', field: 'mean_rank', sorter: 'number', hozAlign: 'right', width: 90 },
        ],
        initialSort: [{ column: 'function_id', dir: 'asc' }],
      });

      pairwiseTab = initTable('pairwiseTable', {
        data: r.pairwiseRows.map(function(row) { return Object.assign({}, row); }),
        layout: 'fitDataStretch',
        movableColumns: true,
        maxHeight: '400px',
        columns: [
          { title: 'Experiment A', field: 'experiment_a', headerFilter: 'input', minWidth: 130 },
          { title: 'Experiment B', field: 'experiment_b', headerFilter: 'input', minWidth: 130 },
          { title: 'A Wins', field: 'wins_a', sorter: 'number', hozAlign: 'right' },
          { title: 'B Wins', field: 'wins_b', sorter: 'number', hozAlign: 'right' },
          { title: 'Ties', field: 'ties', sorter: 'number', hozAlign: 'right' },
          { title: 'A Rate %', field: 'a_rate', sorter: 'number', hozAlign: 'right', formatter: pctFmt },
          { title: 'B Rate %', field: 'b_rate', sorter: 'number', hozAlign: 'right', formatter: pctFmt },
        ],
      });

      var baseCols = [
        { title: 'Func', field: 'function_id', sorter: 'number', headerFilter: 'number', width: 60, hozAlign: 'center' },
        { title: 'Name', field: 'function_name', headerFilter: 'input', minWidth: 100 },
        { title: 'Dim', field: 'dimension', sorter: 'number', headerFilter: 'number', width: 55, hozAlign: 'center' },
        { title: 'Inst', field: 'instance_id', sorter: 'number', headerFilter: 'number', width: 55, hozAlign: 'center' },
        { title: 'Winner', field: 'winner', headerFilter: 'input', minWidth: 120 },
      ];
      var dynamicCols = [];
      if (r.sharedRows.length) {
        for (var key of Object.keys(r.sharedRows[0])) {
          if (!key.startsWith('metric__')) continue;
          dynamicCols.push({ title: key.slice(8), field: key, sorter: 'number', formatter: numFmt, hozAlign: 'right' });
        }
      }
      sharedTable = initTable('sharedTable', {
        data: r.sharedRows.map(function(row) { return Object.assign({}, row); }),
        layout: 'fitDataStretch',
        columns: baseCols.concat(dynamicCols),
        movableColumns: true,
        pagination: 'local', paginationSize: 25,
        maxHeight: '600px',
      });
    }

    // ===== EXPORT =====
    async function saveNodeAsImage(nodeId, fileName) {
      const node = document.getElementById(nodeId);
      if (!node) return;
      const canvas = await html2canvas(node, { backgroundColor: '#ffffff', scale: 2 });
      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a'); a.href = url; a.download = fileName; a.click();
    }
    async function savePlotAsImage(plotId, fileName) {
      const gd = document.getElementById(plotId);
      if (!gd) return;
      const dataUrl = await Plotly.toImage(gd, { format: 'png', width: 1400, height: 900, scale: 2 });
      const a = document.createElement('a'); a.href = dataUrl; a.download = fileName; a.click();
    }

    // ===== INFO MODAL =====
    let currentInfoKey = null;
    let currentLang = 'en';

    function showInfo(key) {
      currentInfoKey = key;
      const info = INFO[key];
      if (!info) return;
      currentLang = 'en';
      updateModalContent();
      document.getElementById('infoModal').classList.add('active');
    }

    function updateModalContent() {
      const info = INFO[currentInfoKey];
      if (!info) return;
      const data = info[currentLang];
      document.getElementById('modalTitle').textContent = data.title;
      document.getElementById('modalBody').innerHTML = data.body;
      document.querySelectorAll('.lang-btn').forEach(b => b.classList.toggle('active', b.dataset.lang === currentLang));
    }

    document.getElementById('modalClose').addEventListener('click', () => {
      document.getElementById('infoModal').classList.remove('active');
    });
    document.getElementById('infoModal').addEventListener('click', (e) => {
      if (e.target === document.getElementById('infoModal')) document.getElementById('infoModal').classList.remove('active');
    });
    document.querySelectorAll('.lang-btn').forEach(btn => {
      btn.addEventListener('click', () => { currentLang = btn.dataset.lang; updateModalContent(); });
    });

    // ===== VERBOSE CONVERGENCE LOGS =====
    var verboseExpData = {};
    var verboseFilteredMeta = [];
    var _verboseDataCache = {};
    var _verboseFetchInFlight = null;

    function collectVerboseData() {
      verboseExpData = {};
      var selected = EXPERIMENTS.filter(function(e) { return PRESELECTED.has(e.id); });
      for (var exp of selected) {
        if (exp.has_verbose && exp.verbose_traces && exp.verbose_traces.length) {
          verboseExpData[exp.id] = { name: exp.display_name, meta: exp.verbose_traces };
        }
      }
    }

    function populateVerboseUI() {
      collectVerboseData();
      var section = document.getElementById('verboseSection');
      var expSel = document.getElementById('verboseExpSelect');
      var ids = Object.keys(verboseExpData);
      if (!ids.length) { section.style.display = 'none'; return; }
      section.style.display = '';
      expSel.innerHTML = '';
      ids.forEach(function(id) {
        var o = document.createElement('option');
        o.value = id; o.textContent = verboseExpData[id].name;
        expSel.appendChild(o);
      });
      populateVerboseFilters();
    }

    function populateVerboseFilters() {
      var expId = document.getElementById('verboseExpSelect').value;
      var funcSel = document.getElementById('verboseFuncFilter');
      var dimSel = document.getElementById('verboseDimFilter');
      funcSel.innerHTML = '<option value="all">All functions</option>';
      dimSel.innerHTML = '<option value="all">All dimensions</option>';
      if (!verboseExpData[expId]) return;
      var funcs = new Set(), dims = new Set();
      verboseExpData[expId].meta.forEach(function(t) {
        if (t.function_id != null) funcs.add(t.function_id);
        if (t.dimension != null) dims.add(t.dimension);
      });
      [...funcs].sort(function(a,b){return a-b;}).forEach(function(f) {
        var o = document.createElement('option');
        o.value = f; o.textContent = 'f' + f + ' ' + (BBOB_NAMES[f]||'').substring(0,25);
        funcSel.appendChild(o);
      });
      [...dims].sort(function(a,b){return a-b;}).forEach(function(d) {
        var o = document.createElement('option');
        o.value = d; o.textContent = d + 'D';
        dimSel.appendChild(o);
      });
      applyVerboseFilter();
    }

    function applyVerboseFilter() {
      var expId = document.getElementById('verboseExpSelect').value;
      var funcVal = document.getElementById('verboseFuncFilter').value;
      var dimVal = document.getElementById('verboseDimFilter').value;
      if (!verboseExpData[expId]) { verboseFilteredMeta = []; return; }
      var allMeta = verboseExpData[expId].meta;
      verboseFilteredMeta = [];
      for (var gi = 0; gi < allMeta.length; gi++) {
        var t = allMeta[gi];
        if (funcVal !== 'all' && t.function_id !== parseInt(funcVal)) continue;
        if (dimVal !== 'all' && t.dimension !== parseInt(dimVal)) continue;
        verboseFilteredMeta.push({ meta: t, globalIdx: gi });
      }
      var traceSel = document.getElementById('verboseTraceSelect');
      traceSel.innerHTML = '';
      verboseFilteredMeta.forEach(function(item, i) {
        var t = item.meta;
        var o = document.createElement('option');
        o.value = i;
        var label = t.title;
        if (t.function_id != null) {
          var fname = BBOB_NAMES[t.function_id] || '';
          label = 'f' + t.function_id + ' ' + t.dimension + 'D i' + t.instance_id + (fname ? ' — ' + fname : '');
        }
        o.textContent = (i+1) + '. ' + label;
        traceSel.appendChild(o);
      });
      if (verboseFilteredMeta.length) {
        traceSel.value = 0;
        renderVerbosePlot();
      } else {
        ['plotVerboseAll','plotVerboseIter','plotVerboseTau','plotVerboseSamples'].forEach(function(id) { Plotly.purge(id); });
        document.getElementById('verboseCounter').textContent = '0 / 0';
      }
    }

    function _showVerboseLoading(show) {
      var plots = ['plotVerboseAll','plotVerboseIter','plotVerboseTau','plotVerboseSamples'];
      if (show) {
        plots.forEach(function(id) {
          var el = document.getElementById(id);
          if (!el.querySelector('.verbose-loading')) {
            var d = document.createElement('div');
            d.className = 'verbose-loading';
            d.style.cssText = 'display:flex;align-items:center;justify-content:center;height:300px;color:var(--text2);font-size:14px;';
            d.textContent = 'Loading trace data…';
            el.innerHTML = '';
            el.appendChild(d);
          }
        });
      }
    }

    function _drawVerbosePlots(t, traceLabel, expDisplayName) {
      var c = [getPaletteColor(0), getPaletteColor(1), getPaletteColor(2), getPaletteColor(3)];

      var evalNums = t.values.map(function(_, i) { return i + 1; });
      Plotly.react('plotVerboseAll', [
        { x: evalNums, y: t.values, mode: 'markers', name: 'f(x)', marker: { size: 3, color: c[0], opacity: 0.4 } },
        { x: evalNums, y: t.running_best, mode: 'lines', name: 'running best', line: { color: c[1], width: 2 } },
      ], {
        title: expDisplayName + ' — ' + traceLabel,
        xaxis: { title: 'Cumulative evaluation' },
        yaxis: { title: 'f(x)', tickformat: '.3g' },
        margin: { t: 40, b: 50, l: 65, r: 20 }, legend: { orientation: 'h', y: 1.12 }, height: 360,
      }, { responsive: true });

      Plotly.react('plotVerboseIter', [
        { x: t.iter_indices, y: t.iter_best, mode: 'lines+markers', name: 'best', line: { color: c[0] }, marker: { size: 5 } },
        { x: t.iter_indices, y: t.iter_mean, mode: 'lines+markers', name: 'mean', line: { color: c[3], dash: 'dash' }, marker: { size: 4 } },
        { x: t.iter_indices, y: t.iter_median, mode: 'lines+markers', name: 'median', line: { color: c[2], dash: 'dot' }, marker: { size: 4 } },
      ], {
        title: 'Per-iteration — ' + traceLabel,
        xaxis: { title: 'Iteration', dtick: 1 }, yaxis: { title: 'f(x)', tickformat: '.3g' },
        margin: { t: 40, b: 50, l: 65, r: 20 }, legend: { orientation: 'h', y: 1.12 }, height: 360,
      }, { responsive: true });

      var tauTraces = [];
      var hasTau = t.iter_taus && t.iter_taus.some(function(v) { return v !== null; });
      if (hasTau) {
        tauTraces.push({ x: t.iter_indices, y: t.iter_taus, mode: 'lines+markers', name: 'tau (elite floor)',
          line: { color: c[1], width: 2 }, marker: { size: 5 }, yaxis: 'y' });
      }
      var hasStd = t.iter_std && t.iter_std.some(function(v) { return v > 0; });
      if (hasStd) {
        tauTraces.push({ x: t.iter_indices, y: t.iter_std, mode: 'lines+markers', name: 'std(f(x))',
          line: { color: c[2], dash: 'dash' }, marker: { size: 4 }, yaxis: hasTau ? 'y2' : 'y' });
      }
      var tauLayout = {
        title: 'Tau & spread — ' + traceLabel,
        xaxis: { title: 'Iteration', dtick: 1 },
        yaxis: { title: hasTau ? 'tau' : 'std', tickformat: '.3g' },
        margin: { t: 40, b: 50, l: 65, r: hasStd && hasTau ? 65 : 20 },
        legend: { orientation: 'h', y: 1.12 }, height: 360,
      };
      if (hasTau && hasStd) {
        tauLayout.yaxis2 = { title: 'std', overlaying: 'y', side: 'right', tickformat: '.3g' };
      }
      if (tauTraces.length) {
        Plotly.react('plotVerboseTau', tauTraces, tauLayout, { responsive: true });
      } else {
        Plotly.react('plotVerboseTau', [{ x: [0], y: [0], mode: 'text', text: ['No tau data'], textposition: 'middle center' }],
          { xaxis: {visible:false}, yaxis: {visible:false}, height: 360, margin: {t:40,b:20,l:20,r:20} }, {responsive:true});
      }

      var improvement = [];
      for (var k = 0; k < t.iter_best.length; k++) {
        if (k === 0) { improvement.push(0); }
        else { improvement.push(R(t.iter_best[k-1] - t.iter_best[k], 6)); }
      }
      Plotly.react('plotVerboseSamples', [
        { x: t.iter_indices, y: t.iter_count, type: 'bar', name: 'samples/iter',
          marker: { color: c[0], opacity: 0.6 }, yaxis: 'y' },
        { x: t.iter_indices, y: improvement, mode: 'lines+markers', name: 'best-value drop',
          line: { color: c[1], width: 2 }, marker: { size: 5 }, yaxis: 'y2' },
      ], {
        title: 'Sampling & best-value drop — ' + traceLabel,
        xaxis: { title: 'Iteration', dtick: 1 },
        yaxis: { title: 'samples', side: 'left' },
        yaxis2: { title: 'best-value drop', overlaying: 'y', side: 'right', tickformat: '.3g' },
        margin: { t: 40, b: 50, l: 55, r: 55 }, legend: { orientation: 'h', y: 1.12 }, height: 360,
        barmode: 'overlay',
      }, { responsive: true });
    }

    function renderVerbosePlot() {
      var expId = document.getElementById('verboseExpSelect').value;
      if (!verboseExpData[expId] || !verboseFilteredMeta.length) return;
      var localIdx = parseInt(document.getElementById('verboseTraceSelect').value) || 0;
      if (localIdx < 0 || localIdx >= verboseFilteredMeta.length) localIdx = 0;
      var item = verboseFilteredMeta[localIdx];
      var globalIdx = item.globalIdx;
      var traceLabel = item.meta.title;
      var expDisplayName = verboseExpData[expId].name;
      document.getElementById('verboseCounter').textContent = (localIdx+1) + ' / ' + verboseFilteredMeta.length;

      var cacheKey = expId + '|' + globalIdx;
      if (_verboseDataCache[cacheKey]) {
        _drawVerbosePlots(_verboseDataCache[cacheKey], traceLabel, expDisplayName);
        return;
      }

      _showVerboseLoading(true);
      var fetchId = cacheKey;
      _verboseFetchInFlight = fetchId;

      fetch('/api/verbose_trace?exp=' + encodeURIComponent(expId) + '&idx=' + globalIdx)
        .then(function(resp) {
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          return resp.json();
        })
        .then(function(data) {
          _verboseDataCache[cacheKey] = data;
          if (_verboseFetchInFlight === fetchId) {
            _drawVerbosePlots(data, traceLabel, expDisplayName);
          }
        })
        .catch(function(err) {
          console.error('Verbose trace fetch error:', err);
          if (_verboseFetchInFlight === fetchId) {
            ['plotVerboseAll','plotVerboseIter','plotVerboseTau','plotVerboseSamples'].forEach(function(id) {
              document.getElementById(id).innerHTML = '<div style="padding:20px;text-align:center;color:#e63946;">Failed to load trace: ' + err.message + '</div>';
            });
          }
        });
    }

    document.getElementById('verboseExpSelect').addEventListener('change', populateVerboseFilters);
    document.getElementById('verboseFuncFilter').addEventListener('change', applyVerboseFilter);
    document.getElementById('verboseDimFilter').addEventListener('change', applyVerboseFilter);
    document.getElementById('verboseTraceSelect').addEventListener('change', renderVerbosePlot);
    document.getElementById('verbosePrev').addEventListener('click', function() {
      var sel = document.getElementById('verboseTraceSelect');
      var idx = parseInt(sel.value) || 0;
      if (idx > 0) { sel.value = idx - 1; renderVerbosePlot(); }
    });
    document.getElementById('verboseNext').addEventListener('click', function() {
      var sel = document.getElementById('verboseTraceSelect');
      var idx = parseInt(sel.value) || 0;
      if (idx < verboseFilteredMeta.length - 1) { sel.value = idx + 1; renderVerbosePlot(); }
    });

    // ===== MAIN REFRESH =====
    function refreshAll() {
      const result = computeComparison();
      if (result.error) {
        lastResult = null;
        updateKpis(null);
        setStatus(result.error, true);
        ['plotWins','plotRelative','plotDimension','plotFunctionRank','plotECDF','plotConvergence','plotMedianGap','plotPairwise'].forEach(id => Plotly.purge(id));
        aggregateTable = destroyTable(aggregateTable);
        sharedTable = destroyTable(sharedTable);
        functionSummaryTab = destroyTable(functionSummaryTab);
        pairwiseTab = destroyTable(pairwiseTab);
        return;
      }
      lastResult = result;
      setStatus('Compared ' + result.selected.length + ' experiments on ' + result.validKeyCount + ' shared triplets with metric "' + result.metric + '".', false);
      updateKpis(result);
      populateChartFilters(result);
      renderPlots(result);
      renderTables(result);
      populateVerboseUI();
    }

    // ===== RELOAD EXPERIMENTS =====
    document.getElementById('btnReloadExperiments').addEventListener('click', function() {
      var btn = this;
      btn.textContent = 'Reloading...'; btn.disabled = true;
      fetch(window.location.pathname.replace(/[^/]*$/, '') + 'api/reload')
        .then(function(r) { return r.json(); })
        .then(function(data) {
          if (data.reload_url) {
            window.location.href = data.reload_url;
          } else if (data.error) {
            alert('Reload failed: ' + data.error);
          }
        })
        .catch(function(e) { alert('Reload not available (server may not support it). Refresh the page manually after re-running the build script.'); })
        .finally(function() { btn.textContent = '\u21bb Reload experiments'; btn.disabled = false; });
    });

    // ===== EVENT LISTENERS =====
    document.getElementById('searchExperiments').addEventListener('input', filterTree);
    document.getElementById('btnSelectAll').addEventListener('click', () => {
      for (const exp of EXPERIMENTS) PRESELECTED.add(exp.id);
      buildTreeUI(); refreshAll();
    });
    document.getElementById('btnClearAll').addEventListener('click', () => {
      PRESELECTED.clear(); buildTreeUI(); refreshAll();
    });
    document.getElementById('btnRefresh').addEventListener('click', refreshAll);
    document.getElementById('metricSelect').addEventListener('change', refreshAll);
    ['functionFilter','dimensionFilter','instanceFilter'].forEach(id => {
      document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') refreshAll(); });
    });

    // Per-chart filter listeners
    document.getElementById('dimScalingFuncFilter').addEventListener('change', () => { if (lastResult) renderDimensionScaling(lastResult); });
    document.getElementById('funcRankDimFilter').addEventListener('change', () => { if (lastResult) renderFunctionRankHeatmap(lastResult); });
    document.getElementById('ecdfFuncFilter').addEventListener('change', () => { if (lastResult) renderECDF(lastResult); });
    document.getElementById('ecdfDimFilter').addEventListener('change', () => { if (lastResult) renderECDF(lastResult); });
    document.getElementById('convFuncFilter').addEventListener('change', () => { if (lastResult) renderConvergenceRate(lastResult); });
    document.getElementById('medianGapDimFilter').addEventListener('change', () => { if (lastResult) renderMedianGap(lastResult); });

    document.querySelectorAll('[data-export-node]').forEach(btn => {
      btn.addEventListener('click', () => saveNodeAsImage(btn.dataset.exportNode, btn.dataset.exportNode + '.png'));
    });
    document.querySelectorAll('[data-export-plot]').forEach(btn => {
      btn.addEventListener('click', () => savePlotAsImage(btn.dataset.exportPlot, btn.dataset.exportPlot + '.png'));
    });
    document.querySelectorAll('[data-info]').forEach(btn => {
      btn.addEventListener('click', () => showInfo(btn.dataset.info));
    });

    // ===== INIT =====
    try {
      buildTreeUI();
      refreshAll();
    } catch(e) {
      console.error('[Dashboard] Init error:', e);
      document.getElementById('statusMessage').textContent = 'Initialization error: ' + e.message;
      document.getElementById('statusMessage').className = 'status-msg error';
    }
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive COCO/BBOB HTML dashboard from experiment directories."
    )
    parser.add_argument(
        "--roots", nargs="+", default=["exdata"],
        help="Root directories to recursively scan for experiments.",
    )
    parser.add_argument(
        "--output", default="outputs/coco_comparison_dashboard.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--select", nargs="*", default=[],
        help="Optional experiment directories to preselect in the dashboard.",
    )
    parser.add_argument(
        "--serve", action="store_true", default=False,
        help="Start a local HTTP server after generating the dashboard (useful for WSL).",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port for the HTTP server (used with --serve).",
    )
    return parser.parse_args()


_EXPERIMENTS_REGISTRY: List[Experiment] = []
_TRACE_CACHE: Dict[str, Optional[dict]] = {}
_TRACE_CACHE_MAX = 256


def _get_trace_data(exp_id: str, trace_idx: int) -> Optional[dict]:
    """Load and cache verbose trace data for a single (experiment, trace) pair."""
    cache_key = f"{exp_id}|{trace_idx}"
    if cache_key in _TRACE_CACHE:
        return _TRACE_CACHE[cache_key]

    exp = next((e for e in _EXPERIMENTS_REGISTRY if e.exp_id == exp_id), None)
    if exp is None or trace_idx < 0 or trace_idx >= len(exp.verbose_traces):
        return None

    trace_meta = exp.verbose_traces[trace_idx]
    npz_path = exp.abs_path / "verbose_logs" / trace_meta.filename
    if not npz_path.exists():
        return None

    result = _process_single_npz(npz_path)

    if len(_TRACE_CACHE) >= _TRACE_CACHE_MAX:
        oldest_key = next(iter(_TRACE_CACHE))
        del _TRACE_CACHE[oldest_key]
    _TRACE_CACHE[cache_key] = result
    return result


def _build_dashboard(args, project_root: Path) -> Tuple[Path, int]:
    """Build/rebuild the dashboard HTML. Returns (output_path, n_experiments)."""
    global _EXPERIMENTS_REGISTRY
    roots = [Path(p).resolve() if Path(p).is_absolute() else (project_root / p).resolve() for p in args.roots]
    experiment_dirs = discover_experiment_dirs(roots)
    if not experiment_dirs:
        raise SystemExit("No experiment directories found under provided roots.")

    experiments: List[Experiment] = []
    for idx, exp_dir in enumerate(experiment_dirs, start=1):
        exp = load_experiment(exp_dir, project_root, idx)
        if exp is not None:
            experiments.append(exp)

    if not experiments:
        raise SystemExit("No valid experiment data found (CSV or INFO parsing yielded zero records).")

    assign_unique_display_names(experiments)
    _EXPERIMENTS_REGISTRY = experiments
    _TRACE_CACHE.clear()

    explicit_select = [
        Path(p).resolve() if Path(p).is_absolute() else (project_root / p).resolve()
        for p in args.select
    ]
    preselected_ids = choose_preselected(experiments, explicit_select)

    payload = to_json_payload(experiments, preselected_ids)
    html = render_dashboard_html(payload)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path, len(experiments)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()

    output_path, n_experiments = _build_dashboard(args, project_root)
    print(f"Dashboard written to: {output_path}", flush=True)
    print(f"Experiments loaded: {n_experiments}", flush=True)

    if args.serve:
        serve_dir = str(output_path.parent)
        filename = output_path.name

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=serve_dir, **kw)

            def log_message(self, fmt, *a):
                print(f"  {self.address_string()} - {fmt % a}", flush=True)

            def handle_one_request(self):
                try:
                    super().handle_one_request()
                except BrokenPipeError:
                    pass

            def finish(self):
                try:
                    super().finish()
                except BrokenPipeError:
                    pass

            def _json_response(self, code, obj):
                body = json.dumps(obj, separators=(",", ":")).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                path_clean = parsed.path.rstrip("/")

                if path_clean.endswith("/api/reload"):
                    try:
                        _out, n = _build_dashboard(args, project_root)
                        self._json_response(200, {"ok": True, "experiments": n, "reload_url": f"/{filename}"})
                        print(f"  [Reload] Dashboard rebuilt with {n} experiments.", flush=True)
                    except Exception as exc:
                        self._json_response(500, {"error": str(exc)})
                    return

                if path_clean.endswith("/api/verbose_trace"):
                    qs = parse_qs(parsed.query)
                    exp_id = qs.get("exp", [None])[0]
                    idx_str = qs.get("idx", [None])[0]
                    if not exp_id or idx_str is None:
                        self._json_response(400, {"error": "Missing exp or idx parameter"})
                        return
                    try:
                        trace_idx = int(idx_str)
                    except ValueError:
                        self._json_response(400, {"error": "idx must be an integer"})
                        return
                    data = _get_trace_data(exp_id, trace_idx)
                    if data is None:
                        self._json_response(404, {"error": "Trace not found or empty"})
                        return
                    self._json_response(200, data)
                    return

                super().do_GET()

        import signal
        import socket
        import subprocess

        def _get_lan_ip() -> str:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(1)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except Exception:
                pass
            try:
                ip = socket.gethostbyname(socket.gethostname())
                if not ip.startswith("127."):
                    return ip
            except Exception:
                pass
            return "localhost"

        def _port_in_use(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("127.0.0.1", port)) == 0

        def _kill_port_holder(port: int) -> bool:
            """Try to kill whatever process holds *port*. Returns True on success."""
            try:
                out = subprocess.check_output(
                    ["fuser", f"{port}/tcp"], stderr=subprocess.DEVNULL, text=True,
                ).strip()
                pids = [int(p) for p in out.split() if p.strip().isdigit()]
                for pid in pids:
                    if pid == os.getpid():
                        continue
                    os.kill(pid, signal.SIGTERM)
                import time; time.sleep(0.5)
                for pid in pids:
                    try:
                        os.kill(pid, 0)
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                import time; time.sleep(0.3)
                return not _port_in_use(port)
            except Exception:
                return False

        def _find_free_port(start: int, max_attempts: int = 20) -> int:
            for offset in range(max_attempts):
                port = start + offset
                if not _port_in_use(port):
                    return port
            raise RuntimeError(f"No free port found in range {start}–{start + max_attempts - 1}")

        port = args.port
        if _port_in_use(port):
            print(f"Port {port} is in use.", end=" ", flush=True)
            if _kill_port_holder(port):
                print(f"Freed port {port}.", flush=True)
            else:
                old_port = port
                port = _find_free_port(port + 1)
                print(f"Could not free port {old_port}, using port {port} instead.", flush=True)

        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("0.0.0.0", port), Handler)

        def _shutdown(signum, frame):
            print("\nShutting down server...", flush=True)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            t = threading.Thread(target=httpd.shutdown)
            t.start()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        ip = _get_lan_ip()
        url_lan = f"http://{ip}:{port}/{filename}"
        url_local = f"http://localhost:{port}/{filename}"
        print(f"\nServing dashboard at:", flush=True)
        print(f"  Windows browser:  {url_local}", flush=True)
        print(f"  WSL LAN:          {url_lan}", flush=True)
        print(f"\nIf localhost doesn't work, run in Windows PowerShell (admin):", flush=True)
        print(f"  netsh interface portproxy add v4tov4 listenport={port} listenaddress=0.0.0.0 connectport={port} connectaddress={ip}", flush=True)
        print("Press Ctrl+C to stop the server.\n", flush=True)

        try:
            httpd.serve_forever(poll_interval=0.5)
        except Exception:
            pass
        finally:
            try:
                httpd.server_close()
            except Exception:
                pass
            print("Server stopped.", flush=True)
    else:
        print("Tip: re-run with --serve to start an HTTP server for easy browser access from Windows.")


if __name__ == "__main__":
    main()
