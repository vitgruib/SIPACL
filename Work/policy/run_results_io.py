"""
Shared CSV load/save for PPO run summaries (same columns as the former Excel schema).

Uses UTF-8 with BOM so double-click in Excel shows columns correctly on Windows.
List columns (episodic_returns, episodic_lengths, lp_per_episode, eval_returns) are Python repr
of lists via str(list), with empty lists stored as empty cells (same pattern for returns and LP).
Legacy CSV files may still have extra columns (e.g. old lp_scene_*); append_run_row_csv merges them.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Tuple

RUN_RESULTS_HEADER: List[str] = [
    "run_name",
    "time",
    "seed",
    "replay_resample_prob",
    "total_timesteps",
    "num_envs",
    "learning_rate",
    "gamma",
    "exp_name",
    "eval_episodes",
    "eval_max_steps",
    "mean_episodic_return",
    "std_episodic_return",
    "mean_episodic_length",
    "num_episodes",
    "SPS",
    "episodic_returns",
    "episodic_lengths",
    "lp_per_episode",
    "mean_eval_return",
    "std_eval_return",
    "num_eval_episodes",
    "eval_returns",
]

_RUN_RESULTS_HEADER_SET = set(RUN_RESULTS_HEADER)


def _lp_scene_col_sort_key(name: str) -> Tuple[int, Any]:
    """Sort extra columns: lp_scene_<int> by scene id, then other names lexically."""
    if name.startswith("lp_scene_"):
        suf = name[len("lp_scene_") :]
        try:
            return (0, int(suf))
        except ValueError:
            pass
    return (1, name)


def _sorted_extra_keys(keys: Iterable[str]) -> List[str]:
    return sorted((k for k in keys if k not in _RUN_RESULTS_HEADER_SET), key=_lp_scene_col_sort_key)


def _unique_headers(header_row: List[Any]) -> List[str]:
    """First-row names; preserve order; disambiguate duplicates (same as Excel path)."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for j, cell in enumerate(header_row):
        if cell is None or (isinstance(cell, str) and cell.strip() == ""):
            base = f"_col{j}"
        else:
            base = str(cell).strip()
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}_{n}")
    return out


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def load_runs(path: str) -> List[Dict[str, Any]]:
    """
    Load all data rows as dicts keyed by header names (same shape as the old Excel export).
    Skips rows with empty run_name.
    """
    lower = path.lower()
    if lower.endswith(".xlsx"):
        raise ValueError(
            "This project now uses CSV for run results. Use a .csv file "
            "(export from Excel if needed) or re-run training to append to runs_results.csv."
        )
    if not lower.endswith(".csv"):
        raise ValueError(f"Expected a .csv file, got: {path}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    headers = _unique_headers(rows[0])
    runs: List[Dict[str, Any]] = []
    for row in rows[1:]:
        if not row or row[0] is None or str(row[0]).strip() == "":
            continue
        d: Dict[str, Any] = {}
        for j, h in enumerate(headers):
            d[h] = row[j] if j < len(row) else None
        for j in range(len(headers), len(row)):
            if row[j] is not None and str(row[j]).strip() != "":
                d[f"_extra_col{j}"] = row[j]
        runs.append(d)
    return runs


def append_run_row_csv(path: str, row_values: Dict[str, Any]) -> None:
    """
    Append one run: fixed columns in RUN_RESULTS_HEADER order, then optional extra columns
    from older files. Extra column names are merged across the whole file so older rows get
    empty cells for columns added later.
    """
    missing = _RUN_RESULTS_HEADER_SET - set(row_values.keys())
    if missing:
        raise KeyError(f"row_values missing keys: {sorted(missing)}")

    extras_in_row = _sorted_extra_keys(row_values.keys())

    def merged_header(first_row_cells: List[str]) -> List[str]:
        if not first_row_cells:
            return list(RUN_RESULTS_HEADER) + extras_in_row
        file_hdr = _unique_headers(first_row_cells)
        extras_in_file = [h for h in file_hdr if h not in _RUN_RESULTS_HEADER_SET]
        merged_extras = _sorted_extra_keys(set(extras_in_file) | set(extras_in_row))
        return list(RUN_RESULTS_HEADER) + merged_extras

    def row_to_dict(headers: List[str], row: List[str]) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for j, h in enumerate(headers):
            d[h] = row[j] if j < len(row) else ""
        return d

    def dict_to_line(d: Dict[str, Any], headers: List[str]) -> List[str]:
        return [_csv_cell(d.get(h, "")) for h in headers]

    if not os.path.isfile(path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        header = merged_header([])
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(header)
            w.writerow(dict_to_line(row_values, header))
        return

    with open(path, newline="", encoding="utf-8-sig") as f:
        existing = list(csv.reader(f))

    if not existing:
        header = merged_header([])
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(header)
            w.writerow(dict_to_line(row_values, header))
        return

    old_header = _unique_headers(existing[0])
    header = merged_header(existing[0])
    body_rows: List[List[str]] = []
    for row in existing[1:]:
        if not row or row[0] is None or str(row[0]).strip() == "":
            continue
        d = row_to_dict(old_header, row)
        body_rows.append(dict_to_line(d, header))

    body_rows.append(dict_to_line(row_values, header))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        w.writerows(body_rows)
