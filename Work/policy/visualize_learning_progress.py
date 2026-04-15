#!/usr/bin/env python3
"""
Plot learning-progress (LP) from a run row in ``runs_results.csv``.

Opens an **interactive figure window** by default (Matplotlib ``plt.show()``). Pass ``-o path.png``
only if you also want a PNG written to disk.

Reads **the last data row** in the file (bottom of the sheet = most recently appended run),
then parses the ``lp_per_episode`` cell (same ``str(list)`` format as ``episodic_returns``).

By default, LP values **≥ 100000** are not drawn at their true magnitude; they are shown as
gray **^** markers at a placeholder height. Override with ``--mask-lp-above`` (e.g. ``1e100``
to disable).

Usage (from ``Work/``)::

  python policy/visualize_learning_progress.py
  python policy/visualize_learning_progress.py --runs-results policy/runs_results.csv
  python policy/visualize_learning_progress.py -o lp.png
    # -o optional; default opens only a GUI window (close window to exit)
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Any, List, Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

from run_results_io import load_runs

DEFAULT_MASK_LP_ABOVE = 100_000.0
DEFAULT_LP_COLUMN = "lp_per_episode"


def _lp_mask_threshold(mask_lp_above: float) -> Optional[float]:
    """Values >= threshold are plotted as masked placeholders; None / non-finite disables."""
    if mask_lp_above is None or not np.isfinite(mask_lp_above) or mask_lp_above <= 0:
        return None
    return float(mask_lp_above)


def _masked_scatter_y_placeholder(lp_values: List[float], mask: np.ndarray) -> float:
    """Y position for masked points (not the real LP); keeps them visible below normal range."""
    arr = np.asarray(lp_values, dtype=np.float64)
    ok = (~mask) & np.isfinite(arr)
    if not np.any(ok):
        return 0.0
    lo = float(np.nanmin(arr[ok]))
    if lo > 0:
        return lo * 0.5
    return lo - 1e-6


def _parse_str_list_floats(cell: Any) -> List[float]:
    """Parse CSV cell written as str(list) from PPO (same as episodic_returns)."""
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(v, list):
        return []
    out: List[float] = []
    for x in v:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize LP from the last row of runs_results.csv (lp_per_episode)."
    )
    p.add_argument(
        "--runs-results",
        default=os.path.join(_script_dir, "runs_results.csv"),
        help="Path to runs_results.csv (default: policy/runs_results.csv next to this script).",
    )
    p.add_argument(
        "--run-index",
        type=int,
        default=-1,
        help="Which run row to plot: -1 = last row in file (default), -2 = second-to-last, etc.",
    )
    p.add_argument(
        "--column",
        default=DEFAULT_LP_COLUMN,
        help=f"Column name to parse as str(list) of floats (default: {DEFAULT_LP_COLUMN}).",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="If set, also save the figure to this PNG path. Default: only display a window.",
    )
    p.add_argument(
        "--mask-lp-above",
        type=float,
        default=DEFAULT_MASK_LP_ABOVE,
        help=(
            "LP values >= this are not shown at true magnitude; shown as masked markers "
            f"(default {DEFAULT_MASK_LP_ABOVE:g}). Use a very large value (e.g. 1e100) to disable."
        ),
    )
    args = p.parse_args()

    path = os.path.abspath(args.runs_results)
    if not os.path.isfile(path):
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        runs = load_runs(path)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if not runs:
        print("No runs in CSV (empty or only header).", file=sys.stderr)
        sys.exit(1)

    try:
        row = runs[args.run_index]
    except IndexError:
        print(f"run_index {args.run_index} out of range ({len(runs)} runs).", file=sys.stderr)
        sys.exit(1)

    col = args.column
    if col not in row:
        print(f"Column {col!r} not in row. Keys: {sorted(row.keys())}", file=sys.stderr)
        sys.exit(1)

    lpv = _parse_str_list_floats(row[col])
    if not lpv:
        print(
            f"No numeric LP values in column {col!r} for run_name={row.get('run_name', '')!r}.",
            file=sys.stderr,
        )
        sys.exit(1)

    seq = list(range(1, len(lpv) + 1))
    run_name = str(row.get("run_name", "")).strip() or "(unnamed)"

    fig, ax_lp = plt.subplots(1, 1, figsize=(10, 3.8))
    thr = _lp_mask_threshold(args.mask_lp_above)
    lp_arr = np.asarray(lpv, dtype=np.float64)
    if thr is not None:
        big = lp_arr >= thr
    else:
        big = np.zeros(len(lpv), dtype=bool)
    y_ph = _masked_scatter_y_placeholder(lpv, big) if thr is not None and np.any(big) else None

    xs_ok = [s for s, b in zip(seq, big) if not b]
    ys_ok = [v for v, b in zip(lpv, big) if not b]
    if xs_ok:
        ax_lp.scatter(xs_ok, ys_ok, s=14, alpha=0.8, label="LP", c="C0")
    if y_ph is not None:
        xs_m = [s for s, b in zip(seq, big) if b]
        if xs_m:
            ys_m = [y_ph] * len(xs_m)
            ax_lp.scatter(
                xs_m,
                ys_m,
                s=36,
                alpha=0.85,
                marker="^",
                c="0.35",
                edgecolors="k",
                linewidths=0.4,
                label=f"masked (LP≥{thr:g})",
            )

    unmasked = lp_arr[~big]
    unmasked = unmasked[np.isfinite(unmasked)]
    if unmasked.size > 0:
        lo = float(np.nanmin(unmasked))
        hi = float(np.nanmax(unmasked))
        span = hi - lo
        pad = 0.05 * span if span > 0 else max(abs(hi) * 0.05, 1e-9)
        ymax = hi + pad
        ymin = lo - pad
        if y_ph is not None and y_ph < ymin:
            ymin = y_ph - pad
        ax_lp.set_ylim(ymin, ymax)
    elif y_ph is not None:
        ax_lp.set_ylim(y_ph - 1.0, y_ph + 1.0)

    ax_lp.set_xlabel("episode index (order in lp_per_episode list)")
    ax_lp.set_ylabel("lp_per_episode (from runs_results row)")
    ax_lp.set_title(f"Learning progress — {run_name}")
    ax_lp.legend(loc="upper right", fontsize=8)
    ax_lp.grid(True, alpha=0.3)

    fig.tight_layout()
    if args.output:
        out = os.path.abspath(args.output)
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Wrote {out} (run_index={args.run_index}, n={len(lpv)})")
    else:
        print(f"Showing plot (run_index={args.run_index}, n={len(lpv)}). Close the window to exit.")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
