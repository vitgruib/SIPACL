#!/usr/bin/env python3
"""
Visualize runs from a CSV (default: policy/tempresults.csv):
  Eval: one datapoint per eval episode (eval_returns, else mean_eval_return).
  Boxplot + frequency dot plot; dot spacing scales with stack depth; table of
  mean of ep means, std of ep means, and mean of ep std (across runs).
  Column names come from the first CSV row.
Usage (from Work/): python policy/visualize_runs.py [-f policy/tempresults.csv]
"""
import argparse
import ast
import os
import sys

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from run_results_io import load_runs

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("Install matplotlib: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def _num(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _parse_float_list(cell):
    """Parse eval_returns / episodic_returns cell (list or '[1.0, 2.0, ...]') to floats."""
    if cell is None or (isinstance(cell, str) and cell.strip() == ""):
        return []
    if isinstance(cell, (list, tuple)):
        out = []
        for x in cell:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                pass
        return out
    try:
        out = ast.literal_eval(str(cell).strip())
        return [float(x) for x in out] if out else []
    except (ValueError, SyntaxError, TypeError):
        return []


def _eval_episode_returns_for_run(run):
    """All eval episode returns for one CSV row; fallback to one point from mean_eval_return."""
    vals = _parse_float_list(run.get("eval_returns"))
    if vals:
        return np.asarray(vals, dtype=float)
    m = _num(run.get("mean_eval_return"))
    if not np.isnan(m):
        return np.array([m], dtype=float)
    return np.array([], dtype=float)


def _build_eval_series_by_prob(prob_levels, by_prob):
    """List of (prob, 1d array) with every eval episode in that group."""
    series = []
    for p in prob_levels:
        parts = [_eval_episode_returns_for_run(r) for r in by_prob[p]]
        parts = [x for x in parts if x.size > 0]
        if not parts:
            continue
        cat = np.concatenate(parts)
        cat = cat[np.isfinite(cat)]
        if cat.size > 0:
            series.append((p, cat))
    return series


def _build_eval_series_flat(runs):
    """Concatenate all eval episode returns across runs (no replay_resample_prob)."""
    parts = [_eval_episode_returns_for_run(r) for r in runs]
    parts = [x for x in parts if x.size > 0]
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def _max_horizontal_stack(series, edges):
    """Largest number of points in one histogram bin at one x-column (for dot spacing)."""
    nb = len(edges) - 1
    m = 1
    for _p, vals in series:
        bin_ids = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, nb - 1)
        by_bin = {}
        for yi, b in zip(vals, bin_ids):
            by_bin.setdefault(int(b), []).append(float(yi))
        for vs in by_bin.values():
            m = max(m, len(vs))
    return m


def _group_inter_intra_stats(runs_list):
    """
    Per run: ep mean = mean(eval episodes), ep std = std(eval episodes).
    mean of ep means = mean over runs of ep mean.
    std of ep means = std over runs of ep mean.
    mean of ep std = mean over runs of ep std.
    """
    run_means = []
    run_stds = []
    n_ep = 0
    for r in runs_list:
        ep = _eval_episode_returns_for_run(r)
        if ep.size == 0:
            continue
        n_ep += int(ep.size)
        run_means.append(float(np.mean(ep)))
        run_stds.append(float(np.std(ep, ddof=1)) if ep.size > 1 else 0.0)
    if not run_means:
        return None
    mus = np.array(run_means, dtype=float)
    sds = np.array(run_stds, dtype=float)
    return {
        "n_runs": len(run_means),
        "n_episodes": n_ep,
        "inter_mean": float(np.mean(mus)),
        "inter_std": float(np.std(mus, ddof=1)) if mus.size > 1 else 0.0,
        "intra_mean_std": float(np.mean(sds)),
    }


def _stats_table_rows(prob_levels, by_prob, runs):
    """Rows for matplotlib.table; one row per setting or single 'all' row."""
    headers = [
        "replay_p",
        "runs",
        "episodes",
        "mean of ep means",
        "std of ep means",
        "mean of ep std",
    ]
    rows = []
    if prob_levels:
        for p in prob_levels:
            st = _group_inter_intra_stats(by_prob[p])
            if st is None:
                rows.append([_prob_short(p), "0", "0", "—", "—", "—"])
            else:
                rows.append(
                    [
                        _prob_short(p),
                        str(st["n_runs"]),
                        str(st["n_episodes"]),
                        f"{st['inter_mean']:.3g}",
                        f"{st['inter_std']:.3g}",
                        f"{st['intra_mean_std']:.3g}",
                    ]
                )
    else:
        st = _group_inter_intra_stats(runs)
        if st is None:
            rows.append(["(all)", "0", "0", "—", "—", "—"])
        else:
            rows.append(
                [
                    "(all)",
                    str(st["n_runs"]),
                    str(st["n_episodes"]),
                    f"{st['inter_mean']:.3g}",
                    f"{st['inter_std']:.3g}",
                    f"{st['intra_mean_std']:.3g}",
                ]
            )
    return headers, rows


def _draw_inter_intra_table(ax, headers, rows):
    ax.axis("off")
    ax.set_title(
        "Per run: ep mean = mean(eval returns), ep std = std(eval returns). "
        "Columns pool those run-level stats across runs.",
        fontsize=8,
        pad=6,
    )
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.65)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("0.88")
            cell.set_text_props(fontweight="bold")


def _replay_resample_prob(run):
    """Sampling / replay resample probability from a row (NaN if missing)."""
    v = run.get("replay_resample_prob")
    if v is None or v == "":
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _prob_display(p):
    if np.isnan(p):
        return "?"
    if abs(p - (-1)) < 1e-9:
        return "replay_resample_prob = -1"
    return f"replay_resample_prob = {p:g}"


def _prob_short(p):
    """Compact label for tick text (e.g. -1, 0.5)."""
    if np.isnan(p):
        return "?"
    f = float(p)
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:g}"


def _group_runs_by_prob(runs):
    """Map probability value -> list of runs (only rows with a numeric prob)."""
    groups = {}
    for r in runs:
        p = _replay_resample_prob(r)
        if np.isnan(p):
            continue
        groups.setdefault(p, []).append(r)
    return groups


_MEDIAN_COLOR = "#d95f02"
_MEAN_COLOR = "#1a9850"
_WHISKER_COLOR = "0.35"


def _ylim_tight(y_lo, y_hi, padding_frac=0.008):
    """Minimal padding around [y_lo, y_hi] for easy numeric comparison."""
    r = float(y_hi) - float(y_lo)
    if r <= 0:
        r = max(abs(float(y_lo)) * 0.002, 1e-6)
    pad = max(r * padding_frac, r * 0.003)
    return float(y_lo) - pad, float(y_hi) + pad


def _ylim_zoom_data(vals, padding_frac=0.008):
    """Y limits from raw numeric values with tight padding."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    return _ylim_tight(float(arr.min()), float(arr.max()), padding_frac=padding_frac)


def _ylim_from_boxplot(bp, fallback_arr, padding_frac=0.008):
    """Y limits from drawn boxplot geometry (whiskers, box, caps, medians, means, fliers)."""
    ys = []
    for w in bp.get("whiskers") or []:
        ys.extend(np.asarray(w.get_ydata(), dtype=float).ravel().tolist())
    for cap in bp.get("caps") or []:
        ys.extend(np.asarray(cap.get_ydata(), dtype=float).ravel().tolist())
    for med in bp.get("medians") or []:
        ys.extend(np.asarray(med.get_ydata(), dtype=float).ravel().tolist())
    for box in bp.get("boxes") or []:
        path = box.get_path()
        if path is not None:
            ys.extend(path.vertices[:, 1].tolist())
    for m in bp.get("means") or []:
        ys.extend(np.asarray(m.get_ydata(), dtype=float).ravel().tolist())
    for f in bp.get("fliers") or []:
        ys.extend(np.asarray(f.get_ydata(), dtype=float).ravel().tolist())
    ys = [y for y in ys if np.isfinite(y)]
    if not ys:
        return _ylim_zoom_data(fallback_arr, padding_frac=padding_frac)
    return _ylim_tight(min(ys), max(ys), padding_frac=padding_frac)


def _style_y_axis(ax):
    """More tick marks / readable scale when values are close together."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=14, steps=[1, 2, 2.5, 5, 10], min_n_ticks=6))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _add_boxplot_legend(ax):
    h = [
        Patch(facecolor="0.85", edgecolor="0.4", linewidth=1, label="Box = IQR (25–75%)"),
        plt.Line2D([0], [0], color=_MEDIAN_COLOR, lw=2.8, solid_capstyle="butt", label="Median"),
        plt.Line2D([0], [0], color=_MEAN_COLOR, lw=2.5, solid_capstyle="butt", label="Mean"),
        plt.Line2D([0], [0], color=_WHISKER_COLOR, lw=1.8, label="Whiskers"),
        plt.Line2D([0], [0], color="0.35", marker="o", lw=0, markersize=7, label="Outliers (fliers)"),
    ]
    ax.legend(handles=h, loc="upper right", fontsize=8, framealpha=0.92, title="Boxplot")


def _boxplot_series(ax, series, prob_to_color, title, ylabel):
    """series: list of (replay_prob, 1d ndarray of eval episode returns)."""
    if not series:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No numeric values", ha="center", va="center", transform=ax.transAxes)
        return
    data, colors, tick_labels = [], [], []
    for p, ys in series:
        if ys.size == 0:
            continue
        data.append(ys)
        colors.append(prob_to_color[p])
        tick_labels.append(_prob_short(p))
    if not data:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No numeric values", ha="center", va="center", transform=ax.transAxes)
        return
    positions = np.arange(1, len(data) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops={"color": _MEDIAN_COLOR, "linewidth": 2.5},
        meanprops={"color": _MEAN_COLOR, "linewidth": 2.2, "linestyle": "-"},
        whiskerprops={"color": _WHISKER_COLOR, "linewidth": 1.5},
        capprops={"color": _WHISKER_COLOR, "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.9, "markeredgecolor": "0.25"},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("replay_resample_prob")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(see legend: each point = one eval episode)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fallback = np.concatenate(data)
    lo, hi = _ylim_from_boxplot(bp, fallback)
    ax.set_ylim(lo, hi)
    _style_y_axis(ax)
    if lo <= 0 <= hi:
        ax.axhline(0, color="gray", linewidth=0.5)
    _add_boxplot_legend(ax)


def _boxplot_flat_array(ax, ys, title, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_xlabel("run")
    if ys.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No numeric values", ha="center", va="center", transform=ax.transAxes)
        return
    bp = ax.boxplot(
        [ys],
        positions=[1],
        widths=0.45,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops={"color": _MEDIAN_COLOR, "linewidth": 2.5},
        meanprops={"color": _MEAN_COLOR, "linewidth": 2.2, "linestyle": "-"},
        whiskerprops={"color": _WHISKER_COLOR, "linewidth": 1.5},
        capprops={"color": _WHISKER_COLOR, "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.9, "markeredgecolor": "0.25"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("C0")
        patch.set_alpha(0.75)
    ax.set_xticks([1])
    ax.set_xticklabels(["(all runs)"])
    ax.set_title(title + "\n(see legend: each point = one eval episode)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    lo, hi = _ylim_from_boxplot(bp, ys)
    ax.set_ylim(lo, hi)
    _style_y_axis(ax)
    if lo <= 0 <= hi:
        ax.axhline(0, color="gray", linewidth=0.5)
    _add_boxplot_legend(ax)


def _sync_eval_ylim(ax_box, ax_freq):
    """Match y-range across eval box and frequency panels for comparison."""
    lo = min(ax_box.get_ylim()[0], ax_freq.get_ylim()[0])
    hi = max(ax_box.get_ylim()[1], ax_freq.get_ylim()[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    ax_box.set_ylim(lo, hi)
    ax_freq.set_ylim(lo, hi)


def _frequency_dotplot_series(ax, series, prob_to_color, title, ylabel):
    """Stacked dots per bin; horizontal spacing shrinks when stacks are deep; marker size scales with N."""
    if not series:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No numeric values", ha="center", va="center", transform=ax.transAxes)
        return

    all_y = np.concatenate([s[1] for s in series])
    n_bins = int(np.clip(np.round(len(all_y) ** 0.38), 7, 22))
    edges = np.histogram_bin_edges(all_y, bins=max(5, min(n_bins, len(all_y))))
    nb = len(edges) - 1

    max_stack = _max_horizontal_stack(series, edges)
    stack_dx = min(0.11, 0.42 / max(1, max_stack))

    xs_all, ys_all, cs_all = [], [], []
    for i, (p, vals) in enumerate(series):
        pos = float(i + 1)
        bin_ids = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, nb - 1)
        by_bin = {}
        for yi, b in zip(vals, bin_ids):
            by_bin.setdefault(int(b), []).append(float(yi))
        for b in sorted(by_bin.keys()):
            vs = sorted(by_bin[b])
            c = len(vs)
            for k, y in enumerate(vs):
                xs_all.append(pos + (k - (c - 1) / 2.0) * stack_dx)
                ys_all.append(y)
                cs_all.append(prob_to_color[p])

    npt = len(ys_all)
    ms = max(9, min(34, int(22000 / max(1, npt))))
    ax.scatter(
        xs_all,
        ys_all,
        c=cs_all,
        s=ms,
        alpha=0.78,
        edgecolors="0.35",
        linewidths=0.12,
        zorder=3,
    )
    ax.set_xticks(np.arange(1, len(series) + 1))
    ax.set_xticklabels([_prob_short(p) for p, _ in series])
    ax.set_xlabel("replay_resample_prob")
    ax.set_ylabel(ylabel)
    ax.set_title(
        title
        + f"\n(same y-bin → horizontal stack; max stack={max_stack}, spacing={stack_dx:.3f}; N={npt} episodes)",
        fontsize=10,
    )
    ax.margins(x=0.12)
    ax.grid(True, axis="y", alpha=0.35)
    lo, hi = _ylim_zoom_data(np.array(ys_all, dtype=float))
    ax.set_ylim(lo, hi)
    _style_y_axis(ax)
    if lo <= 0 <= hi:
        ax.axhline(0, color="gray", linewidth=0.5)
    leg_dot = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="0.45", markersize=8, markeredgecolor="0.25", lw=0, label="One dot = one eval episode"),
        plt.Line2D([0], [0], color="0.45", lw=8, solid_capstyle="butt", label="Stack = same return bin"),
    ]
    ax.legend(handles=leg_dot, loc="upper right", fontsize=8, framealpha=0.92, title="Frequency dot plot")


def _frequency_dotplot_flat_array(ax, ys, title, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_xlabel("run")
    if ys.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No numeric values", ha="center", va="center", transform=ax.transAxes)
        return
    n_bins = int(np.clip(np.round(len(ys) ** 0.38), 7, 22))
    edges = np.histogram_bin_edges(ys, bins=max(5, min(n_bins, len(ys))))
    nb = len(edges) - 1
    flat_series = [(None, ys)]
    max_stack = _max_horizontal_stack(flat_series, edges)
    stack_dx = min(0.11, 0.42 / max(1, max_stack))

    bin_ids = np.clip(np.searchsorted(edges, ys, side="right") - 1, 0, nb - 1)
    by_bin = {}
    for yi, b in zip(ys, bin_ids):
        by_bin.setdefault(int(b), []).append(float(yi))
    xs_all, ys_all = [], []
    pos = 1.0
    for b in sorted(by_bin.keys()):
        vs = sorted(by_bin[b])
        c = len(vs)
        for k, y in enumerate(vs):
            xs_all.append(pos + (k - (c - 1) / 2.0) * stack_dx)
            ys_all.append(y)
    npt = len(ys_all)
    ms = max(9, min(34, int(22000 / max(1, npt))))
    ax.scatter(xs_all, ys_all, c="C0", s=ms, alpha=0.78, edgecolors="0.35", linewidths=0.12, zorder=3)
    ax.set_xticks([1])
    ax.set_xticklabels(["(all runs)"])
    ax.set_title(
        title + f"\n(same y-bin → horizontal stack; max stack={max_stack}, spacing={stack_dx:.3f}; N={npt} episodes)",
        fontsize=10,
    )
    ax.margins(x=0.12)
    ax.grid(True, axis="y", alpha=0.35)
    lo, hi = _ylim_zoom_data(np.array(ys_all, dtype=float))
    ax.set_ylim(lo, hi)
    _style_y_axis(ax)
    if lo <= 0 <= hi:
        ax.axhline(0, color="gray", linewidth=0.5)
    leg_dot = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=8, markeredgecolor="0.25", lw=0, label="One dot = one eval episode"),
        plt.Line2D([0], [0], color="C0", lw=8, solid_capstyle="butt", label="Stack = same return bin"),
    ]
    ax.legend(handles=leg_dot, loc="upper right", fontsize=8, framealpha=0.92, title="Frequency dot plot")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize eval (box + frequency) from CSV (default: tempresults.csv)."
    )
    parser.add_argument("-f", "--file", default=None, help="Path to csv (default: Work/policy/tempresults.csv)")
    args = parser.parse_args()

    if args.file:
        path = args.file
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "tempresults.csv")

    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        runs = load_runs(path)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    if not runs:
        print("No data rows in CSV.")
        sys.exit(0)

    by_prob = _group_runs_by_prob(runs)
    prob_levels = sorted(by_prob.keys(), key=lambda p: (p != p, float(p)))  # NaN-safe sort
    prob_to_color = {p: f"C{i}" for i, p in enumerate(prob_levels)} if prob_levels else {}

    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.52], hspace=0.42, wspace=0.34, top=0.88)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_freq = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[1, :])

    if prob_levels:
        series = _build_eval_series_by_prob(prob_levels, by_prob)
        _boxplot_series(
            ax_box,
            series,
            prob_to_color,
            "Eval — boxplot (every eval episode)",
            "Return",
        )
        _frequency_dotplot_series(
            ax_freq,
            series,
            prob_to_color,
            "Eval — frequency dot (every eval episode)",
            "Return",
        )
        _sync_eval_ylim(ax_box, ax_freq)
        hdr, tbl_rows = _stats_table_rows(prob_levels, by_prob, runs)
    else:
        ys_all = _build_eval_series_flat(runs)
        _boxplot_flat_array(ax_box, ys_all, "Eval — boxplot (every eval episode)", "Return")
        _frequency_dotplot_flat_array(ax_freq, ys_all, "Eval — frequency dot (every eval episode)", "Return")
        _sync_eval_ylim(ax_box, ax_freq)
        hdr, tbl_rows = _stats_table_rows([], {}, runs)

    _draw_inter_intra_table(ax_tbl, hdr, tbl_rows)

    if prob_levels:
        leg = [
            plt.Line2D([0], [0], color=prob_to_color[p], lw=10, solid_capstyle="butt", label=_prob_display(p))
            for p in prob_levels
        ]
        fig.legend(
            handles=leg,
            title="replay_resample_prob",
            loc="lower center",
            ncol=min(len(prob_levels), 4),
            bbox_to_anchor=(0.5, 0.99),
            fontsize="small",
        )
    fig.subplots_adjust(bottom=0.08, top=0.82 if prob_levels else 0.88)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(os.path.dirname(path), f"{base}_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
