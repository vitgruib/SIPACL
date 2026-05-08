#!/usr/bin/env python3
"""
Matched Pairs Analysis — eval return comparison between two PLR conditions.

Runs a full paired statistical test (paired t-test, Wilcoxon signed-rank,
Cohen's d, binomial sign test) on per-seed eval means read from runs_results.csv,
then plots a 2×2 diagnostic figure.

═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION  (edit the block below, then run)
═══════════════════════════════════════════════════════════════════════════════

    P_BASE     — the condition treated as baseline  (x-axis, left bar)
    P_TREATMENT— the condition treated as treatment  (y-axis, right bar)
    ROW_START  — first CSV row to include (1-indexed; row 1 = header row)
    ROW_END    — last  CSV row to include (1-indexed, inclusive)
                 Set to None to read all rows after ROW_START.

Default ROW_START=2 / ROW_END=61 reads the first 60 data rows (seeds 1–30
for each of the two 30-seed batches, assuming one run per seed per condition).

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════

    python Work/policy/compare_pval.py

Run from anywhere in the repo — all paths resolve relative to this file.
No CLI arguments. No files written. A window opens; close it to exit.

═══════════════════════════════════════════════════════════════════════════════
WHAT IT READS
═══════════════════════════════════════════════════════════════════════════════

    Work/policy/runs_results.csv   (appended automatically by ppo.py)

    Only rows with replay_resample_prob matching P_BASE or P_TREATMENT are
    used.  Rows outside [ROW_START, ROW_END] are ignored before any matching.
    If the same (seed, p) pair appears more than once in the selected rows,
    the last occurrence wins (later row = more recent run).
    A seed contributes only if it has BOTH conditions present.

═══════════════════════════════════════════════════════════════════════════════
WHAT IT SHOWS
═══════════════════════════════════════════════════════════════════════════════

    [A] Bar chart     — mean eval return per condition, ±2·SEM error bars,
                        individual seed means overlaid as jittered dots.

    [B] Difference    — per-seed (treatment − baseline), x-sorted by baseline
                        performance low→high.  Shaded band = 95% CI of the
                        mean difference; excludes 0 ⟺ paired t significant.

    [C] Scatter       — x = baseline eval mean, y = treatment eval mean.
                        Above diagonal = treatment won on that seed.

    [D] Win rate      — fraction of seeds the treatment won, with 95% Wilson CI.
                        Dashed line at 0.5 = chance under H₀.

    Suptitle: paired-t p, Wilcoxon p, Cohen's d, mean difference, n pairs.
"""
import ast
import os
import sys

import numpy as np
from scipy import stats

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
from run_results_io import load_runs

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
except ImportError:
    sys.exit("Install matplotlib: pip install matplotlib")

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION — edit these four lines
# ═══════════════════════════════════════════════════════════════════════════════

CSV         = os.path.join(_DIR, "runs_results.csv")
P_BASE      = -1.0   # baseline condition  (replay_resample_prob value)
P_TREATMENT =  0.75  # treatment condition (replay_resample_prob value)
ROW_START   =  2     # first CSV row to read (1-indexed; row 1 = header)
ROW_END     = 61     # last  CSV row to read (1-indexed, inclusive); None = no limit

# ═══════════════════════════════════════════════════════════════════════════════

# Human-readable labels derived from the p values above.
# :g strips trailing zeros  →  -1.0 → "-1",  0.75 → "0.75",  0.5 → "0.5"
_LBL_BASE = f"p={P_BASE:g}"
_LBL_TRT  = f"p={P_TREATMENT:g}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _parse_list(cell):
    if not cell or (isinstance(cell, str) and not cell.strip()):
        return []
    try:
        return [float(x) for x in ast.literal_eval(str(cell).strip())]
    except Exception:
        return []


def _eval_mean(run):
    # Prefer the raw list of per-episode eval returns (more precise than the
    # stored scalar mean, which may have been rounded or computed differently).
    vals = _parse_list(run.get("eval_returns"))
    if vals:
        return float(np.mean(vals))
    return _num(run.get("mean_eval_return"))


def _style_y(ax):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _style_x(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Step 0: load and slice CSV rows ───────────────────────────────────────
    # ROW_START and ROW_END are 1-indexed CSV row numbers where row 1 = header.
    # load_runs() returns data rows only (header stripped), so CSV row k
    # corresponds to data index k-2  (row 2 → index 0, row 3 → index 1, ...).
    #
    # Conversion:
    #   data start index = ROW_START - 2
    #   data end   index = ROW_END - 1   (ROW_END is inclusive → exclusive slice end)
    #
    # Example with defaults ROW_START=2, ROW_END=61:
    #   start = 0, end = 60  →  runs[0:60]  →  first 60 data rows
    all_runs = load_runs(CSV)
    if not all_runs:
        sys.exit(f"No data found in {CSV}")

    start_idx = max(0, ROW_START - 2)
    end_idx   = (ROW_END - 1) if ROW_END is not None else len(all_runs)
    runs = all_runs[start_idx:end_idx]

    if not runs:
        sys.exit(
            f"No rows remain after applying ROW_START={ROW_START}, ROW_END={ROW_END}. "
            f"CSV has {len(all_runs)} data rows (rows 2–{len(all_runs)+1})."
        )

    # ── Step 1: collect matched seed pairs ────────────────────────────────────
    # For each (seed, p) combination keep the most-recent row only.
    # The CSV may contain re-runs of the same seed; the last row in the selected
    # window wins because we overwrite the dict entry.
    by_seed: dict = {}
    for r in runs:
        p = _num(r.get("replay_resample_prob"))
        s_raw = r.get("seed")
        if not s_raw:
            continue
        try:
            s = int(float(s_raw))
        except (TypeError, ValueError):
            continue
        # Only keep rows that belong to one of the two conditions being compared.
        if abs(p - P_BASE) < 1e-9 or abs(p - P_TREATMENT) < 1e-9:
            by_seed.setdefault(s, {})[p] = r   # later row silently overwrites

    # A seed "matches" only when it has a run under BOTH conditions.
    # Sorting makes the order stable and reproducible.
    matched = sorted(s for s, d in by_seed.items() if P_BASE in d and P_TREATMENT in d)
    if len(matched) < 2:
        sys.exit(
            f"Not enough matched seed pairs found for {_LBL_BASE} vs {_LBL_TRT}. "
            f"Check that both p values appear in the selected CSV rows."
        )

    # ── Step 2: build per-seed mean-return arrays ─────────────────────────────
    # base_means[i]  = eval mean return for seed matched[i] under P_BASE
    # treat_means[i] = eval mean return for seed matched[i] under P_TREATMENT
    # Arrays are ALIGNED: index i refers to the same seed in both.
    base_means  = np.array([_eval_mean(by_seed[s][P_BASE])      for s in matched])
    treat_means = np.array([_eval_mean(by_seed[s][P_TREATMENT])  for s in matched])

    # diffs[i] = treatment advantage for seed i.  Positive = treatment better.
    diffs = treat_means - base_means

    n = len(matched)   # number of matched pairs

    # ── Step 3: paired statistics ─────────────────────────────────────────────

    # --- Paired t-test ---
    # H0: mean(diffs) = 0.
    # We test the per-seed difference directly because seeds are matched 1-to-1.
    # Treating them as independent samples would discard the within-seed
    # correlation and reduce statistical power.
    # ttest_rel(a, b) tests (a - b), so passing base first means a positive
    # t-statistic = base > treatment.  The p-value is two-sided either way.
    t_stat, p_val = stats.ttest_rel(base_means, treat_means)

    # --- Mean difference and its 95% CI ---
    # mean_diff: how much better the treatment is on average across seeds.
    # 95% CI uses the t-distribution with df=n-1 (not z=1.96) because n is
    # finite and std_diff is estimated from the same sample.
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1))   # ddof=1 → sample std
    sem_diff  = std_diff / np.sqrt(n)
    t_crit    = stats.t.ppf(0.975, df=n - 1)   # 97.5th percentile of t_{n-1}
    ci_lo     = mean_diff - t_crit * sem_diff
    ci_hi     = mean_diff + t_crit * sem_diff
    # If [ci_lo, ci_hi] excludes 0, the paired t-test is significant at α=0.05.

    # --- Wilcoxon signed-rank test ---
    # Non-parametric alternative to the paired t-test; does not assume normality.
    # H0: median(diffs) = 0.  Use alongside t-test to check robustness.
    w_stat, p_wil = stats.wilcoxon(diffs)

    # --- Cohen's d (paired) ---
    # Effect size = mean_diff / std_diff  (SD of the paired differences).
    # Benchmarks: 0.2 = small, 0.5 = medium, 0.8 = large.
    # Note: this uses the SD of differences, NOT the pooled within-group SD.
    cohens_d = mean_diff / (std_diff + 1e-12)

    # ── Step 4: win-rate and sign test ────────────────────────────────────────

    # "Win" = seed where treatment's eval mean exceeded baseline's eval mean.
    n_wins_trt  = int((diffs > 0).sum())
    n_wins_base = int((diffs < 0).sum())
    # n_wins_trt + n_wins_base ≤ n  (ties, if any, count for neither).

    # Binomial sign test: H0 = each seed is equally likely to go either way.
    # Two-sided; does not use the magnitude of differences, only their sign.
    binom_res = stats.binomtest(n_wins_trt, n, 0.5)
    binom_p   = binom_res.pvalue

    # --- Wilson 95% CI for the treatment win proportion ---
    # Wilson CI (Agresti & Coull, 1998) is preferred over the Wald interval
    # (prop ± z·sqrt(p(1-p)/n)) because Wald can produce values outside [0,1]
    # when the proportion is near 0 or 1, which is possible here.
    # Formulae:
    #   denom  = 1 + z² / n
    #   centre = (prop + z²/(2n)) / denom
    #   half   = z · sqrt(prop·(1-prop)/n + z²/(4n²)) / denom
    prop   = n_wins_trt / n
    z_b    = 1.96
    denom  = 1 + z_b**2 / n
    centre = (prop + z_b**2 / (2 * n)) / denom
    half   = z_b * np.sqrt(prop * (1 - prop) / n + z_b**2 / (4 * n**2)) / denom
    win_ci_lo = centre - half
    win_ci_hi = centre + half

    # ── Step 5: per-group SEM for Panel A ────────────────────────────────────
    # Each bar = mean of the n per-seed means for that condition.
    # Error bars = 2·SEM of those n values.
    # 2·SEM ≈ 95% CI for the group mean (exact would be t_{n-1}·SEM, close
    # enough for a bar chart at n≥20).
    sem_base  = float(np.std(base_means,  ddof=1) / np.sqrt(n))
    sem_treat = float(np.std(treat_means, ddof=1) / np.sqrt(n))

    # ── Figure ────────────────────────────────────────────────────────────────
    col_base  = "C0"        # blue   — baseline
    col_treat = "C1"        # orange — treatment
    col_win   = "#2ca02c"   # green  — treatment won this seed
    col_lose  = "#d62728"   # red    — baseline won this seed

    fig = plt.figure(figsize=(14, 11))
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    ax_bar     = fig.add_subplot(gs[0, 0])   # top-left
    ax_diff    = fig.add_subplot(gs[0, 1])   # top-right
    ax_scatter = fig.add_subplot(gs[1, 0])   # bottom-left
    ax_win     = fig.add_subplot(gs[1, 1])   # bottom-right

    # =========================================================================
    # Panel A — Mean eval return ± 2*SEM  (top-left)
    # =========================================================================
    # Each bar = mean across all n seeds for that condition.
    # Error bars = 2·SEM of the n seed-means.
    # Jittered dots show individual seed means so you can see the distribution
    # and check that the bars reflect the spread honestly.

    bar_pos  = [0, 1]
    bar_vals = [float(np.mean(base_means)), float(np.mean(treat_means))]
    bar_errs = [2 * sem_base, 2 * sem_treat]

    bars = ax_bar.bar(
        bar_pos, bar_vals,
        yerr=bar_errs,
        color=[col_base, col_treat],
        alpha=0.78, width=0.5,
        ecolor="0.2", capsize=8,
        error_kw={"linewidth": 2.0},
        zorder=2,
    )

    # Jitter seed=1 makes the dot positions reproducible across runs.
    rng = np.random.default_rng(1)
    for xi, vals, col in [(0, base_means, col_base), (1, treat_means, col_treat)]:
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax_bar.scatter(
            np.full(len(vals), xi) + jitter, vals,
            color=col, s=22, alpha=0.55,
            edgecolors="0.3", linewidths=0.4, zorder=3,
        )

    # Numeric mean label above each error bar cap.
    for bar, v, e in zip(bars, bar_vals, bar_errs):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            v + e + max(bar_vals) * 0.01,
            f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax_bar.set_xticks(bar_pos)
    ax_bar.set_xticklabels([_LBL_BASE + "\n(baseline)", _LBL_TRT + "\n(treatment)"], fontsize=10)
    ax_bar.set_ylabel("Mean eval return", fontsize=9)
    ax_bar.set_title(
        "(A)  Mean eval return\nbars = mean across seeds,  error = ±2·SEM,  dots = seeds",
        fontsize=9,
    )
    ax_bar.grid(True, axis="y", alpha=0.25, zorder=0)
    _style_y(ax_bar)
    lo_b = min(base_means.min(), treat_means.min()) * 0.92
    hi_b = (max(bar_vals) + max(bar_errs)) * 1.12
    ax_bar.set_ylim(lo_b, hi_b)

    # =========================================================================
    # Panel B — Per-seed differences + 95% CI band  (top-right)
    # =========================================================================
    # Each dot = one seed's (treatment − baseline) difference.
    # x-axis: seeds ranked by their baseline eval mean, low → high.
    #   Ordering is cosmetic — it spreads points and reveals trends, but implies
    #   no causal relationship between baseline performance and the difference.
    # y-axis: diffs[i] = treat_means[i] − base_means[i].  Positive = treatment better.
    # Shaded band: 95% CI of mean_diff from the paired t-distribution.
    #   If the band excludes y=0, the paired t-test is significant at α=0.05
    #   (the CI and the t-test are mathematically equivalent).

    order    = np.argsort(base_means)   # sort indices: lowest baseline first
    d_sorted = diffs[order]
    seed_pos = np.arange(n)

    dot_colors = [col_win if d > 0 else col_lose for d in d_sorted]

    ax_diff.axhline(0, color="0.4", linewidth=0.8, linestyle="--", zorder=1)

    # 95% CI band: mean_diff ± t_{0.975, n-1} · sem_diff  (computed in Step 3).
    ax_diff.axhspan(ci_lo, ci_hi, color="steelblue", alpha=0.15, zorder=2)
    ax_diff.axhline(mean_diff, color="steelblue", linewidth=2.2, zorder=3,
                    label=f"Mean diff = {mean_diff:.2f}")
    ax_diff.axhline(ci_lo, color="steelblue", linewidth=0.9, linestyle=":", zorder=3)
    ax_diff.axhline(ci_hi, color="steelblue", linewidth=0.9, linestyle=":", zorder=3)

    ax_diff.scatter(seed_pos, d_sorted, c=dot_colors, s=50, zorder=4,
                    edgecolors="0.2", linewidths=0.5)

    ax_diff.set_xlabel(f"Seeds ranked by {_LBL_BASE} eval mean (low → high)", fontsize=9)
    ax_diff.set_ylabel(f"Diff  ({_LBL_TRT} − {_LBL_BASE})", fontsize=9)
    ax_diff.set_title(
        f"(B)  Per-seed difference + 95% CI\n"
        f"mean={mean_diff:.2f}   95% CI [{ci_lo:.2f}, {ci_hi:.2f}]   "
        f"paired-t p={p_val:.3f}   d={cohens_d:.2f}",
        fontsize=9,
    )
    ax_diff.grid(True, axis="y", alpha=0.25)
    _style_y(ax_diff)
    lo_d = min(d_sorted.min(), ci_lo) * 1.15
    hi_d = max(d_sorted.max(), ci_hi) * 1.15
    ax_diff.set_ylim(lo_d, hi_d)
    legend_h = [
        mpatches.Patch(color=col_win,                label=f"Treatment wins  ({(d_sorted > 0).sum()})"),
        mpatches.Patch(color=col_lose,               label=f"Baseline wins   ({(d_sorted < 0).sum()})"),
        mpatches.Patch(color="steelblue", alpha=0.4, label="95% CI of mean diff"),
    ]
    ax_diff.legend(handles=legend_h, fontsize=8, loc="upper left", framealpha=0.9)

    # =========================================================================
    # Panel C — Head-to-head scatter  (bottom-left)
    # =========================================================================
    # x = baseline eval mean for a seed; y = treatment eval mean for the same seed.
    # Diagonal (y = x): equal performance.  Above = treatment won.
    # Colour encodes winner identically to Panel B (green/red).
    # This panel reveals whether the advantage is consistent across seeds or
    # driven by a few outliers, and whether the two conditions correlate.

    lims = [
        min(base_means.min(), treat_means.min()) * 0.96,
        max(base_means.max(), treat_means.max()) * 1.04,
    ]
    ax_scatter.plot(lims, lims, color="0.55", linewidth=1.0,
                    linestyle="--", zorder=1, label="y = x  (equal)")

    win_mask  = diffs > 0
    loss_mask = ~win_mask
    ax_scatter.scatter(
        base_means[win_mask], treat_means[win_mask],
        color=col_win, s=55, edgecolors="0.2", linewidths=0.5,
        zorder=3, label=f"Treatment wins  ({win_mask.sum()})",
    )
    ax_scatter.scatter(
        base_means[loss_mask], treat_means[loss_mask],
        color=col_lose, s=55, edgecolors="0.2", linewidths=0.5,
        zorder=3, label=f"Baseline wins   ({loss_mask.sum()})",
    )

    ax_scatter.set_xlabel(f"{_LBL_BASE}  eval mean return", fontsize=9)
    ax_scatter.set_ylabel(f"{_LBL_TRT}  eval mean return", fontsize=9)
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_title(
        f"(C)  Head-to-head per seed\n(above diagonal = {_LBL_TRT} wins)",
        fontsize=9,
    )
    ax_scatter.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax_scatter.grid(True, alpha=0.22)
    _style_x(ax_scatter)
    _style_y(ax_scatter)

    # =========================================================================
    # Panel D — Win-rate bar + 95% Wilson CI  (bottom-right)
    # =========================================================================
    # Left bar:  proportion of seeds where treatment won (eval mean > baseline).
    # Right bar: proportion where baseline won.
    # (Ties, if any, are excluded from both bars so they may not sum to 1.)
    #
    # Error bar on left bar only: 95% Wilson CI for the treatment win proportion.
    # Wilson CI (rather than Wald) is used because Wald can produce values
    # outside [0,1] when proportions are near the boundaries.
    #
    # Dashed line at 0.5: chance level under H₀ (each seed equally likely to
    # go either way).  If the Wilson CI excludes 0.5, the sign test is significant.

    proportions = [n_wins_trt / n, n_wins_base / n]
    bar_x = [0, 1]
    bars_win = ax_win.bar(
        bar_x, proportions,
        color=[col_win, col_lose],
        alpha=0.78, width=0.5,
        edgecolor="0.2", linewidth=0.8,
    )

    # errorbar yerr format: [[lower_distance], [upper_distance]] from the bar top.
    # lower_distance = prop - win_ci_lo  (how far the CI extends below prop)
    # upper_distance = win_ci_hi - prop  (how far the CI extends above prop)
    ax_win.errorbar(
        0, prop,
        yerr=[[prop - win_ci_lo], [win_ci_hi - prop]],
        fmt="none", color="0.15", capsize=7, linewidth=1.8, zorder=5,
    )

    ax_win.axhline(0.5, color="0.45", linewidth=1.0, linestyle="--", label="chance (0.5)")

    for bar, pct in zip(bars_win, proportions):
        ax_win.text(
            bar.get_x() + bar.get_width() / 2,
            pct + 0.02, f"{pct:.0%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax_win.set_xticks(bar_x)
    ax_win.set_xticklabels([f"{_LBL_TRT} wins", f"{_LBL_BASE} wins"], fontsize=9)
    ax_win.set_ylabel("Proportion of seeds", fontsize=9)
    ax_win.set_ylim(0, 1.0)
    ax_win.set_title(
        f"(D)  Seed-level win rate\n"
        f"{n_wins_trt}/{n} seeds   "
        f"Wilson 95% CI [{win_ci_lo:.2f}, {win_ci_hi:.2f}]   "
        f"sign-test p={binom_p:.3f}",
        fontsize=9,
    )
    # Horizontal lines mark the Wilson CI bounds for readability.
    ax_win.axhline(win_ci_lo, color=col_win, linewidth=0.8, linestyle=":",
                   label=f"Wilson CI [{win_ci_lo:.2f}, {win_ci_hi:.2f}]")
    ax_win.axhline(win_ci_hi, color=col_win, linewidth=0.8, linestyle=":")
    ax_win.axhspan(win_ci_lo, win_ci_hi, color=col_win, alpha=0.08)
    ax_win.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax_win.grid(True, axis="y", alpha=0.25)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    # All headline numbers in one line so the figure is self-contained.
    row_desc = f"CSV rows {ROW_START}–{ROW_END if ROW_END is not None else 'end'}"
    fig.suptitle(
        f"Matched Pairs Analysis:  {_LBL_TRT} (treatment)  vs  {_LBL_BASE} (baseline)"
        f"   |   n={n} matched seed pairs   |   {row_desc}\n"
        f"Paired t: p={p_val:.3f}   Wilcoxon: p={p_wil:.3f}   "
        f"Cohen's d={cohens_d:.2f}   mean diff={mean_diff:.2f}   "
        f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    plt.show()


if __name__ == "__main__":
    main()
