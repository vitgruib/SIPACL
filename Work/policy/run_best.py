"""Find the best run in runs_results.csv and launch ppo.py to evaluate its saved model.

"Best" = highest mean_eval_return among rows that have a real mean_eval_return
(not NaN/blank) and a matching checkpoint in Work/runs/<run_name>.pt.

Usage:
    python policy/run_best.py [-- extra ppo.py args]

Any args after `--` (or unrecognized by this script) are forwarded verbatim to
ppo.py, so you can override things like eval episodes or rendering, e.g.:

    python policy/run_best.py --eval-episodes 10 --render False
"""
import csv
import math
import os
import subprocess
import sys

_policy_dir = os.path.dirname(os.path.abspath(__file__))
_work_dir = os.path.dirname(_policy_dir)
RESULTS_CSV = os.path.join(_policy_dir, "runs_results.csv")
RUNS_DIR = os.path.join(_work_dir, "runs")
PPO_PY = os.path.join(_policy_dir, "ppo.py")


def _parse_float(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def find_best_run():
    with open(RESULTS_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    best = None
    for row in rows:
        mer = _parse_float(row.get("mean_eval_return"))
        if mer is None:
            continue
        model_path = os.path.join(RUNS_DIR, row["run_name"] + ".pt")
        if not os.path.exists(model_path):
            continue
        if best is None or mer > best[0]:
            best = (mer, row, model_path)

    if best is None:
        raise SystemExit(
            f"No run in {RESULTS_CSV} has both a mean_eval_return and a saved "
            f"checkpoint under {RUNS_DIR}/. Nothing to evaluate."
        )
    return best


if __name__ == "__main__":
    mean_eval_return, row, model_path = find_best_run()
    print(
        f"Best run: {row['run_name']} "
        f"(seed={row['seed']}, replay_resample_prob={row['replay_resample_prob']}, "
        f"mean_eval_return={mean_eval_return:.4f} over {row['num_eval_episodes']} eval episodes)"
    )
    print(f"Model: {model_path}")

    extra_args = sys.argv[1:]
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    cmd = [
        sys.executable,
        PPO_PY,
        "--evaluate-model",
        "--model-to-evaluate-path",
        model_path,
        *extra_args,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=_work_dir, check=True)
