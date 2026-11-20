#!/usr/bin/env python3
"""
Compare the runs of a grid by their validation history: the best checkpoint per wait budget.

Reads <sessions>/<curriculum>_s<seed>_test<year>/models/*/validation.jsonl (as written by
train.py --val-every; a copy with only these files is enough) and prints, for each run and each
wait budget (average wait at the worst arrival scale), the feasible checkpoint with the highest
savings vs baseline_off. Feasible = loses no more jobs than the baseline (see src/validation.py).
Then a summary per curriculum (mean ± std over seeds and held-out years), and a plot of savings
against wait for every feasible checkpoint.

Validation runs on the training years; the held-out year is only in the final evaluation.

Example:
  python analyze_validation.py --sessions sessions/v2_cluster
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import numpy as np

from src.validation import (best_per_budget, entry_feasible, entry_wait, entry_wait_tail_metrics,
                            format_wait_tail, last_stage_start, load_history)

RUN_NAME = re.compile(r"^(?P<curriculum>.+)_s(?P<seed>\d+)_test(?P<year>\d{4})$")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5, fixed order
MARKERS = ["o", "s", "^", "D", "v"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", default="sessions/v2", help="Directory holding the run sessions (default: sessions/v2).")
    parser.add_argument("--budgets", default="6,12,24", help="Wait budgets in hours, comma-separated (default: 6,12,24; 'any' is always added).")
    parser.add_argument("--out-dir", default="", help="Where to write validation_best.csv and validation_frontier.png (default: --sessions).")
    args = parser.parse_args()
    budgets = tuple(float(b) for b in args.budgets.split(",") if b.strip()) + (None,)
    out_dir = args.out_dir or args.sessions

    runs = []
    for path in sorted(glob.glob(os.path.join(args.sessions, "*", "models", "*", "validation.jsonl"))):
        name = path.split(os.sep)[-4]
        m = RUN_NAME.match(name)
        if not m:
            continue
        history = load_history(os.path.dirname(path))
        if history:
            after = last_stage_start(os.path.join(args.sessions, name))
            runs.append(dict(name=name, **m.groupdict(), history=history,
                             best=best_per_budget(history, budgets, after_step=after)))
    if not runs:
        raise SystemExit(f"no validation.jsonl under {args.sessions}/*/models/*/")
    labels = list(runs[0]["best"].keys())
    curricula = list(dict.fromkeys(r["curriculum"] for r in sorted(runs, key=lambda r: r["curriculum"])))

    # Per run
    print("Best feasible checkpoint per wait budget (savings vs baseline_off %, mean over scales; "
          "all wait metrics at the scale with the highest mean wait)\n")
    head = f"{'run':22s} {'validated':>9s} " + " ".join(f"{l:>68s}" for l in labels)
    print(head)
    rows = []
    for r in sorted(runs, key=lambda r: (r["curriculum"], r["seed"], r["year"])):
        last = r["history"][-1]["step"]
        cells = []
        for l in labels:
            e = r["best"][l]
            p95, tail_mean = entry_wait_tail_metrics(e) if e is not None else (np.nan, np.nan)
            cells.append("—" if e is None else f"{e['savings_off_pct']:+5.1f}% mean {entry_wait(e):4.1f}h, "
                         f"{format_wait_tail(p95, tail_mean)} @{e['step'] / 1e6:.1f}M")
            rows.append(dict(run=r["name"], curriculum=r["curriculum"], seed=r["seed"], test_year=r["year"], budget=l,
                             step=None if e is None else e["step"], savings_off_pct=None if e is None else e["savings_off_pct"],
                             wait_h=None if e is None else entry_wait(e),
                             p95_wait_h=float(p95) if np.isfinite(p95) else None,
                             tail_mean_wait_h=float(tail_mean) if np.isfinite(tail_mean) else None,
                             last_validated_step=last))
        print(f"{r['name']:22s} {last / 1e6:8.1f}M " + " ".join(f"{c:>68s}" for c in cells))

    # Per curriculum
    print(f"\nPer curriculum: best savings within each budget, mean ± std over runs (runs with a feasible checkpoint / runs)\n")
    print(f"{'curriculum':12s} " + " ".join(f"{l:>20s}" for l in labels))
    for c in curricula:
        cells = []
        for l in labels:
            vals = [r["best"][l]["savings_off_pct"] for r in runs if r["curriculum"] == c and r["best"][l] is not None]
            n = sum(r["curriculum"] == c for r in runs)
            cells.append("—" if not vals else f"{np.mean(vals):+5.1f} ± {np.std(vals):4.1f} ({len(vals)}/{n})")
        print(f"{c:12s} " + " ".join(f"{x:>20s}" for x in cells))

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "validation_best.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Savings against wait, every feasible checkpoint; one color and marker per curriculum.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=130, sharey=True)
    wait_metrics = [("Mean wait (h)", entry_wait),
                    ("P95 wait (h)", lambda h: entry_wait_tail_metrics(h)[0]),
                    ("Slowest 5% mean wait (h)", lambda h: entry_wait_tail_metrics(h)[1])]
    for ax, (label, metric) in zip(axes, wait_metrics):
        finite = []
        for i, c in enumerate(curricula):
            pts = [(metric(h), h["savings_off_pct"]) for r in runs if r["curriculum"] == c
                   for h in r["history"] if entry_feasible(h) and np.isfinite(metric(h))]
            if pts:
                x, y = zip(*pts)
                finite.extend(x)
                ax.scatter(x, y, s=22, color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], alpha=0.75,
                           edgecolors="white", linewidths=0.6, label=f"{c} ({len(pts)} checkpoints)")
        ax.axhline(0, color="#5f5e58", linewidth=0.8, zorder=0)
        ax.set_xlabel(label)
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if finite:
            ax.set_xlim(0, max(max(finite) * 1.05, 1.0))
            ax.legend(frameon=False, fontsize=9, loc="lower right")
        else:
            ax.text(0.5, 0.5, "No recorded values", transform=ax.transAxes, ha="center")
    ax = axes[0]
    for b in budgets[:-1]:
        ax.axvline(b, color="#8a8a85", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
        ax.text(b, 1.0, f" {b:g} h", transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=8, color="#5f5e58")
    ax.set_ylabel("savings vs baseline_off (%), mean over scales")
    fig.suptitle("Validated checkpoints that lose no jobs (training-year prices)\n"
                 "Wait metrics at the scale with the highest mean wait", fontsize=11)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "validation_frontier.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"\nWritten {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
