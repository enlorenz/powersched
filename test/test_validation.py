"""Checks the checkpoint validation (src/validation.py): fixed arrivals, feasibility, best-checkpoint record.

Run with: python -m test.test_validation
"""

import contextlib
import csv
import io
import json
import math
import os
import tempfile
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.environment import ComputeClusterEnv, Weights
from src.plot_config import PlotConfig
from src.validation import (ValidationConfig, run_validation, record_result, load_history, window_starts,
                            best_per_budget, last_stage_start, entry_wait_tail_metrics, format_best,
                            format_result, BEST_FILE)
from src.workloadgen import WorkloadGenConfig, WorkloadGenerator

CONFIG = ValidationConfig(arrival_scales=(1.0,), windows=2, episodes_per_window=1, seed=5)


class FixedPolicy:
    """Stands in for a PPO model: always the same action."""
    def __init__(self, action):
        self.action = np.array(action)

    def predict(self, obs, deterministic=True):
        return self.action, None


def _make_env(arrival_scale):
    workload = WorkloadGenerator(WorkloadGenConfig(arrivals="poisson", poisson_lambda=200.0,
                                                   poisson_lambda_duration=6.0, poisson_lambda_nodes=2.0,
                                                   poisson_lambda_cores=24.0))
    with contextlib.redirect_stdout(io.StringIO()):
        env = ComputeClusterEnv(weights=Weights(0.4, 0.4, 0.0, 0.2, 0.0), session="test_validation",
                                render_mode="none", external_prices=None, external_durations=None,
                                external_jobs=None, external_hourly_jobs=None, plot_config=PlotConfig(),
                                steps_per_iteration=10**9, evaluation_mode=True, workload_gen=workload,
                                job_arrival_scale=arrival_scale, output_dir="test/test_output")
    return env


def _run(policy, envs):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_validation(policy, _make_env, CONFIG, envs)


def test_same_arrivals_for_every_policy_and_feasibility():
    envs = {}
    keep_on = _run(FixedPolicy([2, 8]), envs)     # switch on up to 256 nodes per hour: all on
    all_off = _run(FixedPolicy([0, 8]), envs)     # switch everything off: the queue fills, jobs are lost
    keep_on_again = _run(FixedPolicy([2, 8]), envs)
    a, b = keep_on["scales"]["1"], all_off["scales"]["1"]
    assert a["jobs_submitted"] == b["jobs_submitted"] > 0
    assert a["baseline_cost_off"] == b["baseline_cost_off"]
    assert a["baseline_jobs_completed"] == b["baseline_jobs_completed"]
    assert a["episodes"] == b["episodes"] == CONFIG.windows * CONFIG.episodes_per_window
    assert keep_on == keep_on_again  # reusing the envs changes nothing
    assert keep_on["feasible"] and keep_on["score"] is not None
    assert not all_off["feasible"] and all_off["score"] is None


def _result(savings, wait, lost=0, overdue=0, submitted=100000):
    """A validation result with one scale; lost > 0 loses more jobs than the baseline."""
    scale = dict(savings_off_pct=savings, avg_wait_h=wait, jobs_lost=lost, baseline_jobs_lost=0, completion_ratio=0.97,
                 overdue_jobs_end=overdue, jobs_submitted=submitted)
    return {"scales": {"1": scale}, "feasible": lost == 0, "savings_off_pct": savings, "score": savings if lost == 0 else None}


def test_record_result_keeps_best_per_wait_budget():
    with tempfile.TemporaryDirectory() as d:
        best = record_result(d, 500000, _result(30.0, 2.0, lost=5))   # loses jobs: never best
        assert all(e is None for e in best.values())
        assert not os.path.exists(os.path.join(d, BEST_FILE))
        record_result(d, 1000000, _result(5.0, 4.0))
        record_result(d, 1500000, _result(15.0, 10.0))
        best = record_result(d, 2000000, _result(25.0, 20.0))
        assert {k: (e["step"] if e else None) for k, e in best.items()} == \
            {"6h": 1000000, "12h": 1500000, "24h": 2000000, "any": 2000000}
        # A re-validated step replaces its entry: 2M now waits 30 h, so 1.5M is best within 24 h.
        best = record_result(d, 2000000, _result(25.0, 30.0))
        assert best["24h"]["step"] == 1500000 and best["any"]["step"] == 2000000
        assert [h["step"] for h in load_history(d)] == [500000, 1000000, 1500000, 2000000]
        with open(os.path.join(d, BEST_FILE)) as f:
            saved = json.load(f)
        assert saved["step"] == 2000000 and saved["by_wait_budget"]["6h"]["step"] == 1000000
        # The completion ratio is ignored unless asked for (all entries above have 0.97).
        assert best_per_budget(load_history(d), min_completion_ratio=0.99)["any"] is None


def test_negative_savings_are_never_picked():
    history = [
        {"step": 1, **_result(-20.0, 1.0)},                # costs more than the baseline
        {"step": 2, **_result(10.0, 9.0, overdue=2000)},   # overdue jobs at window ends do not make it infeasible
    ]
    best = best_per_budget(history)
    assert best["6h"] is None                            # only the negative-savings checkpoint waits < 6 h
    assert best["12h"]["step"] == 2 and best["any"]["step"] == 2


def test_only_last_stage_checkpoints_are_picked():
    history = [
        {"step": 1_000_000, **_result(26.0, 19.8, overdue=26675)},  # end of stage D: best score, never saw real prices
        {"step": 4_000_000, **_result(25.0, 8.0)},
        {"step": 8_000_000, **_result(22.9, 7.1)},
    ]
    assert best_per_budget(history)["any"]["step"] == 1_000_000
    best = best_per_budget(history, after_step=1_000_000)
    assert best["any"]["step"] == 4_000_000 and best["24h"]["step"] == 4_000_000
    with tempfile.TemporaryDirectory() as d:
        assert last_stage_start(d) == 0  # no curriculum.log
        with open(os.path.join(d, "curriculum.log"), "w") as f:
            f.write("[2026-11-10 12:52:00] start curriculum=de\n"
                    "[2026-11-10 12:52:01] stage stageD (to iteration 10): --prices \n"
                    "[2026-11-10 14:00:00] stage stageE (to iteration 80): --prices a.csv\n"
                    "[2026-11-11 12:52:00] stage stageD (to iteration 10): --prices \n"  # resumed run
                    "[2026-11-11 12:52:10] stage stageE (to iteration 80): --prices a.csv\n")
        assert last_stage_start(d) == 1_000_000
        with open(os.path.join(d, "curriculum.log"), "w") as f:
            f.write("[2026-11-10 12:52:01] stage stageE (to iteration 80): --prices a.csv\n")
        assert last_stage_start(d) == 0  # single stage: every checkpoint is a candidate


def test_window_starts_spread_evenly():
    assert window_starts(17520, 4) == [2190, 6570, 10950, 15330]
    assert window_starts(None, 2) == [0, 336]


def test_wait_tail_metrics_pool_episodes_across_reset_windows():
    config = ValidationConfig(arrival_scales=(1.0,), windows=2, episodes_per_window=2, seed=17)

    class HistogramEnv:
        def __init__(self):
            self.prices = SimpleNamespace(external_prices=None)
            self.metrics = SimpleNamespace(episode_costs=[], episode_job_wait_time_histogram=Counter(),
                                           job_wait_time_histogram=Counter({999: 1000}))

        def reset(self, seed=None, options=None):
            if options and options.get("hard_reset"):
                self.index = (seed - config.seed) * config.episodes_per_window
            else:
                self.index += 1
            self.metrics.episode_job_wait_time_histogram.clear()
            return {}, {}

        def step(self, action):
            histograms = ({0: 50}, {0: 50}, {10: 1}, {20: 1})
            histogram = histograms[self.index]
            self.metrics.episode_job_wait_time_histogram.update(histogram)
            count = sum(histogram.values())
            self.metrics.episode_costs.append(dict(
                agent_cost=1.0, baseline_cost_off=2.0, jobs_submitted=count, jobs_launched=count,
                jobs_completed=count, baseline_jobs_completed=count,
                avg_wait_time=sum(age * n for age, n in histogram.items()) / count,
            ))
            return {}, 0.0, True, False, {}

    envs = {1.0: HistogramEnv()}
    first = run_validation(FixedPolicy([0, 0]), None, config, envs)
    second = run_validation(FixedPolicy([0, 0]), None, config, envs)
    assert first == second  # Neither earlier validation calls nor hard-reset windows leak into the pool.
    scale = first["scales"]["1"]
    assert scale["episodes"] == 4 and scale["jobs_launched"] == 102
    assert scale["p95_wait_h"] == 0.0
    assert math.isclose(scale["tail_mean_wait_h"], 30.0 / 5.1)
    assert "wait 0.3h, p95 0.0h, slowest 5% mean 5.9h" in format_result(1, first)


def test_entry_tail_metrics_use_the_scale_with_highest_mean():
    entry = _result(5.0, 10.0)
    entry["step"] = 1
    entry["scales"]["1"].update(p95_wait_h=11.0, tail_mean_wait_h=12.0)
    entry["scales"]["2"] = dict(entry["scales"]["1"], avg_wait_h=9.0, p95_wait_h=100.0, tail_mean_wait_h=200.0)
    assert entry_wait_tail_metrics(entry) == (11.0, 12.0)
    assert "wait 10.0h, p95 11.0h, slowest 5% mean 12.0h" in format_best({"any": entry})
    old = {"step": 1, **_result(5.0, 4.0)}
    assert all(math.isnan(value) for value in entry_wait_tail_metrics(old))
    assert "p95 n/a, slowest 5% mean n/a" in format_best({"6h": old})


def test_validation_analysis_reports_tail_metrics_and_legacy_missing_values():
    import analyze_validation

    with tempfile.TemporaryDirectory() as directory:
        models = os.path.join(directory, "de_s7_test2024", "models", "weights")
        os.makedirs(models)
        record_result(models, 1, _result(5.0, 4.0))
        result = _result(10.0, 8.0)
        result["scales"]["1"].update(p95_wait_h=12.0, tail_mean_wait_h=15.0)
        record_result(models, 2, result)
        output = io.StringIO()
        with patch("sys.argv", ["analyze_validation", "--sessions", directory]), contextlib.redirect_stdout(output):
            analyze_validation.main()
        with open(os.path.join(directory, "validation_best.csv")) as f:
            rows = {row["budget"]: row for row in csv.DictReader(f)}
        assert rows["6h"]["p95_wait_h"] == "" and rows["6h"]["tail_mean_wait_h"] == ""
        assert rows["12h"]["wait_h"] == "8.0"
        assert rows["12h"]["p95_wait_h"] == "12.0" and rows["12h"]["tail_mean_wait_h"] == "15.0"
        assert "p95 n/a, slowest 5% mean n/a" in output.getvalue()
        assert "p95 12.0h, slowest 5% mean 15.0h" in output.getvalue()
        assert os.path.getsize(os.path.join(directory, "validation_frontier.png")) > 0


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASSED: {name}")
    print(f"All {len(tests)} validation tests passed.")


if __name__ == "__main__":
    main()
