"""Keep pooled wait tails intact through sweep parsing and reporting."""

from contextlib import redirect_stdout
from dataclasses import asdict
import importlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import analysis_metrics as metrics
from src.multi_run_analysis import aggregate_multi_run
from test.test_multi_run_analysis import balanced_frame


EPISODE = (
    "Episode 1: Savings=€10/€20, Power=1/2/2 MWh, "
    "CostPer1kCompleted=10/20/20 €/1k, Jobs=100/100 (100.0%), "
    "AvgWait=2.0h, P95Wait=5.0h, TailMeanWait=7.0h, Lost=0, "
    "Agent Occupancy (Nodes)=50.0%, Baseline Occupancy (Nodes)=60.0%, "
    "PropPower=1/2/2 MWh, PropCost=€10/€20/€20, PropSavings=€10/€10\n"
)
SUMMARY = """=== JOB PROCESSING METRICS ===
Agent:
  Average Wait Time: 3.0 hours
  P95 Wait Time: 12.0 hours
  Slowest 5% Mean Wait Time: 20.0 hours
Baseline:
  Average Wait Time: 1.0 hours
  P95 Wait Time: 4.0 hours
  Slowest 5% Mean Wait Time: 6.0 hours
Job Arrivals/Hour (mean ± std): 100.0 ± 1.0
Total Lost Jobs (Agent): 0
Total Lost Jobs (Baseline): 0
"""


def run_sweep(module_name, stdout):
    module = importlib.import_module(module_name)
    function_name, extra = {
        "analyze_lambda_occupancy": ("run_lambda_eval", (100,)),
        "analyze_seed_occupancy": ("run_seed_eval", (1,)),
        "analyze_arrivalscale_occupancy": ("run_scale_eval", (1.0, "sampling")),
    }[module_name]
    completed = SimpleNamespace(stdout=stdout, stderr="", returncode=0)
    with patch.object(module, "build_train_command", return_value=["train.py"]), \
            patch.object(module.subprocess, "run", return_value=completed), redirect_stdout(io.StringIO()):
        return getattr(module, function_name)(
            SimpleNamespace(eval_months=12, echo_train_output=False), Path("."), *extra)[0]


class WaitTailAnalysisTests(unittest.TestCase):
    def test_episode_fields_do_not_disturb_legacy_positional_parser(self):
        parsed = metrics.parse_episode_metrics(EPISODE)
        self.assertEqual(len(parsed), 18)
        np.testing.assert_array_equal(parsed[6], [2.0])

    def test_pooled_tails_propagate_through_all_sweeps_and_csv_json(self):
        for name in ("analyze_lambda_occupancy", "analyze_seed_occupancy", "analyze_arrivalscale_occupancy"):
            with self.subTest(sweep=name):
                stats = run_sweep(name, EPISODE + SUMMARY)
                self.assertEqual(stats.agent_p95_wait_hours, 12)
                self.assertEqual(stats.agent_tail_mean_wait_hours, 20)
                self.assertEqual(stats.baseline_p95_wait_hours, 4)
                self.assertEqual(stats.baseline_tail_mean_wait_hours, 6)
                self.assertEqual(stats.p95_wait_delta_hours, 8)
                self.assertEqual(stats.tail_mean_wait_delta_hours, 14)
                self.assertAlmostEqual(stats.effective_savings_mean, 10 / 3)
                row = metrics.csv_common_row(stats)
                self.assertEqual(row["agent_p95_wait_hours"], "12.000000")
                self.assertEqual(asdict(stats)["agent_tail_mean_wait_hours"], 20)
                start = metrics.CSV_COMMON_FIELDNAMES.index("agent_avg_wait_hours")
                self.assertEqual(metrics.CSV_COMMON_FIELDNAMES[start:start + 3], [
                    "agent_avg_wait_hours", "agent_p95_wait_hours", "agent_tail_mean_wait_hours"])

    def test_missing_run_tails_are_not_approximated_from_episodes(self):
        stats = run_sweep("analyze_lambda_occupancy", EPISODE)
        self.assertTrue(np.isnan(stats.agent_p95_wait_hours))
        self.assertTrue(np.isnan(stats.agent_tail_mean_wait_hours))

    def test_missing_agent_tails_do_not_capture_baseline_values(self):
        old_agent = SUMMARY.replace("  P95 Wait Time: 12.0 hours\n", "").replace(
            "  Slowest 5% Mean Wait Time: 20.0 hours\n", "")
        values = metrics.parse_wait_tail_summary(old_agent)
        self.assertTrue(np.isnan(values["agent_p95_wait_hours"]))
        self.assertTrue(np.isnan(values["agent_tail_mean_wait_hours"]))
        self.assertEqual(values["baseline_p95_wait_hours"], 4)

    def test_wait_comparison_plots_three_statistics_in_hours(self):
        stats = run_sweep("analyze_lambda_occupancy", EPISODE + SUMMARY)
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        metrics.plot_wait_deltas(ax, [stats], np.array([50.0]), np.array([1.0]),
                                 np.array([[0.1, 0.2, 0.3, 1.0]]), False)
        self.assertEqual(ax.get_legend_handles_labels()[1], ["Mean", "P95", "Slowest 5% mean"])
        np.testing.assert_allclose([container.lines[0].get_ydata()[0] for container in ax.containers],
                                   [2.0, 8.0, 14.0])

    def test_multi_run_aggregates_run_tails_and_keeps_columns_adjacent(self):
        raw = balanced_frame()
        raw["mean_wait_h"] = 2.0
        raw["p95_wait_h"] = raw["eval_seed"].astype(float)
        raw["tail_mean_wait_h"] = raw["p95_wait_h"] + 10.0
        result = aggregate_multi_run(raw)
        start = result.metrics.index("mean_wait_h")
        self.assertEqual(result.metrics[start:start + 3], ["mean_wait_h", "p95_wait_h", "tail_mean_wait_h"])
        expected = raw.groupby(["model", "scale"])["p95_wait_h"].mean().to_numpy()
        np.testing.assert_allclose(result.grand_total["p95_wait_h_mean"], expected)


if __name__ == "__main__":
    unittest.main()
