"""Wait-tail reporting keeps pooled results distinct from episode statistics."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt

from src.callbacks import ComputeClusterCallback
from src.evaluation_csv import _parse_episode_lines, build_evaluation_csv_row, evaluation_csv_fieldnames
from src.evaluation_summary import build_episode_summary_line
from src.metrics_tracker import MetricsTracker
from src.plotter import plot_episode_summary


EPISODE = (
    "Episode 1: Jobs=100/120 (83%), AvgWait=2.0h, "
    "P95Wait=4.0h, TailMeanWait=6.0h, Lost=0, "
    "PropPower=1.0/2.0/3.0 MWh, PropCost=€100/€200/€300, "
    "CostPer1kCompleted=1000/2000/3000 €/1k"
)


def aggregate(agent_tail="", baseline_tail=""):
    return (
        "\n=== JOB PROCESSING METRICS ===\nAgent:\n"
        "  Jobs Launched: 100 / 120\n"
        "  Jobs Completed: 100 / 120 (83.3%)\n"
        "  Average Wait Time: 3.0 hours\n"
        f"{agent_tail}"
        "Baseline:\n"
        "  Jobs Launched: 100 / 120\n"
        "  Jobs Completed: 100 / 120 (83.3%)\n"
        "  Average Wait Time: 0.0 hours\n"
        f"{baseline_tail}"
    )


def csv_row(stdout):
    return build_evaluation_csv_row(
        stdout, sweep_parameter="scale", sweep_value=1.0,
        model="test", oracle_enabled=False,
    )


class WaitCsvTests(unittest.TestCase):
    def test_tracker_episode_values_reach_summary_parser_and_training_logger(self):
        metrics = MetricsTracker()
        metrics.episode_job_wait_time_histogram.update({2: 19, 12: 1})
        metrics.episode_jobs_launched = 20
        metrics.episode_total_job_wait_time_launch = 50
        metrics.episode_baseline_job_wait_time_histogram.update({1: 19, 6: 1})
        episode = metrics.record_episode_completion(1)
        line = build_episode_summary_line(1, episode, 0, 0.0, 0.0, 0.0, 0.0)
        self.assertIn("AvgWait=2.5h, P95Wait=2.0h, TailMeanWait=12.0h", line)
        parsed = _parse_episode_lines(line)[0]
        self.assertEqual(parsed["p95_wait_time"], 2.0)
        self.assertEqual(parsed["tail_mean_wait_time"], 12.0)
        logged = {}
        callback = SimpleNamespace(logger=SimpleNamespace(record_mean=logged.__setitem__))
        ComputeClusterCallback._record_episode(callback, episode)
        self.assertEqual(logged["metrics/p95_wait_hours"], 2.0)
        self.assertEqual(logged["metrics/tail_mean_wait_hours"], 12.0)
        self.assertEqual(logged["metrics/baseline_p95_wait_hours"], 1.0)
        self.assertEqual(logged["metrics/baseline_tail_mean_wait_hours"], 6.0)

    def test_final_pooled_metrics_are_used_in_adjacent_columns(self):
        row = csv_row(EPISODE + aggregate(
            "  P95 Wait Time: 8.0 hours\n"
            "  Slowest 5% Mean Wait Time: 12.0 hours\n"
        ))
        self.assertEqual(row["mean_wait_h"], 3.0)
        self.assertEqual(row["p95_wait_h"], 8.0)
        self.assertEqual(row["tail_mean_wait_h"], 12.0)
        fields = evaluation_csv_fieldnames("scale")
        i = fields.index("mean_wait_h")
        self.assertEqual(fields[i:i + 3], ("mean_wait_h", "p95_wait_h", "tail_mean_wait_h"))

    def test_episode_statistics_do_not_substitute_for_missing_pooled_values(self):
        for output in (EPISODE, EPISODE + aggregate()):
            with self.subTest(output=output):
                row = csv_row(output)
                self.assertIsNone(row["p95_wait_h"])
                self.assertIsNone(row["tail_mean_wait_h"])

    def test_baseline_tail_values_do_not_become_agent_values(self):
        row = csv_row(EPISODE + aggregate(baseline_tail=(
            "  P95 Wait Time: 20.0 hours\n"
            "  Slowest 5% Mean Wait Time: 30.0 hours\n"
        )))
        self.assertIsNone(row["p95_wait_h"])
        self.assertIsNone(row["tail_mean_wait_h"])

    def test_old_episode_output_still_parses(self):
        legacy = EPISODE.replace("P95Wait=4.0h, TailMeanWait=6.0h, ", "")
        row = csv_row(legacy + aggregate())
        self.assertEqual(row["mean_wait_h"], 3.0)
        self.assertIsNone(row["p95_wait_h"])
        self.assertIsNone(row["tail_mean_wait_h"])


class WaitPlotTests(unittest.TestCase):
    def test_wait_series_share_hours_axis(self):
        env = SimpleNamespace(weights="test", plots_dir="/tmp")
        episodes = [dict(avg_wait_time=2.0, p95_wait_time=8.0, tail_mean_wait_time=12.0)]
        with patch("src.plotter.plt.close"):
            result = plot_episode_summary(env, episodes, save=False, show=False)
            figure = plt.gcf()
        try:
            wait_axis = figure.axes[1]
            self.assertEqual(wait_axis.get_ylabel(), "Wait (hours)")
            self.assertEqual([line.get_label() for line in wait_axis.lines], [
                "Avg wait (h)", "P95 wait (h)", "Slowest 5% mean wait (h)",
            ])
            self.assertEqual(result["p95_wait_time_avg"], 8.0)
            self.assertEqual(result["tail_mean_wait_time_avg"], 12.0)
        finally:
            plt.close(figure)

    def test_old_episode_records_do_not_invent_tail_values(self):
        env = SimpleNamespace(weights="test", plots_dir="/tmp")
        result = plot_episode_summary(env, [dict(avg_wait_time=2.0)], save=False, show=False)
        self.assertTrue(math.isnan(result["p95_wait_time_avg"]))
        self.assertTrue(math.isnan(result["tail_mean_wait_time_avg"]))


if __name__ == "__main__":
    unittest.main()
