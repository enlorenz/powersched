"""Launch-wait percentiles and exact slowest-five-percent means."""

from collections import Counter
import unittest

import numpy as np

from src.config import CORES_PER_NODE, MAX_NODES
from src.job_management import JobQueue, dispatch_jobs, process_ongoing_jobs
from src.metrics_tracker import MetricsTracker


def launch(metrics, ages, *, baseline=False, dispatcher="firstfit"):
    queue = JobQueue(capacity=len(ages))
    queue.add([1] * len(ages), [1] * len(ages), [1] * len(ages))
    queue.rows()[:, 1] = sorted(ages, reverse=True)
    nodes = np.zeros(MAX_NODES, dtype=np.int32)
    cores = np.full(MAX_NODES, CORES_PER_NODE, dtype=np.int32)
    running = {}
    started, _ = dispatch_jobs(queue, nodes, cores, running, 0, metrics,
                               is_baseline=baseline, dispatcher=dispatcher)
    assert started == len(ages)
    return nodes, cores, running


class TestWaitMetrics(unittest.TestCase):
    def test_empty_single_and_zero_waits(self):
        for histogram, expected in (({}, (0, 0)), ({7: 1}, (7, 7)), ({0: 100}, (0, 0))):
            with self.subTest(histogram=histogram):
                self.assertEqual(MetricsTracker.wait_tail_metrics(histogram), expected)

    def test_cutoff_ties_do_not_enlarge_the_tail(self):
        # 95 jobs at zero, five at 100: p95 is zero, but the tail is only the five slow jobs.
        self.assertEqual(MetricsTracker.wait_tail_metrics({0: 95, 100: 5}), (0, 100))
        # The percentile falls inside a tie: include just enough of that tie in the tail.
        self.assertEqual(MetricsTracker.wait_tail_metrics({0: 94, 10: 5, 30: 1}), (10, 14))

    def test_fractional_boundary_job(self):
        # 21 jobs: the tail comprises one 30h job and 0.05 of a 10h job.
        p95, tail_mean = MetricsTracker.wait_tail_metrics({0: 19, 10: 1, 30: 1})
        self.assertEqual(p95, 10)
        self.assertAlmostEqual(tail_mean, (30 + 0.05 * 10) / 1.05)

    def test_large_population_rank_is_exact(self):
        self.assertEqual(MetricsTracker.wait_tail_metrics({0: 19 * 10**15, 12: 10**15}), (0, 12))

    def test_both_dispatchers_record_once_at_launch_for_each_side(self):
        for dispatcher in ("firstfit", "easy"):
            for baseline in (False, True):
                with self.subTest(dispatcher=dispatcher, baseline=baseline):
                    metrics = MetricsTracker()
                    nodes, cores, running = launch(metrics, [0] * 19 + [8], baseline=baseline, dispatcher=dispatcher)
                    self.assertEqual(metrics.get_wait_tail_metrics(baseline=baseline), (0, 8))
                    self.assertEqual(metrics.get_wait_tail_metrics(baseline=baseline, episode=False), (0, 8))
                    self.assertEqual(metrics.get_wait_tail_metrics(baseline=not baseline), (0, 0))
                    self.assertEqual(metrics.baseline_jobs_completed if baseline else metrics.jobs_completed, 0)
                    data = metrics.record_episode_completion(1)
                    prefix = "baseline_" if baseline else ""
                    self.assertEqual(data[prefix + "avg_wait_time"], 0.4)
                    self.assertEqual(data[prefix + "p95_wait_time"], 0)
                    self.assertEqual(data[prefix + "tail_mean_wait_time"], 8)
                    process_ongoing_jobs(nodes, cores, running, metrics, is_baseline=baseline)
                    self.assertEqual(metrics.get_wait_tail_metrics(baseline=baseline), (0, 8))
                    histogram = metrics.baseline_job_wait_time_histogram if baseline else metrics.job_wait_time_histogram
                    self.assertEqual(sum(histogram.values()), 20)

    def test_full_evaluation_pools_jobs_instead_of_episode_tail_metrics(self):
        metrics = MetricsTracker()
        launch(metrics, [100])
        first = metrics.record_episode_completion(1)
        metrics.reset_episode_metrics()
        launch(metrics, [0] * 19)
        second = metrics.record_episode_completion(2)
        self.assertEqual((first["p95_wait_time"], second["p95_wait_time"]), (100, 0))
        self.assertEqual(metrics.get_wait_tail_metrics(), (0, 0))
        self.assertEqual(metrics.get_wait_tail_metrics(episode=False), (0, 100))
        self.assertEqual(metrics.job_wait_time_histogram, Counter({0: 19, 100: 1}))

    def test_episode_and_timeline_resets_keep_agent_and_baseline_separate(self):
        metrics = MetricsTracker()
        launch(metrics, [4])
        launch(metrics, [9], baseline=True)
        metrics.reset_episode_metrics()
        self.assertEqual(metrics.get_wait_tail_metrics(), (0, 0))
        self.assertEqual(metrics.get_wait_tail_metrics(baseline=True), (0, 0))
        self.assertEqual(metrics.get_wait_tail_metrics(episode=False), (4, 4))
        self.assertEqual(metrics.get_wait_tail_metrics(baseline=True, episode=False), (9, 9))
        metrics.reset_timeline_metrics()
        self.assertEqual(metrics.get_wait_tail_metrics(episode=False), (0, 0))
        self.assertEqual(metrics.get_wait_tail_metrics(baseline=True, episode=False), (0, 0))

    def test_pending_job_is_counted_in_the_episode_where_it_launches(self):
        metrics = MetricsTracker()
        queue = JobQueue(capacity=1)
        queue.add([1], [1], [1])
        queue.rows()[:, 1] = 7
        nodes = np.full(MAX_NODES, -1, dtype=np.int32)
        cores = np.full(MAX_NODES, CORES_PER_NODE, dtype=np.int32)
        running = {}
        self.assertEqual(dispatch_jobs(queue, nodes, cores, running, 0, metrics)[0], 0)
        self.assertEqual(metrics.record_episode_completion(1)["p95_wait_time"], 0)
        metrics.reset_episode_metrics()
        queue.age(100)
        nodes[0] = 0
        self.assertEqual(dispatch_jobs(queue, nodes, cores, running, 0, metrics)[0], 1)
        self.assertEqual(metrics.get_wait_tail_metrics(), (8, 8))
        self.assertEqual(metrics.get_wait_tail_metrics(episode=False), (8, 8))


if __name__ == "__main__":
    unittest.main()
