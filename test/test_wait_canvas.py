"""The optional wait canvas accepts both CSV schemas and pooled run statistics."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from test.test_multi_run_plotting import PLOTTER, plot_inputs


WAIT_ALIASES = {
    "mean_wait_h": "agent_avg_wait_hours",
    "p95_wait_h": "agent_p95_wait_hours",
    "tail_mean_wait_h": "agent_tail_mean_wait_hours",
}
MODELS = ("e3p5s2", "e3p4s3", "e4p4s2", "e2p2s6")
WAIT_PAIRS = (
    ("mean_wait_h", "p95_wait_h", "Mean wait (h)", "P95 wait (h)"),
    ("mean_wait_h", "tail_mean_wait_h", "Mean wait (h)", "Slowest 5% mean wait (h)"),
    ("p95_wait_h", "tail_mean_wait_h", "P95 wait (h)", "Slowest 5% mean wait (h)"),
)


def run_plot(arguments):
    with patch.object(sys, "argv", [str(PLOTTER), *map(str, arguments)]), \
            patch.object(Figure, "savefig") as savefig, redirect_stdout(io.StringIO()):
        module = runpy.run_path(str(PLOTTER))
    return module, [Path(call.args[0]) for call in savefig.call_args_list]


class WaitCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.directories = plot_inputs(cls.root / "inputs")
        source = sorted(cls.directories[0].glob("*/eval_main.csv"))[0]
        base = pd.read_csv(source)
        base["p95_wait_h"] = base.mean_wait_h * 3
        base["tail_mean_wait_h"] = base.mean_wait_h * 5
        cls.canonical = pd.concat([
            base.assign(model=model, mean_wait_h=base.mean_wait_h + i,
                        p95_wait_h=base.p95_wait_h + 2 * i,
                        tail_mean_wait_h=base.tail_mean_wait_h + 3 * i)
            for i, model in enumerate(MODELS)
        ], ignore_index=True)
        long_path = cls.root / "eval_long.csv"
        cls.canonical.rename(columns=WAIT_ALIASES).to_csv(long_path, index=False)
        cls.single, cls.single_paths = run_plot([
            long_path, "--output-dir", cls.root / "single", "--name-suffix", "custom_wait",
        ])

        raw_frames = []
        for path in cls.directories[0].glob("*/eval_main.csv"):
            frame = pd.read_csv(path)
            frame["p95_wait_h"] = frame.mean_wait_h * 3
            frame["tail_mean_wait_h"] = frame.mean_wait_h * 5
            frame = frame.rename(columns=WAIT_ALIASES)
            frame.to_csv(path, index=False)
            raw_frames.append(frame)
        cls.raw = pd.concat(raw_frames, ignore_index=True)
        cls.multi, cls.multi_paths = run_plot([
            "--multi-run", "--input-dirs", cls.directories[0],
            "--output-dir", cls.root / "multi",
        ])

    @classmethod
    def tearDownClass(cls):
        plt.close("all")
        cls.temp.cleanup()

    def test_long_headers_are_preserved_and_aliased(self):
        frame = self.single["df"]
        for canonical, alias in WAIT_ALIASES.items():
            with self.subTest(metric=canonical):
                self.assertIn(alias, frame)
                np.testing.assert_allclose(frame[canonical], self.canonical[canonical])
                np.testing.assert_allclose(frame[canonical], frame[alias])

    def test_new_canvas_contains_every_input_model(self):
        self.assertNotIn("e3p4s3", self.single["MODELS"])
        self.assertEqual(len(self.single["fig_wait"].axes), 6)
        for ax, metric in zip(self.single["fig_wait"].axes, WAIT_ALIASES):
            curves = {line.get_label(): line for line in ax.lines if line.get_label() in MODELS}
            self.assertEqual(set(curves), set(MODELS))
            for model, curve in curves.items():
                expected = self.canonical.loc[self.canonical.model.eq(model)].sort_values("arrivals")
                np.testing.assert_allclose(curve.get_xdata(), expected.arrivals)
                np.testing.assert_allclose(curve.get_ydata(), expected[metric])
            self.assertFalse(any(isinstance(item, ErrorbarContainer) for item in ax.containers))

    def test_six_square_panels_render_in_two_rows(self):
        fig = self.single["fig_wait"]
        fig.canvas.draw()
        extents = [ax.get_window_extent() for ax in fig.axes]
        for ax, box in zip(fig.axes, extents):
            self.assertEqual(ax.get_box_aspect(), 1)
            self.assertAlmostEqual(box.width, box.height, places=5)
        self.assertEqual(len(extents), 6)
        for row in (extents[:3], extents[3:]):
            self.assertLess(row[0].x1, row[1].x0)
            self.assertLess(row[1].x1, row[2].x0)
            np.testing.assert_allclose([box.y0 for box in row], row[0].y0)
        for top, bottom in zip(extents[:3], extents[3:]):
            self.assertLess(bottom.y1, top.y0)
        self.assertFalse(fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[3]))

    def test_wait_output_names_follow_suffix_in_both_modes(self):
        for paths, suffix, directory in (
            (self.single_paths, "custom_wait", "single"),
            (self.multi_paths, "multi_run", "multi"),
        ):
            with self.subTest(suffix=suffix):
                wait_paths = [path for path in paths if path.stem.startswith("eval_oracle_wait_times_")]
                self.assertEqual(wait_paths, [
                    self.root / directory / f"eval_oracle_wait_times_{suffix}.png",
                    self.root / directory / f"eval_oracle_wait_times_{suffix}.pdf",
                ])

    def test_multi_run_uses_metric_means_and_total_sd_in_hours(self):
        frame = self.multi["df"].sort_values("scale")
        for ax, (canonical, alias) in zip(self.multi["fig_wait"].axes, WAIT_ALIASES.items()):
            expected = self.raw.groupby("scale")[alias].agg(["mean", "std"])
            np.testing.assert_allclose(frame[canonical], expected["mean"])
            np.testing.assert_allclose(frame[f"{canonical}_std_total"], expected["std"])
            np.testing.assert_allclose(frame[f"{canonical}_std_total"], frame[f"{alias}_std_total"])
            curve = next(line for line in ax.lines if line.get_label() == "e2p2s6")
            np.testing.assert_allclose(curve.get_ydata(), expected["mean"])
            errors = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
            self.assertEqual(len(errors), 1)
            self.assertTrue(errors[0].has_yerr)
            self.assertFalse(errors[0].has_xerr)
            segments = np.asarray(errors[0].lines[2][0].get_segments())
            np.testing.assert_allclose(segments[:, 0, 1], expected["mean"] - expected["std"])
            np.testing.assert_allclose(segments[:, 1, 1], expected["mean"] + expected["std"])
        self.assertIn("total observed sample SD", self.multi["fig_wait"]._suptitle.get_text())
        self.assertIn("top row", self.multi["fig_wait"]._suptitle.get_text().lower())

    def test_correlations_show_all_models_without_error_bars_or_arrival_markers(self):
        for module, expected_frame in ((self.single, self.canonical), (self.multi, self.multi["df"])):
            for ax, (x_metric, y_metric, xlabel, ylabel) in zip(module["fig_wait"].axes[3:], WAIT_PAIRS):
                with self.subTest(multi_run=module["args"].multi_run, pair=(x_metric, y_metric)):
                    self.assertEqual(ax.get_xlabel(), xlabel)
                    self.assertEqual(ax.get_ylabel(), ylabel)
                    scatters = {item.get_label(): item for item in ax.collections}
                    self.assertEqual(set(scatters), set(module["available_models"]))
                    for model, scatter in scatters.items():
                        expected = expected_frame.loc[expected_frame.model.eq(model), [x_metric, y_metric]]
                        np.testing.assert_allclose(scatter.get_offsets(), expected.to_numpy())
                    self.assertFalse(any(isinstance(item, ErrorbarContainer) for item in ax.containers))
                    self.assertFalse(any(line.get_label() == "scale = 1" for line in ax.lines))

    def test_correlations_filter_each_pair_without_requiring_arrivals(self):
        rows = pd.DataFrame({
            "model": ["e2p2s6"] * 5,
            "scale": [0.5, 1.0, 1.5, 2.0, 2.5],
            "arrivals": [100.0, np.nan, 300.0, 400.0, 500.0],
            "mean_wait_h": [1.0, 2.0, np.nan, 4.0, np.inf],
            "p95_wait_h": [3.0, 6.0, 9.0, np.nan, 15.0],
            "tail_mean_wait_h": [5.0, 10.0, 15.0, 20.0, 25.0],
        })
        plot = self.multi["plot_wait_canvas"]
        with patch.dict(plot.__globals__, {"df": rows, "available_models": ["e2p2s6"]}):
            fig = plot()
        self.addCleanup(plt.close, fig)
        expected_pairs = (
            [[1, 3], [2, 6]],
            [[1, 5], [2, 10], [4, 20]],
            [[3, 5], [6, 10], [9, 15], [15, 25]],
        )
        for ax, expected in zip(fig.axes[3:], expected_pairs):
            scatter = next(item for item in ax.collections if item.get_label() == "e2p2s6")
            np.testing.assert_allclose(scatter.get_offsets(), expected)

    def test_explicit_agent_headers_take_precedence_over_short_headers(self):
        frame = self.canonical.copy()
        for canonical, alias in WAIT_ALIASES.items():
            frame[alias] = frame[canonical] + 100
        source = self.root / "eval_canonical.csv"
        frame.to_csv(source, index=False)
        module, _ = run_plot([source, "--output-dir", self.root / "canonical"])
        self.assertIsNotNone(module["fig_wait"])
        for canonical, alias in WAIT_ALIASES.items():
            np.testing.assert_allclose(module["df"][canonical], frame[alias])

    def test_short_headers_remain_supported(self):
        source = self.root / "eval_short.csv"
        self.canonical.to_csv(source, index=False)
        module, _ = run_plot([source, "--output-dir", self.root / "short"])
        self.assertIsNotNone(module["fig_wait"])
        for ax, metric in zip(module["fig_wait"].axes, WAIT_ALIASES):
            curve = next(line for line in ax.lines if line.get_label() == "e3p4s3")
            expected = self.canonical.loc[self.canonical.model.eq("e3p4s3")].sort_values("arrivals")
            np.testing.assert_allclose(curve.get_ydata(), expected[metric])

    def test_either_missing_tail_column_disables_the_new_canvas(self):
        for missing in ("p95_wait_h", "tail_mean_wait_h"):
            with self.subTest(missing=missing):
                source = self.root / f"eval_missing_{missing}.csv"
                self.canonical.drop(columns=missing).to_csv(source, index=False)
                module, paths = run_plot([source, "--output-dir", self.root / f"missing_{missing}"])
                self.assertIsNone(module["fig_wait"])
                self.assertFalse(any("wait_times" in path.name for path in paths))

    def test_present_but_blank_tail_columns_still_render(self):
        frame = self.canonical.copy()
        frame[["p95_wait_h", "tail_mean_wait_h"]] = np.nan
        source = self.root / "eval_blank.csv"
        frame.to_csv(source, index=False)
        module, _ = run_plot([source, "--output-dir", self.root / "blank"])
        self.assertIsNotNone(module["fig_wait"])
        module["fig_wait"].canvas.draw()
        for ax in module["fig_wait"].axes[1:3]:
            for line in ax.lines:
                if line.get_label() in MODELS:
                    self.assertTrue(np.isnan(line.get_ydata()).all())
        for ax in module["fig_wait"].axes[3:]:
            self.assertTrue(all(len(item.get_offsets()) == 0 for item in ax.collections))


if __name__ == "__main__":
    unittest.main()
