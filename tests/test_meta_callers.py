"""Keep examples and optional web UI aligned with the declared study design."""

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, cast
import unittest
from unittest.mock import Mock, patch
import warnings

import polars as pl

from test_meta_evidence import complete


class NotebookMetaTests(unittest.TestCase):
    def test_notebook_passes_planned_tiers_and_direction(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/Meta-Evaluation.ipynb"
        notebook = json.loads(path.read_text())
        cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        calls = [cell for cell in cells if "corr_results = analyser(" in cell]
        self.assertEqual(len(calls), 1)
        fixture = Mock(return_value=pl.DataFrame())
        scope = {
            "analyser": fixture,
            "aci_project": object(),
            "tiered_perturbation_types": [[], [], [], []],
        }
        exec(compile(calls[0], str(path), "exec"), scope)
        options = fixture.call_args.kwargs
        self.assertEqual(list(options["expected_tiers"]), [0, 1, 2, 3])
        self.assertIs(options["lower_tier_is_better"], True)
        self.assertEqual(options["seed"], 0)
        for cell in cells:
            if not any(
                line.lstrip().startswith(("%", "!")) for line in cell.splitlines()
            ):
                ast.parse(cell)

    def test_notebook_does_not_claim_intended_tiers_are_validated(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/Meta-Evaluation.ipynb"
        content = "\n".join(
            "".join(c["source"])
            for c in json.loads(path.read_text())["cells"]
            if c["cell_type"] == "markdown"
        )
        self.assertIn("not independently checked", content)
        self.assertIn("expected_sample_ids", content)
        self.assertIn("tier_validation_field", content)
        self.assertNotIn(
            "highest correlation can be expected to be the most suitable", content
        )


@unittest.skipUnless(importlib.util.find_spec("gradio"), "webui extra not installed")
class WebMetaTests(unittest.TestCase):
    def test_results_require_planned_tiers(self):
        import gradio as gr
        from evalsense.webui.components.results import load_project

        for text in ("", "0,1", "0,1,one", "0,1,1", "0,1,2.0"):
            with (
                self.subTest(text=text),
                patch(
                    "evalsense.webui.components.results.Project",
                    return_value=complete({1: [0, 1, 2]}),
                ),
            ):
                with self.assertRaises(gr.Error):
                    load_project("fixture", True, text)

    def test_results_keep_counts_and_interval_status_visible(self):
        from evalsense.webui.components.results import load_project

        data = complete({1: [2, 1, 0]})
        for log in data.get_logs(type="evaluation").values():
            for sample in log.samples or []:
                sample.metadata["perturbation_tier"] = sample.metadata[
                    "perturbation_type_tier"
                ]
        with (
            patch(
                "evalsense.webui.components.results.Project",
                return_value=data,
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            table, _ = load_project("fixture", True, "0, 1, 2")
        self.assertEqual(table["avg_correlation"].iloc[0], 1)
        self.assertEqual(table["n_included"].iloc[0], 1)
        self.assertEqual(table["ci_status"].iloc[0], "insufficient_samples")
        self.assertEqual(table["validation_policy"].iloc[0], "not_checked")

    def test_old_two_tier_generation_config_fails_before_loading_models(self):
        from evalsense.webui.execution import execute_meta_evaluation

        with patch("evalsense.webui.execution.get_dataset_manager") as load:
            with self.assertRaisesRegex(ValueError, "at least three"):
                execute_meta_evaluation(
                    cast(Any, {"perturbation_tier_subprompts": ["one", "two"]})
                )
            load.assert_not_called()

    def test_default_generation_design_has_at_least_three_tiers(self):
        from evalsense.webui.state import get_initial_state

        # Reading defaults must not start an evaluation or create a project.
        state = get_initial_state()
        self.assertGreaterEqual(state["perturbation_tiers"], 3)
        self.assertEqual(
            len(state["perturbation_tier_subprompts"]), state["perturbation_tiers"]
        )


if __name__ == "__main__":
    unittest.main()
