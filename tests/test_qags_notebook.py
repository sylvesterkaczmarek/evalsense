"""Check executable notebook selectors against the evaluator's actual score names."""

import ast
import asyncio
import json
from pathlib import Path
import unittest

import polars as pl

from evalsense.evaluation.evaluators.qags import QagsScoreCalculator, get_qags_evaluator
from evalsense.generation import ModelConfig
from test_qags import FixtureConfig, model_with


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "Experiments.ipynb"


def code_cells() -> list[str]:
    return [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]


def notebook_evaluator_names() -> dict[str, str]:
    names = {}
    for cell in code_cells():
        if "get_qags_evaluator(" not in cell:
            continue
        for node in ast.walk(ast.parse(cell)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_qags_evaluator"
            ):
                args = {kw.arg: kw.value for kw in node.keywords}
                name = ast.literal_eval(args["name"])
                model_name = ast.literal_eval(args["model_name"])
                mode = "ternary" if name.startswith("Ternary") else "judge"
                evaluator = get_qags_evaluator(
                    name=name,
                    model_name=model_name,
                    config=FixtureConfig(mode, logprobs=False),
                    model_config=ModelConfig("mockllm/judge"),
                )
                names[mode] = evaluator.name
    return names


class NotebookNamesTests(unittest.TestCase):
    def test_summary_selection_matches_current_evaluator_output(self):
        names = notebook_evaluator_names()
        self.assertEqual(set(names), {"ternary", "judge"})
        calculator = QagsScoreCalculator(
            model_with("One?", "One?", "yes", "yes"),
            FixtureConfig("ternary", logprobs=False),
            name=names["ternary"],
        )
        score = asyncio.run(calculator.calculate_async(prediction="candidate"))
        assert isinstance(score.value, dict)
        columns = {f"{name}/mean": [value] for name, value in score.value.items()}
        columns[f"{names['judge']}/mean"] = [0.5]
        for name in (
            "BLEU Precision/BLEU",
            "ROUGE-1/mean",
            "ROUGE-2/mean",
            "ROUGE-L/mean",
            "BERTScore F1/mean",
        ):
            columns[name] = [0.5]
        frame = pl.DataFrame({"model": ["fixture"], **columns})
        selection = next(
            cell
            for cell in code_cells()
            if "summary_results.sort(" in cell and "Accuracy/mean" in cell
        )
        result = eval(
            compile(selection, str(NOTEBOOK), "eval"),
            {"summary_results": frame, "pl": pl},
        )
        self.assertEqual(result.height, 1)
        self.assertIn(f"{names['ternary']} Accuracy/mean", result.columns)
        self.assertIn(f"{names['judge']}/mean", result.columns)

    def test_correlation_aliases_match_current_names(self):
        names = notebook_evaluator_names()
        cell = next(cell for cell in code_cells() if "metric_labels=" in cell)
        call = next(
            node
            for node in ast.walk(ast.parse(cell))
            if isinstance(node, ast.Call)
            and any(kw.arg == "metric_labels" for kw in node.keywords)
        )
        labels = ast.literal_eval(
            next(kw.value for kw in call.keywords if kw.arg == "metric_labels")
        )
        self.assertEqual(
            set(labels),
            {
                f"{names['ternary']} {metric}"
                for metric in ("Coverage", "Groundedness", "Accuracy")
            },
        )
        self.assertEqual(len(set(labels.values())), len(labels))

    def test_executable_cells_do_not_select_legacy_names(self):
        for cell in code_cells():
            self.assertNotIn("Ternary Ref. QAGS (", cell)
            self.assertNotIn("Judge Ref. QAGS (", cell)


if __name__ == "__main__":
    unittest.main()
