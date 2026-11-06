import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from evaluation import MODEL_FILES, evaluate_model, validate_data
import train

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_PRESENT = all((ROOT / "model" / filename).exists() for filename in MODEL_FILES.values())


class ReportTests(unittest.TestCase):
    def test_training_prefers_local_workbook(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            workbook = directory / "default of credit card clients.xls"
            workbook.touch()
            with patch.object(train, "ROOT", directory), patch.object(train.pd, "read_excel") as read_excel:
                train.load_dataset()
            read_excel.assert_called_once_with(workbook, header=1, engine="xlrd")

    def test_readme_uses_measured_results(self):
        metadata = {"metrics": {name: {metric: 0.5 for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}
                                for name in MODEL_FILES}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            readme = directory / "README.md"
            readme.write_text((ROOT / "README.md").read_text())
            with patch.object(train, "ROOT", directory):
                train.update_readme(metadata)
            text = readme.read_text()
            for name in MODEL_FILES:
                self.assertIn(f"| {name} | 0.5000", text)
            self.assertIn("highest held-out MCC", text)


@unittest.skipUnless(ARTIFACTS_PRESENT, "Run train.py to generate saved-model integration fixtures.")
class SavedModelTests(unittest.TestCase):
    def test_saved_models_reproduce_report_without_fitting(self):
        metadata = json.loads((ROOT / "model" / "results.json").read_text())
        csv_path = ROOT / "test_data.csv"
        self.assertEqual(hashlib.sha256(csv_path.read_bytes()).hexdigest(), metadata["test_sha256"])
        features, target = validate_data(pd.read_csv(csv_path))
        self.assertEqual(len(features), 6000)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Inference must not fit")):
            for name, filename in MODEL_FILES.items():
                model = joblib.load(ROOT / "model" / filename)
                metrics, _, _ = evaluate_model(model, features, target)
                for metric, value in metrics.items():
                    self.assertAlmostEqual(value, metadata["metrics"][name][metric], places=10)

    def test_frontend_and_model_selection_without_fitting(self):
        with patch.object(Pipeline, "fit", side_effect=AssertionError("App must not fit")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertEqual(len(app.dataframe[0].value), 5)
            self.assertEqual(app.metric[0].value, "6,000")
            app.selectbox[0].select("Support Vector Machine").run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Support Vector Machine", [heading.value for heading in app.subheader])
            app.radio[0].set_value("Upload CSV").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.info[0].value, "No test CSV selected.")


if __name__ == "__main__":
    unittest.main()