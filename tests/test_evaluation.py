import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from evaluation import FEATURES, TARGET, evaluate_model, validate_data


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(1.0, index=range(6), columns=FEATURES)
        self.frame[TARGET] = [0, 1, 0, 1, 0, 1]

    def test_feature_order_and_optional_id(self):
        self.frame["ID"] = range(6)
        features, target = validate_data(self.frame[self.frame.columns[::-1]])
        self.assertEqual(list(features), FEATURES)
        self.assertEqual(target.tolist(), [0, 1, 0, 1, 0, 1])

    def test_invalid_uploads(self):
        for value in [np.nan, np.inf, "bad"]:
            frame = self.frame.astype(object)
            frame.loc[0, "AGE"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_data(frame)
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_data(self.frame.drop(columns="PAY_0"))
        with self.assertRaisesRegex(ValueError, "no data"):
            validate_data(self.frame.iloc[:0])
        self.frame.loc[0, TARGET] = 2
        with self.assertRaisesRegex(ValueError, "target must be"):
            validate_data(self.frame)

    def test_metrics_and_single_class_auc(self):
        features, target = validate_data(self.frame)
        model = DummyClassifier(strategy="most_frequent").fit(features, target)
        metrics, predictions, scores = evaluate_model(model, features, target)
        self.assertEqual(metrics["Accuracy"], 0.5)
        self.assertEqual(metrics["AUC"], 0.5)
        self.assertEqual(metrics["Recall"], 0.0)
        self.assertEqual(len(predictions), len(scores))
        metrics, _, _ = evaluate_model(model, features.iloc[:1], target.iloc[:1])
        self.assertIsNone(metrics["AUC"])


if __name__ == "__main__":
    unittest.main()