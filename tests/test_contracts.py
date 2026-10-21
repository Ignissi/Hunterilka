import unittest

import numpy as np
import pandas as pd

from scripts.evaluate import period_metrics
from src.contracts import feature_contract, validate_feature_tables
from src.report import number
from src.storage import load_config


class ContractTests(unittest.TestCase):
    def test_same_columns_do_not_make_changed_window_compatible(self):
        config = load_config()
        prepared = {"dataset_id": "one", "feature_contract": feature_contract(config)}
        validate_feature_tables(config, {"dataset_id": "one"}, prepared)
        with self.assertRaises(ValueError):
            validate_feature_tables(dict(config, window_days=10), {"dataset_id": "one"}, prepared)
        with self.assertRaises(ValueError):
            validate_feature_tables(config, {"dataset_id": "two"}, prepared)

    def test_single_class_period_is_reportable(self):
        test = pd.DataFrame({"timestamp": ["2022-09-10T00:00:00", "2022-09-10T00:01:00", "2022-09-11T00:00:00"],
                             "is_laundering": [0, 1, 1]})
        result = period_metrics(test, np.array([0.1, 0.8, 0.9]), 0.5)
        self.assertIsNone(result["tail"]["ap"])
        self.assertEqual(number(result["tail"]["ap"]), "н/д")
