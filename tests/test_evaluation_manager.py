import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluation_manager import EvaluationManager


class CombinedEvaluationTest(unittest.TestCase):
    threshold_styles = ("ct_quantile", "tvt_quantile")

    def make_method_result(self, method, quantiles):
        scores = {
            threshold_style: {
                quantile: {1: [np.array([0.5])]}
                for quantile in quantiles
            }
            for threshold_style in self.threshold_styles
        }

        return {
            "method": method,
            "quantiles": quantiles,
            "window_sizes": [1],
            "calibration_uncertainty_scores": {},
            "test_uncertainty_scores": {},
            "calibration_thresholds": {
                threshold_style: {}
                for threshold_style in self.threshold_styles
            },
            "test_scores_by_threshold": scores,
            "avg_inference_time": 0.01,
            "max_episode_length": 1,
            "successful_test_rollouts": np.array([True]),
            "id_test_rollouts": np.array([True]),
            "ood_test_rollouts": np.array([False]),
            "cfg": OmegaConf.create({"detection_patience": 1}),
        }

    @patch(
        "evaluation.evaluation_manager.calculate_metrics",
        return_value={"TWA": 1.0},
    )
    def test_combined_quantiles_are_not_repeated_per_threshold_style(
        self,
        _calculate_metrics,
    ):
        manager = EvaluationManager.__new__(EvaluationManager)

        total_results = {
            "rnd_oe": self.make_method_result(
                "rnd_oe",
                [0.9, 0.95],
            ),
            "entropy": self.make_method_result(
                "entropy",
                [0.8, 0.85],
            ),
        }
        combination = {
            "m1": {"name": "rnd_oe"},
            "m2": {"name": "entropy"},
            "operation": "and",
        }

        combined = manager._combine_two_methods(
            combination,
            total_results,
        )
        result = combined["rnd_oe_and_entropy"]

        expected_quantiles = [
            "0.9/0.8",
            "0.9/0.85",
            "0.95/0.8",
            "0.95/0.85",
        ]

        self.assertEqual(result["quantiles"], expected_quantiles)
        self.assertEqual(len(result["quantiles"]), 4)
        self.assertEqual(len(set(result["quantiles"])), 4)

        for threshold_style in self.threshold_styles:
            self.assertEqual(
                list(result["test_metrics"][threshold_style]),
                expected_quantiles,
            )


if __name__ == "__main__":
    unittest.main()