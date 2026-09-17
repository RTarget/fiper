import unittest

import torch

from datasets.tac_fiper_a0 import SortingA0Dataset
from evaluation.tac_sorting_fixed_eval import aligned_windows, fixed_metrics


class FixedSortingEvalTest(unittest.TestCase):
    def test_metrics_use_global_denominator_and_strict_threshold(self):
        episodes = [dict(successful=False, scores=[0., 2.]),
                    dict(successful=False, scores=[1., 1.]),
                    dict(successful=True, scores=[0., 1.]),
                    dict(successful=True, scores=[2.])]
        metrics, steps = fixed_metrics(episodes, 1., 5)
        self.assertEqual(steps, [1, None, None, 0])
        for name in ("TP", "TN", "FP", "FN"):
            self.assertEqual(metrics[name], 1)
        self.assertEqual(metrics["balanced_accuracy"], .5)
        self.assertEqual(metrics["avg_detection_time"], .25)
        self.assertEqual(metrics["TWA"], .4375)

    def test_no_detected_failures_uses_original_default(self):
        metrics, _ = fixed_metrics([dict(successful=False, scores=[0.]),
                                    dict(successful=True, scores=[0.])], 1., 2)
        self.assertEqual(metrics["avg_detection_time"], 1.)
        self.assertEqual(metrics["TWA"], .5)

    def test_rejects_invalid_metric_inputs(self):
        for threshold, values in ((0., [0.]), (1., [float("nan")]), (1., [])):
            with self.assertRaises(ValueError):
                fixed_metrics([dict(successful=False, scores=values),
                               dict(successful=True, scores=[0.])], threshold, 2)

    def test_windows_match_constructor_at_every_step(self):
        obs = torch.arange(33, dtype=torch.float32).reshape(11, 3)
        actions = torch.arange(264, dtype=torch.float32).reshape(11, 2, 2, 6)
        dataset = SortingA0Dataset.__new__(SortingA0Dataset)
        dataset.episodes = [(obs, actions)]
        dataset.samples = [(0, 2, t, 1) for t in range(11)]
        windows = aligned_windows(obs, actions)
        for t in range(11):
            sample = dataset[t]
            for key in ("obs_embeddings", "action_preds", "padding_mask"):
                self.assertTrue(torch.equal(windows[key][t], sample[key]))


if __name__ == "__main__":
    unittest.main()
