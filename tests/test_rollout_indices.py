import unittest

import numpy as np
import torch

from datasets.rollout_datasets import ProcessedRolloutDataset


class RolloutIndicesTest(unittest.TestCase):
    def make_dataset(self):
        dataset = ProcessedRolloutDataset.__new__(
            ProcessedRolloutDataset
        )

        metadata = {
            "episode_start_indices": np.array([0, 2, 5, 9, 14]),
            "episode_end_indices": np.array([2, 5, 9, 14, 20]),
            "calibration_rollout_labels": np.array(
                [True, True, True, False, False],
                dtype=bool,
            ),
            "test_rollout_labels": np.array(
                [False, False, False, True, True],
                dtype=bool,
            ),
            "successful_rollout_labels": np.array(
                [True, False, True, True, False],
                dtype=bool,
            ),
            "failed_rollout_labels": np.array(
                [False, True, False, False, True],
                dtype=bool,
            ),
            "id_rollout_labels": np.array(
                [True, True, True, True, True],
                dtype=bool,
            ),
            "ood_rollout_labels": np.array(
                [False, False, False, False, False],
                dtype=bool,
            ),
            "num_steps": 20,
            "num_rollouts": 5,
            "episode_lengths": np.array([2, 3, 4, 5, 6]),
        }

        dataset.data = {
            "metadata": metadata,
            "obs_embeddings": torch.arange(
                20,
                dtype=torch.float32,
            ).reshape(20, 1),
        }
        dataset.dataset_loaded = True
        dataset.required_tensors = ["obs_embeddings"]
        dataset.optional_tensors = []
        dataset.normalize_tensors = {"obs_embeddings": False}
        dataset.allowed_subsets = [
            "all",
            "calibration",
            "test",
            "successful",
            "failed",
        ]
        dataset.allowed_subsubsets = ["all", "id", "ood"]
        dataset.required_metadata_keys = [
            "episode_start_indices",
            "episode_end_indices",
            "calibration_rollout_labels",
            "test_rollout_labels",
            "successful_rollout_labels",
            "failed_rollout_labels",
            "id_rollout_labels",
            "ood_rollout_labels",
            "num_steps",
            "num_rollouts",
            "episode_lengths",
        ]
        return dataset

    def test_filter_uses_global_indices_with_subset(self):
        dataset = self.make_dataset()

        starts, ends, labels = (
            dataset._filter_start_end_episode_indices(
                subset="calibration",
                rollout_indices=[2, 0],
            )
        )

        self.assertEqual(starts.tolist(), [0, 5])
        self.assertEqual(ends.tolist(), [2, 9])
        self.assertEqual(labels.tolist(), [True, True])

    def test_get_subset_uses_global_indices(self):
        dataset = self.make_dataset()

        subset = dataset.get_subset(
            subset="calibration",
            required_tensors="obs_embeddings",
            rollout_indices=[2, 0],
        )

        self.assertEqual(
            subset["obs_embeddings"].reshape(-1).tolist(),
            [0.0, 1.0, 5.0, 6.0, 7.0, 8.0],
        )

    def test_rejects_index_outside_subset(self):
        dataset = self.make_dataset()

        with self.assertRaisesRegex(ValueError, "outside subset"):
            dataset._filter_start_end_episode_indices(
                subset="calibration",
                rollout_indices=[3],
            )

    def test_rejects_duplicate_indices(self):
        dataset = self.make_dataset()

        with self.assertRaisesRegex(ValueError, "duplicates"):
            dataset._filter_start_end_episode_indices(
                subset="calibration",
                rollout_indices=[0, 0],
            )

    def test_rejects_out_of_range_index(self):
        dataset = self.make_dataset()

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            dataset._filter_start_end_episode_indices(
                subset="calibration",
                rollout_indices=[5],
            )


if __name__ == "__main__":
    unittest.main()