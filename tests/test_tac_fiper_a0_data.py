import unittest
from unittest.mock import Mock

import numpy as np
import torch

from datasets.rollout_datasets import ProcessedRolloutDataset
from datasets.tac_fiper_a0 import HISTORY, ROLE_INDICES, SortingA0Dataset, make_a0_dataloader
from evaluation.temporal_models import ObservationActionTemporalModel


def make_source(obs_dim=3, samples=2, horizon=2, action_dim=1):
    """Use the real iterator with synthetic data, including an unreadable test rollout."""
    lengths = np.array([1, 2, 10] * 17)
    ends = np.cumsum(lengths)
    starts = ends - lengths
    calibration = np.arange(51) < 50
    metadata = {
        "num_rollouts": 51, "num_steps": int(ends[-1]),
        "episode_start_indices": starts, "episode_end_indices": ends,
        "episode_lengths": lengths,
        "calibration_rollout_labels": calibration,
        "test_rollout_labels": ~calibration,
        "successful_rollout_labels": np.ones(51, dtype=bool),
        "failed_rollout_labels": np.zeros(51, dtype=bool),
        "id_rollout_labels": np.ones(51, dtype=bool),
        "ood_rollout_labels": np.zeros(51, dtype=bool),
        "actions": {"action_dim": action_dim},
    }
    obs = torch.zeros(int(ends[-1]), obs_dim)
    action = torch.zeros(int(ends[-1]), samples, horizon, action_dim)
    for i, (start, end) in enumerate(zip(starts, ends)):
        for t in range(int(end - start)):
            obs[start + t] = i * 100 + t
            action[start + t] = i * 100 + t
    # Any all-data finite check or test slicing would encounter these sentinels.
    obs[starts[-1]:] = float("nan")
    action[starts[-1]:] = float("nan")
    source = ProcessedRolloutDataset(
        "/unused/sorting", "/unused/configs",
        required_tensors=["obs_embeddings", "action_preds"],
    )
    source.data = {"metadata": metadata, "obs_embeddings": obs, "action_preds": action}
    source.dataset_loaded = True
    source.normalize_tensors = {"obs_embeddings": True, "action_preds": True}
    source.normalize = Mock(side_effect=AssertionError("must not use baseline normalization"))
    source.iterate_episodes = Mock(wraps=source.iterate_episodes)
    return source


class SortingA0DataTest(unittest.TestCase):
    def test_role_isolation_counts_and_no_baseline_normalization(self):
        source = make_source()
        lengths = source.get_metadata()["episode_lengths"]
        seen = []
        for role, indices in ROLE_INDICES.items():
            source.iterate_episodes.reset_mock()
            dataset = SortingA0Dataset(source, role, task="sorting")
            expected = sum(int(lengths[i]) if role == "threshold"
                           else 2 * (int(lengths[i]) - 1) for i in indices)
            self.assertEqual(len(dataset), expected)
            self.assertEqual(source.iterate_episodes.call_count, len(indices))
            actual_indices = []
            for call in source.iterate_episodes.call_args_list:
                self.assertEqual(call.kwargs["subset"], "calibration")
                self.assertEqual(call.kwargs["normalize_tensors"],
                                 {"obs_embeddings": False, "action_preds": False})
                actual_indices.extend(call.kwargs["rollout_indices"])
            self.assertEqual(actual_indices, list(indices))
            seen.extend(actual_indices)
            targets = [int(dataset[i]["target"].item()) for i in range(len(dataset))]
            if role == "threshold":
                self.assertEqual(set(targets), {1})
            else:
                self.assertEqual(targets.count(0), targets.count(1))
        self.assertEqual(sorted(seen), list(range(50)))
        source.normalize.assert_not_called()

    def test_rejects_invalid_metadata_before_tensor_access(self):
        for problem in ("test", "noncalibration", "failed", "overlap", "extra", "dtype", "boundary"):
            with self.subTest(problem=problem):
                source = make_source()
                m = source.get_metadata()
                # Index 3 is selection: also validate it when constructing representation.
                if problem in ("test", "noncalibration"):
                    m["calibration_rollout_labels"][3] = False
                    m["test_rollout_labels"][3] = (problem == "test")
                elif problem == "failed":
                    m["successful_rollout_labels"][3] = False
                    m["failed_rollout_labels"][3] = True
                elif problem == "overlap":
                    m["test_rollout_labels"][3] = True
                elif problem == "extra":
                    m["calibration_rollout_labels"][50] = True
                    m["test_rollout_labels"][50] = False
                elif problem == "dtype":
                    m["calibration_rollout_labels"] = m["calibration_rollout_labels"].astype(int)
                else:
                    m["episode_start_indices"][1] = 0
                with self.assertRaises(ValueError):
                    SortingA0Dataset(source, "representation", task="sorting")
                source.iterate_episodes.assert_not_called()

    def test_rejects_unknown_role_and_task(self):
        source = make_source()
        for role, task in (("test", "sorting"), ("all", "sorting"), ("selection", "stacking")):
            with self.subTest(role=role, task=task), self.assertRaises(ValueError):
                SortingA0Dataset(source, role, task=task)
        source.iterate_episodes.assert_not_called()

    def test_exact_padding_and_episode_local_shift(self):
        dataset = SortingA0Dataset(make_source(), "representation", task="sorting")
        # Every paired timestep, including left padding and windows after t >= 8.
        for index in range(0, len(dataset), 2):
            positive, negative = dataset[index], dataset[index + 1]
            t, rollout = positive["timestep"], positive["rollout_index"]
            start = max(0, t - HISTORY + 1)
            expected = [rollout * 100 + step for step in range(start, t + 1)]
            valid = ~positive["padding_mask"]
            self.assertEqual(positive["obs_embeddings"][valid, 0].tolist(), expected)
            self.assertEqual(positive["action_preds"][valid, 0, 0, 0].tolist(), expected)
            self.assertEqual(negative["action_preds"][valid, 0, 0, 0].tolist(),
                             expected[-1:] + expected[:-1])
            self.assertTrue(torch.equal(positive["obs_embeddings"], negative["obs_embeddings"]))
            self.assertTrue(torch.equal(positive["padding_mask"], negative["padding_mask"]))
            for item in (positive, negative):
                self.assertEqual(torch.count_nonzero(item["obs_embeddings"][~valid]).item(), 0)
                self.assertEqual(torch.count_nonzero(item["action_preds"][~valid]).item(), 0)

    def test_threshold_retains_first_step_and_single_step_episodes(self):
        dataset = SortingA0Dataset(make_source(), "threshold", task="sorting")
        first_steps = [dataset[i] for i in range(len(dataset)) if dataset[i]["timestep"] == 0]
        self.assertEqual(len(first_steps), 10)
        for item in first_steps:
            self.assertEqual(item["padding_mask"].tolist(), [True] * 7 + [False])
            self.assertEqual(item["target"].item(), 1)
        trained = SortingA0Dataset(make_source(), "representation", task="sorting")
        self.assertTrue(all(trained[i]["timestep"] >= 1 for i in range(len(trained))))

    def test_rejects_bad_tensors(self):
        for key in ("obs_embeddings", "action_preds"):
            for problem in ("rank", "integer", "nan", "inf", "length", "device", "overflow"):
                with self.subTest(key=key, problem=problem):
                    source = make_source()
                    tensor = source.data[key]
                    if problem == "rank":
                        source.data[key] = tensor.unsqueeze(1)
                    elif problem == "integer":
                        source.data[key] = torch.zeros_like(tensor, dtype=torch.int64)
                    elif problem in ("nan", "inf"):
                        tensor[0] = float(problem)
                    elif problem == "length":
                        source.data[key] = tensor[:0]
                    elif problem == "device":
                        source.data[key] = torch.empty(tensor.shape, device="meta")
                    else:
                        source.data[key] = tensor.double()
                        source.data[key][0] = 1e100
                    with self.assertRaises(ValueError):
                        SortingA0Dataset(source, "representation", task="sorting")

    def test_source_and_repeated_samples_are_not_mutated(self):
        source = make_source()
        original = source.data["action_preds"][:source.get_metadata()["episode_start_indices"][-1]].clone()
        dataset = SortingA0Dataset(source, "representation", task="sorting")
        item = dataset[1]
        expected = item["action_preds"].clone()
        item["action_preds"].fill_(-999)
        self.assertTrue(torch.equal(dataset[1]["action_preds"], expected))
        self.assertTrue(torch.equal(source.data["action_preds"][:len(original)], original))

    def test_seeded_loader_order_is_reproducible(self):
        def order(seed):
            loader = make_a0_dataloader(make_source(), "representation", task="sorting",
                                        batch_size=7, seed=seed)
            return [(int(r), int(t), int(y)) for b in loader
                    for r, t, y in zip(b["rollout_index"], b["timestep"], b["target"])]
        self.assertEqual(order(123), order(123))
        self.assertNotEqual(order(123), order(456))

    def test_actual_sorting_shapes_forward_backward(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            source = make_source(obs_dim=128, samples=32, horizon=8, action_dim=6)
            loader = make_a0_dataloader(source, "selection", task="sorting", batch_size=2)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["obs_embeddings"].shape), (2, 8, 128))
            self.assertEqual(tuple(batch["action_preds"].shape), (2, 8, 32, 8, 6))
            model = ObservationActionTemporalModel(
                obs_dim=128, action_horizon=8, action_dim=6,
                d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
            ).cpu()
            logits = model(batch["obs_embeddings"], batch["action_preds"], batch["padding_mask"])
            loss = torch.nn.BCEWithLogitsLoss()(logits, batch["target"])
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all().item()
                                for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
