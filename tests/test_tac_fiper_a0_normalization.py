import io
import unittest

import torch
from torch.utils.data import DataLoader

from datasets.tac_fiper_a0 import ROLE_INDICES, SortingA0Dataset
from datasets.tac_fiper_a0_normalization import A0FeatureNormalizer
from evaluation.temporal_models import ObservationActionTemporalModel


def raw_dataset(role="representation", constant=False):
    # Synthetic post-constructor episodes isolate normalization from file I/O.
    data = SortingA0Dataset.__new__(SortingA0Dataset)
    data.role, data.rollout_indices = role, ROLE_INDICES[role]
    data.episodes, data.samples = [], []
    for i, rollout_index in enumerate(data.rollout_indices):
        length = i % 3 + 2
        obs = torch.arange(length * 3, dtype=torch.float32).reshape(length, 3) + i * 10
        action = torch.arange(length * 2 * 2 * 6, dtype=torch.float32).reshape(length, 2, 2, 6) + i
        if constant:
            obs.fill_(5)
            action.fill_(7)
        data.episodes.append((obs, action))
        times = range(length) if role == "threshold" else range(1, length)
        targets = (1,) if role == "threshold" else (1, 0)
        data.samples.extend((i, rollout_index, t, y) for t in times for y in targets)
    return data


class A0NormalizationTest(unittest.TestCase):
    def test_statistics_match_unique_original_rows(self):
        data = raw_dataset()
        state = A0FeatureNormalizer.fit(data).state_dict()
        for prefix, column in (("obs", 0), ("action", 1)):
            rows = torch.cat([e[column].reshape(-1, e[column].shape[-1]) for e in data.episodes]).double()
            torch.testing.assert_close(state[prefix + "_mean"], rows.mean(0).float())
            torch.testing.assert_close(state[prefix + "_scale"], rows.std(0, unbiased=False).float())
            self.assertEqual(state[prefix + "_count"], len(rows))
        self.assertEqual(state["obs_count"], 90)  # includes t=0, no window/negative duplication

    def test_rejects_selection_threshold_and_changed_indices(self):
        for role in ("selection", "threshold"):
            data = raw_dataset(role)
            del data.episodes  # rejection must precede reading episodes
            with self.assertRaisesRegex(ValueError, "representation"):
                A0FeatureNormalizer.fit(data)
        data = raw_dataset()
        data.rollout_indices = ROLE_INDICES["selection"]
        with self.assertRaises(ValueError):
            A0FeatureNormalizer.fit(data)

    def test_constant_features_and_padding_stay_zero(self):
        data = raw_dataset(constant=True)
        normalizer = A0FeatureNormalizer.fit(data)
        for scale in ("obs_scale", "action_scale"):
            self.assertTrue((normalizer.state_dict()[scale] == 1).all().item())
        for index in (0, 1):
            result = normalizer.transform(data[index])
            self.assertEqual(torch.count_nonzero(result["obs_embeddings"]).item(), 0)
            self.assertEqual(torch.count_nonzero(result["action_preds"]).item(), 0)

    def test_holdout_transform_does_not_refit_or_mutate_inputs(self):
        normalizer = A0FeatureNormalizer.fit(raw_dataset())
        before = normalizer.state_dict()
        for role in ("selection", "threshold"):
            batch = next(iter(DataLoader(raw_dataset(role), batch_size=4)))
            batch["obs_embeddings"] += 10000
            original = batch["obs_embeddings"].clone()
            result = normalizer.transform(batch)
            self.assertTrue(torch.equal(batch["obs_embeddings"], original))
            self.assertTrue(torch.equal(result["target"], batch["target"]))
            valid = ~batch["padding_mask"]
            expected = (original[valid] - before["obs_mean"]) / before["obs_scale"]
            torch.testing.assert_close(result["obs_embeddings"][valid], expected)
            self.assertEqual(torch.count_nonzero(result["obs_embeddings"][~valid]).item(), 0)
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, normalizer.state_dict()[key]))

    def test_checkpoint_roundtrip_and_no_state_aliasing(self):
        data = raw_dataset()
        normalizer = A0FeatureNormalizer.fit(data)
        buffer = io.BytesIO()
        torch.save(normalizer.state_dict(), buffer)
        buffer.seek(0)
        restored = A0FeatureNormalizer.from_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
        for key in ("obs_embeddings", "action_preds"):
            self.assertTrue(torch.equal(normalizer.transform(data[0])[key], restored.transform(data[0])[key]))
        exported = normalizer.state_dict()
        exported["obs_mean"].fill_(999)
        exported["rollout_indices"][0] = 999
        self.assertNotEqual(normalizer.state_dict()["rollout_indices"][0], 999)
        self.assertFalse((normalizer.state_dict()["obs_mean"] == 999).all().item())

    def test_rejects_corrupted_state_and_nonfinite_data(self):
        data = raw_dataset()
        normalizer = A0FeatureNormalizer.fit(data)
        for problem in ("role", "scale", "nan"):
            state = normalizer.state_dict()
            if problem == "role":
                state["fit_role"] = "selection"
            elif problem == "scale":
                state["action_scale"][0] = 0
            else:
                state["obs_mean"][0] = float("nan")
            with self.assertRaises(ValueError):
                A0FeatureNormalizer.from_state_dict(state)
        data.episodes[0][0][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            A0FeatureNormalizer.fit(data)
        batch = raw_dataset()[0]
        batch["action_preds"][0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            normalizer.transform(batch)

    def test_normalized_batch_has_finite_loss_and_gradients(self):
        data = raw_dataset()
        batch = A0FeatureNormalizer.fit(data).transform(next(iter(DataLoader(data, batch_size=4))))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = ObservationActionTemporalModel(3, 2, 6, d_model=16, nhead=2,
                                                   num_layers=1, dim_feedforward=32).cpu()
            logits = model(batch["obs_embeddings"], batch["action_preds"], batch["padding_mask"])
            loss = torch.nn.BCEWithLogitsLoss()(logits, batch["target"])
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))


if __name__ == "__main__":
    unittest.main()
