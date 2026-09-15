import unittest

import torch

from evaluation.tac_fiper_a0_training import epoch_loss, rollout_threshold


class IdentityNormalizer:
    def transform(self, batch):
        return batch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, obs, action, mask):
        return self.weight.expand(obs.shape[0])


def batch(size, target):
    return dict(obs_embeddings=torch.zeros(size, 1), action_preds=torch.zeros(size, 1),
                padding_mask=torch.zeros(size, 1, dtype=torch.bool), target=torch.full((size,), float(target)))


class A0TrainingTest(unittest.TestCase):
    def test_threshold_uses_rollout_maxima_not_pooled_steps(self):
        value, maxima = rollout_threshold([(1, 0, 1.0), (1, 1, 3.0), (2, 0, 5.0)], [1, 2])
        self.assertAlmostEqual(value, 4.8)
        self.assertEqual(maxima, {1: 3.0, 2: 5.0})

    def test_threshold_rejects_missing_duplicate_nonfinite_and_wrong_role(self):
        for rows in ([(1, 0, 1.0)], [(1, 0, 1.0), (1, 0, 2.0), (2, 0, 3.0)],
                     [(1, 0, float("nan")), (2, 0, 3.0)], [(3, 0, 1.0), (2, 0, 3.0)]):
            with self.assertRaises(ValueError):
                rollout_threshold(rows, [1, 2])

    def test_selection_loss_is_sample_weighted_and_does_not_update(self):
        model = TinyModel()
        with torch.no_grad():
            model.weight.fill_(2.0)
        before = model.weight.clone()
        value = epoch_loss(model, [batch(3, 1), batch(1, 0)], IdentityNormalizer())
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.full((4,), 2.0), torch.tensor([1., 1., 1., 0.])).item()
        self.assertAlmostEqual(value, expected, places=6)
        self.assertTrue(torch.equal(before, model.weight))
        self.assertIsNone(model.weight.grad)
        self.assertFalse(model.training)

    def test_training_updates_parameters_and_rejects_nonfinite_loss(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        epoch_loss(model, [batch(2, 1)], IdentityNormalizer(), optimizer)
        self.assertGreater(model.weight.item(), 0)
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        with self.assertRaises(ValueError):
            epoch_loss(model, [batch(2, 1)], IdentityNormalizer())


if __name__ == "__main__":
    unittest.main()
