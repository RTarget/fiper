import unittest

import torch

from evaluation.temporal_models import (
    ObservationActionTemporalModel,
    summarize_action_predictions,
)


class TemporalModelsTest(unittest.TestCase):
    def test_action_summary_supports_variable_sample_counts(self):
        for sample_count in (30, 32, 256):
            action_preds = torch.randn(
                4,
                sample_count,
                8,
                6,
            )
            summary = summarize_action_predictions(action_preds)
            self.assertEqual(summary.shape, (4, 8 * 6 * 2))
            self.assertTrue(torch.isfinite(summary).all())

    def test_single_step_forward_backward(self):
        model = ObservationActionTemporalModel(
            obs_dim=128,
            action_horizon=8,
            action_dim=6,
        )
        obs = torch.randn(4, 128)
        actions = torch.randn(4, 32, 8, 6)

        output = model(obs, actions)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(torch.isfinite(output).all())

        output.mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in gradients))

    def test_temporal_forward_with_padding_mask(self):
        model = ObservationActionTemporalModel(
            obs_dim=64,
            action_horizon=16,
            action_dim=3,
            max_history=8,
        )
        obs = torch.randn(3, 5, 64)
        actions = torch.randn(3, 5, 256, 16, 3)
        padding_mask = torch.tensor(
            [
                [False, False, False, False, False],
                [True, True, False, False, False],
                [False, False, False, True, True],
            ]
        )

        output = model(
            obs,
            actions,
            padding_mask=padding_mask,
        )
        self.assertEqual(output.shape, (3,))
        self.assertTrue(torch.isfinite(output).all())

        output.sum().backward()

    def test_supports_different_task_dimensions(self):
        task_shapes = (
            (640, 8, 21, 32),
            (512, 16, 5, 30),
            (96, 16, 3, 256),
        )

        for obs_dim, horizon, action_dim, sample_count in task_shapes:
            model = ObservationActionTemporalModel(
                obs_dim=obs_dim,
                action_horizon=horizon,
                action_dim=action_dim,
            )
            obs = torch.randn(2, obs_dim)
            actions = torch.randn(
                2,
                sample_count,
                horizon,
                action_dim,
            )
            output = model(obs, actions)
            self.assertEqual(output.shape, (2,))
            self.assertTrue(torch.isfinite(output).all())

    def test_rejects_all_padding(self):
        model = ObservationActionTemporalModel(
            obs_dim=16,
            action_horizon=4,
            action_dim=3,
        )
        obs = torch.randn(2, 3, 16)
        actions = torch.randn(2, 3, 30, 4, 3)
        padding_mask = torch.ones(2, 3, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "at least one"):
            model(obs, actions, padding_mask=padding_mask)


if __name__ == "__main__":
    unittest.main()