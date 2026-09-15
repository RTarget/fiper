"""Frozen CPU z-score statistics fitted on raw representation episodes only.

Each original observation timestep counts once (including t=0). Action
statistics pool time, generated samples and horizon, separately per channel.
Neither padded windows nor synthetic negatives contribute to fitting.
Population variance is accumulated in float64; std < 1e-6 uses scale 1.
Transform resets padding to zero and never updates statistics or inputs.
"""

from numbers import Integral

import torch

from datasets.tac_fiper_a0 import ROLE_INDICES, SortingA0Dataset


def _moments(episodes, column):
    count, mean, squared = 0, None, None
    for episode in episodes:
        value = episode[column]
        ndim = 2 if column == 0 else 4
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or not value.is_floating_point() or value.ndim != ndim
                or min(value.shape) < 1 or not torch.isfinite(value).all().item()):
            raise ValueError("normalization requires finite floating CPU episode tensors")
        rows = value.detach().double().reshape(-1, value.shape[-1])
        n = len(rows)
        local_mean = rows.mean(0)
        local_squared = ((rows - local_mean) ** 2).sum(0)
        if mean is None:
            mean, squared, count = local_mean, local_squared, n
        else:
            if mean.shape != local_mean.shape:
                raise ValueError("feature dimensions must match across episodes")
            delta = local_mean - mean
            total = count + n
            squared = squared + local_squared + delta.square() * (count * n / total)
            mean = mean + delta * (n / total)
            count = total
    if not count:
        raise ValueError("cannot fit empty episodes")
    std = (squared / count).clamp_min(0).sqrt()
    scale = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.float(), scale.float(), count


class A0FeatureNormalizer:
    """Fit once, then transform samples/batches or restore from checkpoint state."""

    def __init__(self, state):
        required = {"version", "fit_role", "rollout_indices", "obs_count", "action_count",
                    "obs_mean", "obs_scale", "action_mean", "action_scale"}
        if set(state) != required or state["version"] != 1 or state["fit_role"] != "representation":
            raise ValueError("invalid A0 normalization state")
        if tuple(state["rollout_indices"]) != ROLE_INDICES["representation"]:
            raise ValueError("normalization must originate from fixed representation indices")
        for name in ("obs_count", "action_count"):
            if isinstance(state[name], bool) or not isinstance(state[name], Integral) or state[name] < 1:
                raise ValueError("normalization counts must be positive integers")
        self._state = {}
        for prefix in ("obs", "action"):
            for suffix in ("mean", "scale"):
                name = prefix + "_" + suffix
                value = state[name]
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.dtype != torch.float32 or value.ndim != 1 or value.numel() < 1
                        or not torch.isfinite(value).all().item()):
                    raise ValueError(f"{name} must be a finite float32 CPU vector")
                if suffix == "scale" and not (value > 0).all().item():
                    raise ValueError("normalization scales must be positive")
                self._state[name] = value.detach().clone()
            if self._state[prefix + "_mean"].shape != self._state[prefix + "_scale"].shape:
                raise ValueError("mean and scale dimensions must match")
        self._state.update(version=1, fit_role="representation",
                           rollout_indices=list(ROLE_INDICES["representation"]),
                           obs_count=int(state["obs_count"]), action_count=int(state["action_count"]))

    @classmethod
    def fit(cls, dataset):
        if (not isinstance(dataset, SortingA0Dataset) or dataset.role != "representation"
                or tuple(dataset.rollout_indices) != ROLE_INDICES["representation"]):
            raise ValueError("fit accepts only the fixed representation SortingA0Dataset")
        if len(dataset.episodes) != 30:
            raise ValueError("fit requires all 30 representation episodes")
        for obs, action in dataset.episodes:
            if obs.shape[0] != action.shape[0]:
                raise ValueError("observation and action episode lengths must match")
        obs_mean, obs_scale, obs_count = _moments(dataset.episodes, 0)
        action_mean, action_scale, action_count = _moments(dataset.episodes, 1)
        return cls(dict(version=1, fit_role="representation",
                        rollout_indices=list(dataset.rollout_indices),
                        obs_count=obs_count, action_count=action_count,
                        obs_mean=obs_mean, obs_scale=obs_scale,
                        action_mean=action_mean, action_scale=action_scale))

    def state_dict(self):
        """Return independent tensors and plain metadata for torch.save checkpoints."""
        return {key: value.clone() if isinstance(value, torch.Tensor)
                else list(value) if isinstance(value, list) else value
                for key, value in self._state.items()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state)

    def transform(self, batch):
        """Accept unbatched [H,D]/[H,S,P,A] or batched [B,H,D]/[B,H,S,P,A]."""
        obs, action, mask = (batch[key] for key in ("obs_embeddings", "action_preds", "padding_mask"))
        for name, value in (("observations", obs), ("actions", action)):
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != torch.float32 or not torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite float32 CPU tensors")
        if (obs.ndim not in (2, 3) or action.ndim != obs.ndim + 2
                or obs.shape[:-1] != action.shape[:-3]
                or min(obs.shape) < 1 or min(action.shape) < 1):
            raise ValueError("observation and action batch/history shapes must match")
        if (not isinstance(mask, torch.Tensor) or mask.device.type != "cpu"
                or mask.dtype != torch.bool or mask.shape != obs.shape[:-1]
                or mask.all(dim=-1).any().item()):
            raise ValueError("invalid padding mask")
        if (obs.shape[-1] != self._state["obs_mean"].numel()
                or action.shape[-1] != self._state["action_mean"].numel()):
            raise ValueError("feature dimensions differ from fitted statistics")
        obs_z = (obs - self._state["obs_mean"]) / self._state["obs_scale"]
        action_z = (action - self._state["action_mean"]) / self._state["action_scale"]
        obs_z = obs_z.masked_fill(mask.unsqueeze(-1), 0)
        action_z = action_z.masked_fill(mask[..., None, None, None], 0)
        if not torch.isfinite(obs_z).all().item() or not torch.isfinite(action_z).all().item():
            raise ValueError("normalized features are not finite")
        return dict(batch, obs_embeddings=obs_z, action_preds=action_z)
