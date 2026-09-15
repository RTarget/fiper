"""CPU-only, role-isolated sorting A0 samples; no fitting or file writes."""

from numbers import Integral
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


HISTORY = 8
ALLOCATION_SEED = 20260909
ROLE_INDICES = MappingProxyType({
    "representation": (
        0, 1, 7, 8, 9, 11, 13, 16, 20, 22, 23, 24, 25, 28, 29,
        30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 49,
    ),
    "selection": (3, 6, 10, 12, 21, 27, 31, 43, 47, 48),
    "threshold": (2, 4, 5, 14, 15, 17, 18, 19, 26, 33),
})
TENSOR_KEYS = ("obs_embeddings", "action_preds")


def _validate_metadata(metadata):
    """Check all roles and episode boundaries before requesting any tensors."""
    n = metadata.get("num_rollouts")
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 50:
        raise ValueError("num_rollouts must be an integer >= 50")
    masks = {}
    for name in ("calibration", "test", "successful", "failed"):
        mask = np.asarray(metadata.get(name + "_rollout_labels"))
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError(f"{name} labels must be a bool array of length {n}")
        masks[name] = mask
    calibration, test = masks["calibration"], masks["test"]
    successful, failed = masks["successful"], masks["failed"]
    if np.any(calibration & test):
        raise ValueError("calibration and test overlap")
    if not np.all(calibration | test):
        raise ValueError("calibration and test must cover all rollouts")
    if np.any(successful & failed) or not np.all(successful | failed):
        raise ValueError("successful and failed labels must partition rollouts")
    all_indices = [i for values in ROLE_INDICES.values() for i in values]
    if len(all_indices) != 50 or len(set(all_indices)) != 50:
        raise ValueError("A0 roles must be disjoint and cover 50 rollouts")
    for i in all_indices:
        if test[i]:
            raise ValueError(f"rollout {i} belongs to test")
        if not calibration[i]:
            raise ValueError(f"rollout {i} is not calibration")
        if not successful[i]:
            raise ValueError(f"rollout {i} is not successful calibration")
    if set(np.flatnonzero(calibration)) != set(all_indices):
        raise ValueError("fixed A0 roles must cover exactly the calibration set")

    boundaries = []
    for key in ("episode_start_indices", "episode_end_indices"):
        values = np.asarray(metadata.get(key))
        if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{key} must be an integer array of length {n}")
        boundaries.append(values)
    starts, ends = boundaries
    if (np.any(starts < 0) or np.any(ends <= starts)
            or np.any(starts[1:] < ends[:-1])):
        raise ValueError("episode boundaries must be positive-length and nonoverlapping")
    steps = metadata.get("num_steps")
    if (isinstance(steps, bool) or not isinstance(steps, Integral)
            or steps < int(ends[-1])):
        raise ValueError("num_steps does not cover episode boundaries")
    return ends - starts


def _check_tensor(value, key, ndim, length):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a torch tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{key} must be on CPU")
    if not value.is_floating_point():
        raise ValueError(f"{key} must be floating point")
    if value.ndim != ndim or value.shape[0] != length or min(value.shape) <= 0:
        shape = "[T,D]" if ndim == 2 else "[T,S,P,A]"
        raise ValueError(f"{key} must have shape {shape} with metadata length {length}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{key} must be finite")
    result = value.detach().to(dtype=torch.float32).clone()
    if not torch.isfinite(result).all().item():
        raise ValueError(f"{key} must remain finite in float32")
    return result


class SortingA0Dataset(Dataset):
    """Build samples from an already loaded ProcessedRolloutDataset.

    Caller must supply the actual task name and raw, unnormalized tensors.
    This adapter never loads files or fits/uses the baseline normalizer.
    Only fixed successful calibration roles are accessible through this API.
    Test tensors already resident in the source are neither sliced nor scored.

    Representation/selection: one aligned and one shifted sample per t >= 1.
    Threshold: one aligned sample at every t, including t == 0.
    Negatives rotate only valid positions in the current window to the right
    by one. No full-episode wraparound or future-window access is used.
    """

    def __init__(self, source, role, *, task):
        if task != "sorting":
            raise ValueError("A0 supports only sorting development smoke")
        if role not in ROLE_INDICES:
            raise ValueError("role must be representation, selection, or threshold")
        lengths = _validate_metadata(source.get_metadata())
        self.role = role
        self.rollout_indices = ROLE_INDICES[role]
        self.episodes = []
        self.samples = []
        feature_shape = None
        for rollout_index in self.rollout_indices:
            # One explicit global index per call avoids relying on iterator order.
            episodes = list(source.iterate_episodes(
                subset="calibration",
                rollout_indices=[rollout_index],
                required_tensors=list(TENSOR_KEYS),
                optional_tensors=[],
                required_actions="all",
                optional_actions=[],
                normalize_tensors={key: False for key in TENSOR_KEYS},
                history=0,
                with_success_labels=False,
            ))
            if len(episodes) != 1:
                raise ValueError("expected exactly one episode per global rollout index")
            episode = episodes[0]
            length = int(lengths[rollout_index])
            obs = _check_tensor(episode.get("obs_embeddings"), "obs_embeddings", 2, length)
            action = _check_tensor(episode.get("action_preds"), "action_preds", 4, length)
            shape = (tuple(obs.shape[1:]), tuple(action.shape[1:]))
            if feature_shape is not None and shape != feature_shape:
                raise ValueError("all role episodes must have matching feature shapes")
            feature_shape = shape
            episode_index = len(self.episodes)
            self.episodes.append((obs, action))
            if role == "threshold":
                self.samples.extend((episode_index, rollout_index, t, 1) for t in range(length))
            else:
                self.samples.extend(
                    (episode_index, rollout_index, t, target)
                    for t in range(1, length) for target in (1, 0)
                )
        if not self.samples:
            raise ValueError("role has no usable A0 samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_index, rollout_index, t, target = self.samples[index]
        obs, action = self.episodes[episode_index]
        start = max(0, t - HISTORY + 1)
        valid_length = t - start + 1
        offset = HISTORY - valid_length
        obs_history = obs.new_zeros((HISTORY, *obs.shape[1:]))
        action_history = action.new_zeros((HISTORY, *action.shape[1:]))
        obs_history[offset:] = obs[start:t + 1]
        valid_actions = action[start:t + 1]
        if target == 0:
            valid_actions = torch.cat((valid_actions[-1:], valid_actions[:-1]), dim=0)
        action_history[offset:] = valid_actions
        mask = torch.arange(HISTORY) < offset
        return {
            "obs_embeddings": obs_history,
            "action_preds": action_history,
            "padding_mask": mask,
            "target": torch.tensor(float(target), dtype=torch.float32),
            "rollout_index": rollout_index,
            "timestep": t,
        }


def make_a0_dataloader(source, role, *, task, batch_size=16, seed=ALLOCATION_SEED):
    """CPU batches; only representation is shuffled, using a local RNG."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    dataset = SortingA0Dataset(source, role, task=task)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=(role == "representation"),
        num_workers=0, pin_memory=False, generator=generator,
    )
