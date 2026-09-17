"""Read-only scoring helpers for fixed sorting checkpoint evaluation."""
import math

import numpy as np
import torch

from evaluation.utils import _calculate_accuracy, _calculate_twa


def aligned_windows(obs, actions):
    """All t including t=0, matching the training constructor's positive windows."""
    if obs.ndim != 2 or actions.ndim != 4 or len(obs) != len(actions) or len(obs) < 1:
        raise ValueError("invalid episode shapes")
    if not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
        raise ValueError("non-finite episode")
    length = len(obs)
    observations = obs.new_zeros((length, 8, obs.shape[-1]))
    predictions = actions.new_zeros((length, 8, *actions.shape[1:]))
    mask = torch.ones((length, 8), dtype=torch.bool)
    for t in range(length):
        start = max(0, t - 7)
        valid = t - start + 1
        observations[t, -valid:] = obs[start:t + 1]
        predictions[t, -valid:] = actions[start:t + 1]
        mask[t, -valid:] = False
    return dict(obs_embeddings=observations, action_preds=predictions, padding_mask=mask)


def score_episode(model, normalizer, obs, actions):
    windows = aligned_windows(obs, actions)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(obs), 32):
            batch = normalizer.transform({k: v[start:start + 32] for k, v in windows.items()})
            logits = model(batch["obs_embeddings"], batch["action_preds"], batch["padding_mask"])
            values = torch.nn.functional.softplus(-logits)
            if not torch.isfinite(values).all():
                raise ValueError("non-finite scores")
            scores.extend(values.tolist())
    return scores


def fixed_metrics(episodes, threshold, max_episode_length):
    """Repo definitions: strict >, patience=0, time index/(global max length-1).

    Reuse repo accuracy and TWA functions. Compute confusion directly to avoid
    the legacy np.bool alias in its wrapper; do not mutate NumPy or baseline.
    """
    if not math.isfinite(threshold) or threshold <= 0 or max_episode_length < 2 or not episodes:
        raise ValueError("invalid metric inputs")
    detected, successful, times, first_steps = [], [], [], []
    for episode in episodes:
        values = np.asarray(episode["scores"], dtype=float)
        if values.ndim != 1 or not len(values) or len(values) > max_episode_length or not np.isfinite(values).all():
            raise ValueError("invalid episode scores")
        success = episode["successful"]
        if not isinstance(success, (bool, np.bool_)):
            raise ValueError("success must be boolean")
        hits = np.flatnonzero(values > threshold)
        alarm = bool(len(hits))
        first = int(hits[0]) if alarm else None
        first_steps.append(first)
        detected.append(alarm)
        successful.append(bool(success))
        if alarm and not success:
            times.append(first / (max_episode_length - 1))
    alarm, success = np.asarray(detected), np.asarray(successful)
    if not success.any() or success.all():
        raise ValueError("fixed evaluation requires both success and failure rollouts")
    tp, tn = int((alarm & ~success).sum()), int((~alarm & success).sum())
    fp, fn = int((alarm & success).sum()), int((~alarm & ~success).sum())
    tpr, tnr, accuracy, ba = _calculate_accuracy(tp, tn, fp, fn)
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, TPR=tpr, TNR=tnr, accuracy=accuracy,
                balanced_accuracy=ba, TWA=float(_calculate_twa(detected, success, times)),
                avg_detection_time=float(np.mean(times)) if times else 1.0,
                std_detection_time=float(np.std(times)) if times else 0.0), first_steps
