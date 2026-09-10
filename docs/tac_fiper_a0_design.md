# TAC-FIPER A0 CPU-Only Design

## Scope

A0 is a pipeline-validation milestone for sorting only. It is not a
paper-performance experiment and must not use test rollouts.

The current temporal model is treated as a consistency-score backbone. A0
does not claim that the complete TAC-FIPER method is implemented.

## Fixed Sorting Allocation

The allocation seed is `20260909`.

```text
representation:
[0, 1, 7, 8, 9, 11, 13, 16, 20, 22, 23, 24, 25, 28, 29,
 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 49]

selection:
[3, 6, 10, 12, 21, 27, 31, 43, 47, 48]

threshold:
[2, 4, 5, 14, 15, 17, 18, 19, 26, 33]
```

All three roles must be checked against metadata before loading tensors:

1. Every index must belong to `subset="calibration"`.
2. Every index must have `successful_rollout_labels[index] == True`.
3. The three lists must be pairwise disjoint.
4. Their union must contain exactly 50 calibration rollouts.
5. Any index marked as test must be rejected explicitly.

## Training Objective

The training signal is normal observation-action consistency and does not
use failure labels.

For a timestep history from one successful representation rollout:

- The positive example pairs the observation history with the action
  prediction history from the same rollout and timestep.
- The negative example keeps the observation history fixed and applies a
  deterministic one-step action-history shift within the same rollout.

The shifted action must never be taken from another rollout. This preserves
the marginal action distribution while breaking temporal alignment.

The current scalar output is interpreted as a consistency logit `z`.

```text
positive target: 1
negative target: 0
loss: BCEWithLogitsLoss(z, target)
anomaly score: softplus(-z)
```

The A0 history length is fixed to 8. History construction must remain
episode-local and must provide a padding mask. No history window may cross
an episode boundary.

## Data Pipeline

The pipeline must use the existing `ProcessedRolloutDataset` interface:

1. Load the sorting processed rollout dataset.
2. Call `iterate_episodes` with `subset="calibration"` and explicit role
   indices.
3. Build history windows separately inside each returned rollout.
4. Fit feature normalization statistics on representation rollouts only.
5. Apply those frozen statistics to selection and threshold rollouts.
6. Create positive and shifted-negative examples for representation data.
7. Return a CPU `DataLoader` with observations, action predictions, padding
   masks, and binary consistency targets.

The implementation must not call `subset="test"` and must not infer roles
from filenames. Metadata rollout indices are the only valid identity.

## Role Contracts

Representation rollouts are the only data used to fit normalization
statistics and optimize model parameters.

Selection rollouts are used only after each epoch to compute consistency
loss and choose the best checkpoint. They must not update model parameters,
normalization statistics, or the anomaly threshold.

Threshold rollouts are used only after checkpoint selection. Their anomaly
scores use `softplus(-z)`, and the development-smoke threshold uses
`ct_quantile` at `q=0.9`. This is a false-alarm calibration boundary, not a
failure-detection performance result.

Test rollouts are inaccessible to the A0 constructor, trainer, checkpoint
selector, and threshold fitter.

## CPU Dry-Run Acceptance Criteria

- The device is explicitly `cpu`.
- The deterministic seed is recorded.
- Two short epochs are sufficient for the dry run.
- All losses and scores are finite.
- A checkpoint can be saved and reloaded.
- Selection loss alone determines the selected checkpoint.
