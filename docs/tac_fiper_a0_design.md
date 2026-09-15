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

The shifted action must never be taken from another rollout. Specifically,
rotate the valid action-history positions right by one within the current
window, keeping padding fixed. Do not rotate an entire episode or read
outside the current history. This preserves the action values in the window
while breaking their temporal alignment. Repeated/constant actions may still
produce identical negatives; A0 does not claim all synthetic pairs are separable.

Representation and selection both contain balanced aligned/shifted pairs.
At t=0 there is only one valid history position, so omit both members of that
pair. Threshold contains aligned samples at every timestep, including t=0.
A role containing only one-step rollouts has no train/selection pairs and
must fail explicitly. No success/failure label is used as the BCE target.

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

The constructor milestone is `datasets/tac_fiper_a0.py::SortingA0Dataset`:
pass an already loaded raw `ProcessedRolloutDataset`, a role, and `task="sorting"`.
Its fixed role indices mirror this design and the sorting smoke YAML. It
validates metadata before requesting selected episode tensors and passes
an explicit false normalization dictionary to the existing iterator.
It never loads files, fits statistics, calls the baseline normalizer, or
accesses test tensor slices. The source may already hold test tensors in
memory; controlling physical file reads belongs to the later loading stage.
Feature normalization and the CPU trainer remain separate future steps.

Samples contain float32 `obs_embeddings=[8,D]`, `action_preds=[8,S,P,A]`,
bool `padding_mask=[8]` (True means padding), a scalar float32 `target`,
and global `rollout_index` / episode-local `timestep` for audit and scoring.
`make_a0_dataloader` uses CPU, zero workers, and a private seeded generator;
only representation is shuffled. The constructor rejects non-CPU,
non-floating, non-finite, or incorrectly shaped inputs.

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

## Frozen Feature Normalization

`A0FeatureNormalizer.fit` accepts only the fixed representation dataset.
It reads original episode tensors once, including t=0, without padded
windows or duplicated positive/negative examples. Observation statistics
are per embedding feature over original timesteps. Action statistics are
per action channel over original timesteps, samples, and predicted horizon.
Use population standard deviation, float64 moment accumulation, and float32
stored parameters. Channels with std below 1e-6 use scale 1.

Transform uses `(x - mean) / scale` before the model's action mean/std
summary, then resets masked positions to zero. It never updates statistics.
Selection and threshold reuse these same frozen parameters. State exports
include fitting indices and row counts and can be stored in a checkpoint.

The real-data verifier uses a calibration-only in-memory view with all six
action columns: raw velocity plus its zero-origin integrated displacement.
The original metadata action_dim was 3, which would truncate augmented
columns through the existing iterator. Only the verifier's in-memory
action_dim is set to 6; saved metadata and baseline behavior are unchanged.
This is an explicit A0 input choice, not a claim of numerical equivalence
to the saved processed tensors or a fair baseline comparison already done.

## CPU Training Smoke

Run `python scripts/verify_sorting_a0_cpu.py --train-output /tmp/NEW_DIRECTORY`
with CUDA_VISIBLE_DEVICES empty. The default verifier remains forward-only.
The training option uses two full CPU epochs, seed 20260909, batch size 32,
Adam at 1e-3, gradient clipping at 1.0, and a small d_model=32, one-layer,
four-head Transformer with dropout=0. Only representation updates parameters;
sample-weighted selection BCE selects the checkpoint (earlier epoch wins ties).
No loss improvement is required for a pipeline PASS.

The selected model and frozen normalization are saved and restored on CPU.
Reloaded selection loss and first-batch outputs must match. Only then are
threshold samples constructed and scored using softplus(-logit). Window=1
ct_quantile is the ordinary linear q=0.9 quantile of the ten rollout maxima,
not a pooled timestep quantile or a finite-sample conformal guarantee.
Artifacts are best.pt, threshold_scores.json and manifest.json in a new
directory. The manifest records configuration, losses, roles and provenance.
These artifacts are pipeline checks, not failure-detection performance.
