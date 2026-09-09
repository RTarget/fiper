# TAC-FIPER Calibration Usage Audit

Audit date: 2026-09-09

## Scope

This source-code audit records how the original baseline implementation
uses calibration rollouts during preprocessing, training, model
selection, and threshold estimation.

It does not report TAC-FIPER performance or freeze the draft TAC-FIPER
allocation.

## Relevant dataset fact

PushT calibration contains:

- 21 successful rollouts
- 29 failed rollouts

Calling `get_subset(subset="calibration")` therefore returns both
successful and failed PushT calibration rollouts unless an additional
successful-rollout filter is applied.

No data was added, relabeled, or modified during this audit. The
failed-calibration inclusion described below comes from the combination
of the original dataset split and the original implementation.

## Baseline findings

### RND-OE and RND-A

`rnd/rnd_trainer.py` requests the complete calibration subset without a
successful-rollout filter.

When validation is enabled, `_split_datasets()` randomly divides the
concatenated timestep tensors using a 90/10 ratio. This is a timestep
split, not a rollout split. Validation loss is used for early stopping
and best-model selection.

Consequences under PushT:

- failed calibration enters training and model selection;
- timesteps from one rollout may appear on both sides of the split.

`evaluation/method_eval_classes/rnd_eval.py` only loads and evaluates
the resulting checkpoint.

### Entropy / ACE

`evaluation/method_eval_classes/entropy_eval.py` reads complete
calibration action predictions and derives grid cell sizes from their
ranges.

It has no separately trained neural model, but failed PushT calibration
affects its preprocessing statistics.

### Similarity

`evaluation/method_eval_classes/similarity_eval.py` reads complete
calibration observation embeddings.

Those embeddings are used as the reference database and to fit
statistics such as covariance, PCA, and clustering. Failed PushT
calibration therefore affects the fitted reference distribution.

### Temporal consistency

`evaluation/method_eval_classes/tc_eval.py` computes each score from
overlapping action predictions within the current rollout.

No learned model or calibration-fitted preprocessing was found. It still
depends on the common threshold procedure.

### LogPZO

`evaluation/method_eval_classes/logpzo_eval.py` trains on complete
calibration observation embeddings.

It randomly splits concatenated embedding timesteps into 90% training
and 10% validation data. Validation loss selects `best_model`.

It also replaces the configured device with `cuda` whenever CUDA is
available. It must not be run on the shared server until device
selection is made explicit and GPU availability is checked again.

### Logical combinations

RND/Entropy logical combinations add no new learned model. They inherit
the training, preprocessing, and calibration behavior of their
components.

## Threshold behavior

`BaseEvalClass._get_thresholds()` filters calibration episode scores by
the successful label before computing final thresholds.

Therefore failed calibration does not enter the current final threshold
calculation. This filtering does not prevent failed calibration from
entering earlier method-specific training or preprocessing.

## Reporting decision

The reproduced baseline remains valid as a reproduction of the authors'
implementation and original split. It must be reported as an
original-protocol result.

It must not be presented as a result under the TAC-FIPER strict
protocol.

## Requirements for a strict-protocol rerun

A fair strict rerun must satisfy all of the following:

1. Every split and role is defined by metadata rollout index.
2. Representation and preprocessing use only successful calibration.
3. Model selection uses only successful calibration.
4. Threshold estimation uses only successful calibration.
5. Required roles are pairwise disjoint at rollout level.
6. No timestep from one rollout appears in multiple roles.
7. Test rollouts are used only for final evaluation.
8. `calibration_unused` is not used to fill missing capacity.
9. PushT failed calibration is excluded from TAC training,
   preprocessing, selection, pseudo-negative construction, and
   threshold estimation.
10. All compared baselines are rerun using the same frozen allocation.
11. RND and LogPZO replace timestep splitting with rollout-level roles.
12. Entropy and Similarity fit statistics only on their assigned
    successful preprocessing rollouts.
13. Logical combinations are recomputed from strictly rerun components.

## Unresolved decision

Exact per-task allocation remains unresolved in
`configs/eval/tac_fiper_protocol.yaml`.

This is especially important for PushT with 21 successful calibration
rollouts, and Pretzel and PushChair with 10 each.

This audit establishes no split-conformal, cross-conformal, CV+, or
finite-sample guarantee. Model implementation and strict reruns remain
blocked until the allocation and calibration method are justified and
recorded.
