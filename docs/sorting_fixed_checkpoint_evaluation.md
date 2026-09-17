# Sorting fixed-checkpoint exploratory evaluation

This independent entry point permits scoring the 400 metadata-defined test
rollouts after model selection and threshold fitting have finished. It does
not change the smoke protocol or grant test access to its constructor/trainer.
It evaluates the existing two-epoch, seed-20260909, epoch-2 checkpoint with
its saved normalization and q=.9 threshold, history=8, window=1, patience=0.
No options exist to tune epochs, scores, threshold, window, or feature choice.

Run on CPU (two threads; no GPU reservation needed):

```
python scripts/evaluate_sorting_a0_fixed.py --training-dir /tmp/tac_a0_gpu_train_01 --output /tmp/tac_sorting_fixed_eval_01
```

Output must be a new directory. Frozen artifact/config hashes are written
before reading test payloads. All episode scores, data hashes, first alarms
and metrics are retained. Training artifacts are read-only. Raw 3-D actions
become six columns by concatenating zero-origin integrated displacement at
ts=.035, exactly as the calibration-only training loader. CPU inference may
have tiny differences from GPU; it does not refit the GPU-calibrated threshold.

Metrics preserve the repository definitions: score strictly greater than
threshold, any crossing triggers, and detected-failure time is the zero-based
first crossing divided by (maximum episode length over ALL metadata - 1),
not by that rollout's length. No detected failure means Detection Time=1.
Accuracy and TWA reuse repository helper functions; confusion counts avoid
the legacy np.bool alias without modifying existing baseline code.

This is a TAC-only exploratory evaluation, not a superiority experiment.
The saved original-protocol FiPER models and five-task summary are not a
matched comparison: original RND preprocessing/training consumes the entire
calibration set and uses timestep-based validation. A strict comparison needs
RND-OE representation/selection isolation, representation-only ACE grid fit,
threshold-only calibration, fixed combination settings, and a documented
RND training budget. These adaptations are not implemented in this patch.
Do not report a gain until that matched comparison exists.

The test set was previously used for baseline reproduction. If these results
guide subsequent method changes, label the work exploratory and obtain
independent confirmation; do not call repeated test-guided tuning unseen-test
validation. Current TAC A0 is not the full proposed TAC-FIPER method.
