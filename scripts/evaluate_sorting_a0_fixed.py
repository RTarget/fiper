"""Fixed-checkpoint, CPU-only exploratory test evaluation; never train/refit.

An independent entry point: the development-smoke protocol stays unchanged.
Writes frozen inputs before opening any test rollout, then scores all 400.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from datasets.tac_fiper_a0 import ROLE_INDICES, _validate_metadata
from datasets.tac_fiper_a0_normalization import A0FeatureNormalizer
from evaluation.temporal_models import ObservationActionTemporalModel
from evaluation.tac_sorting_fixed_eval import score_episode, fixed_metrics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = args.training_dir.resolve(strict=True)
    output = args.output.resolve()
    require(not output.exists(), "Output must be a NEW directory; preserve earlier evaluations")
    require(not output.is_relative_to((REPO / "data").resolve()), "Output cannot be inside data")
    require(not output.is_relative_to(training), "Output cannot be inside training artifacts")
    checkpoint = training / "best.pt"
    manifest_path = training / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "PASS" and manifest["scope"] == "sorting_development_smoke_only",
            "Expected successful sorting training smoke")
    require(manifest["seed"] == 20260909 and manifest["epochs"] == 2 and manifest["history_length"] == 8,
            "Training settings differ from the predeclared first evaluation")
    require(manifest["selected_epoch"] == 2 and manifest["performance_evaluated"] is False,
            "Expected the previously selected epoch-2 checkpoint")
    require(manifest["roles"] == {k: list(v) for k, v in ROLE_INDICES.items()}, "Role allocation changed")
    settings = manifest["threshold"]
    require(settings["style"] == "ct_quantile" and settings["quantile"] == 0.9
            and settings["window"] == 1 and settings["aggregation"] == "per_rollout_max",
            "Unexpected frozen threshold settings")
    threshold = float(settings["value"])
    require(math.isfinite(threshold) and threshold > 0, "Threshold must be finite and positive")
    maxima = settings["rollout_maxima"]
    require(set(maxima) == {str(i) for i in ROLE_INDICES["threshold"]}, "Threshold roles mismatch")
    require(math.isclose(float(np.quantile(list(maxima.values()), .9)), threshold, rel_tol=1e-10),
            "Saved threshold does not match saved rollout maxima")
    metadata_path = REPO / "data/sorting/processed_rollouts/metadata.pkl"
    require(digest(metadata_path) == manifest["provenance"]["metadata_sha256"], "Metadata changed since training")
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    _validate_metadata(metadata)
    test_indices = np.flatnonzero(metadata["test_rollout_labels"])
    require(np.array_equal(test_indices, np.arange(50, 450)), "Unexpected sorting test global indices")
    labels = np.asarray(metadata["successful_rollout_labels"])[test_indices]
    require(int(labels.sum()) == 255 and int((~labels).sum()) == 145, "Unexpected sorting test label counts")
    names = list(metadata["rollout_filenames"]["test"])
    require(len(names) == 400 and len(set(names)) == 400 and names == sorted(names), "Test filename identity mismatch")
    ts = float(manifest["provenance"]["integration_ts"])
    require(math.isclose(ts, .035) and manifest["provenance"]["position_origin"] == "zero_no_states",
            "Unexpected action preprocessing")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["epoch"] == manifest["selected_epoch"] and saved["seed"] == manifest["seed"],
            "Checkpoint and manifest mismatch")
    require(saved["model_config"] == manifest["model_config"]
            and math.isclose(saved["selection_loss"], manifest["selected_loss"], rel_tol=1e-10),
            "Checkpoint configuration/loss mismatch")
    require(saved["model_config"]["obs_dim"] == 128 and saved["model_config"]["action_dim"] == 6
            and saved["model_config"]["action_horizon"] == 8, "Unexpected input dimensions")
    torch.set_num_threads(2)
    torch.manual_seed(20260909)
    torch.use_deterministic_algorithms(True)
    model = ObservationActionTemporalModel(**saved["model_config"]).cpu().eval()
    model.load_state_dict(saved["model"])
    normalizer = A0FeatureNormalizer.from_state_dict(saved["normalization"])
    require(normalizer.state_dict()["obs_count"] == 1197, "Unexpected normalization provenance")
    max_length = int(max(metadata["episode_lengths"]))  # same denominator as BaseEvalClass
    test_root = REPO / "data/sorting/rollouts/test"
    require(not test_root.is_symlink(), "Test directory must not redirect elsewhere")
    test_root = test_root.resolve(strict=True)
    paths = []
    for name in names:
        require(isinstance(name, str) and Path(name).name == name and name.endswith(".pkl"), "Invalid filename")
        path = (test_root / name).resolve(strict=True)
        require(path.parent == test_root, "Test path escapes the test directory")
        paths.append(path)
    output.mkdir(parents=True, exist_ok=False)
    frozen = dict(scope="sorting_fixed_checkpoint_exploratory_evaluation", method="TAC_A0",
                  checkpoint_sha256=digest(checkpoint), training_manifest_sha256=digest(manifest_path),
                  metadata_sha256=digest(metadata_path), training_manifest=manifest,
                  threshold=threshold, history=8, window=1, patience=0, comparison="strict >",
                  detection_time_denominator=max_length - 1, device="cpu", fitting=False,
                  test_global_indices=test_indices.tolist(), test_filenames=names,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  working_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()),
                  source_hashes={str(p.relative_to(REPO)): digest(p) for p in
                                 [Path(__file__).resolve(), REPO / "evaluation/tac_sorting_fixed_eval.py",
                                  REPO / "evaluation/temporal_models.py", REPO / "evaluation/utils.py",
                                  REPO / "datasets/tac_fiper_a0_normalization.py"]},
                  baseline_comparison="not_yet_available", no_performance_improvement_claim=True)
    (output / "frozen_inputs.json").write_text(json.dumps(frozen, indent=2) + "\n")
    episodes = []
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for number, (index, path) in enumerate(zip(test_indices, paths), 1):
            payload = path.read_bytes()
            raw = pickle.loads(payload)
            steps = raw["rollout"]
            length = int(metadata["episode_end_indices"][index] - metadata["episode_start_indices"][index])
            require(len(steps) == length and length > 0, f"{path.name}: episode length mismatch")
            # Labels are never passed to the scorer, model, or normalizer.
            require(bool(raw["metadata"]["successful"]) == bool(labels[number - 1]), "Raw/metadata label mismatch")
            obs = torch.as_tensor(np.stack([s["obs_embedding"] for s in steps]), dtype=torch.float32)
            action = torch.as_tensor(np.stack([s["action_pred"] for s in steps]), dtype=torch.float32)
            require(tuple(obs.shape) == (length, 128) and tuple(action.shape) == (length, 32, 8, 3),
                    f"{path.name}: unexpected raw shapes")
            action = torch.cat((action, torch.cumsum(action * ts, dim=-2)), dim=-1)
            scores = score_episode(model, normalizer, obs, action)
            record = dict(rollout_index=int(index), filename=path.name, sha256=hashlib.sha256(payload).hexdigest(),
                          successful=bool(labels[number - 1]), scores=scores)
            episodes.append(record)
            stream.write(json.dumps(record) + "\n")
            if number % 100 == 0:
                print(f"scored {number}/400", flush=True)
    metrics, first_steps = fixed_metrics(episodes, threshold, max_length)
    require(digest(checkpoint) == frozen["checkpoint_sha256"] and digest(manifest_path) == frozen["training_manifest_sha256"],
            "Training artifacts changed during evaluation")
    (output / "alarms.json").write_text(json.dumps([
        dict(rollout_index=e["rollout_index"], first_alarm_step=t,
             normalized_alarm_time=None if t is None else t / (max_length - 1))
        for e, t in zip(episodes, first_steps)], indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"FIXED EVALUATION COMPLETE: {output}")
    print("Exploratory TAC-only result; no FiPER improvement conclusion yet.")


if __name__ == "__main__":
    main()
