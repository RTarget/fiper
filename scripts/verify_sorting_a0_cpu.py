"""Verify sorting A0 using only raw calibration files, without training.

Run from the repository root:
  CUDA_VISIBLE_DEVICES="" python /tmp/verify_sorting_a0_cpu.py

Reads processed metadata and 50 raw calibration pickle files. Never loads
merged .pt tensors, the saved normalizer, or test/calibration_unused rollouts.
Writes only the explicitly named JSON report (default: /tmp/tac_a0_cpu_check.json).
This checks reconstruction and pipeline shapes, not numerical equivalence to
the saved merged tensors, trained performance, or the complete A0 pipeline.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=Path("/tmp/tac_a0_cpu_check.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    require((repo / "datasets/tac_fiper_a0.py").is_file(), "Run from the fiper repository root")
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    import yaml
    from datasets.rollout_datasets import ProcessedRolloutDataset
    from datasets.tac_fiper_a0 import ROLE_INDICES, SortingA0Dataset, _validate_metadata, make_a0_dataloader
    from datasets.tac_fiper_a0_normalization import A0FeatureNormalizer
    from evaluation.temporal_models import ObservationActionTemporalModel
    from tasks.task_manager import TaskManager

    torch.set_num_threads(2)
    torch.manual_seed(20260909)
    root = repo / "data/sorting"
    report_path = args.report.resolve()
    require(not report_path.is_relative_to((repo / "data").resolve()),
            "Report must be outside the original data directory")
    metadata_path = root / "processed_rollouts/metadata.pkl"
    with metadata_path.open("rb") as handle:
        original = pickle.load(handle)
    _validate_metadata(original)  # Validate every role BEFORE reading raw rollouts.

    calibration_indices = np.flatnonzero(original["calibration_rollout_labels"])
    require(np.array_equal(calibration_indices, np.arange(50)),
            "This verifier requires calibration global indices 0..49; do not renumber")
    require(int(original["num_robots"]) == 1, "Only single-robot sorting is supported")
    starts = np.asarray(original["episode_start_indices"])[:50]
    ends = np.asarray(original["episode_end_indices"])[:50]
    require(starts[0] == 0 and np.array_equal(starts[1:], ends[:-1]),
            "Calibration timesteps must form a contiguous prefix")
    filenames = list(original["rollout_filenames"]["calibration"])
    require(len(filenames) == 50 and len(set(filenames)) == 50,
            "Expected exactly 50 unique calibration filenames")
    require(filenames == sorted(filenames),
            "Filename ordering differs from TaskManager's sorted rollout loading")

    with (repo / "configs/task/sorting.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw_mapping = config["action_space"]["actions"]["action_mapping"]
    require(raw_mapping.get("velocity") == [0, 1, 2]
            and raw_mapping.get("position") is None
            and raw_mapping.get("rotation") is None
            and raw_mapping.get("angular_velocity") is None,
            "Unexpected sorting action mapping; inspect before adapting")
    ts = float(config["environment"]["ts"])
    require(np.isfinite(ts) and ts > 0, "Invalid integration timestep")
    require(np.isclose(float(original["env"]["ts"]), ts),
            "Saved metadata and sorting config have different integration timesteps")
    require(not (root / "processed_rollouts/states.pt").exists(),
            "states.pt exists: verify whether original preprocessing used initial positions")

    # Only this exact calibration directory may supply rollout payloads.
    cal_dir = root / "rollouts/calibration"
    require(not cal_dir.is_symlink(), "Calibration directory must not redirect elsewhere")
    cal_dir = cal_dir.resolve(strict=True)
    observations, actions, opened = [], [], []
    for index, name in enumerate(filenames):
        require(isinstance(name, str) and Path(name).name == name and name.endswith(".pkl"),
                "Expected a plain calibration pickle filename")
        path = (cal_dir / name).resolve(strict=True)
        require(path.parent == cal_dir, "Rollout path escapes calibration directory")
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        opened.append(str(path))
        require(isinstance(raw, dict) and isinstance(raw.get("rollout"), list),
                f"{name}: unexpected raw format")
        require(bool(raw["metadata"]["successful"]), f"{name}: raw rollout is not successful")
        steps = raw["rollout"]
        require(len(steps) == int(ends[index] - starts[index]),
                f"{name}: raw length does not match global metadata index {index}")
        obs_array = np.stack([step["obs_embedding"] for step in steps])
        action_array = np.stack([step["action_pred"] for step in steps])
        require(obs_array.shape == (len(steps), 128), f"{name}: unexpected observation shape")
        require(action_array.shape == (len(steps), 32, 8, 3), f"{name}: unexpected raw action shape")
        require(np.issubdtype(obs_array.dtype, np.floating)
                and np.issubdtype(action_array.dtype, np.floating), f"{name}: non-floating features")
        obs = torch.as_tensor(obs_array, dtype=torch.float32, device="cpu")
        action = torch.as_tensor(action_array, dtype=torch.float32, device="cpu")
        require(torch.isfinite(obs).all().item() and torch.isfinite(action).all().item(),
                f"{name}: non-finite features")
        observations.append(obs)
        actions.append(action)

    # Use the SERVER's actual pure augmentation method without initializing
    # TaskManager, whose loading/initialization paths can read test or save data.
    raw_actions = torch.cat(actions, dim=0)
    augmentation_metadata = {
        "actions": {"action_mapping": copy.deepcopy(raw_mapping)}, "env": {"ts": ts},
    }
    converted = TaskManager._augment_actions(None, {
        "metadata": augmentation_metadata, "action_preds": raw_actions,
    })
    augmented_actions = converted["action_preds"]
    expected_actions = torch.cat((raw_actions, torch.cumsum(raw_actions * ts, dim=-2)), dim=-1)
    require(torch.equal(augmented_actions, expected_actions), "Unexpected server augmentation behavior")
    require(tuple(augmented_actions.shape[1:]) == (32, 8, 6), "Expected six augmented action columns")

    # Calibration is already global indices 0..49, so this prefix view does not
    # renumber any rollout. There are no dummy tensors for test episodes.
    metadata = copy.deepcopy(original)
    for key in ("episode_start_indices", "episode_end_indices", "episode_lengths",
                "calibration_rollout_labels", "test_rollout_labels",
                "successful_rollout_labels", "failed_rollout_labels",
                "id_rollout_labels", "ood_rollout_labels"):
        metadata[key] = np.asarray(original[key])[:50].copy()
    metadata["num_rollouts"] = 50
    metadata["num_steps"] = int(ends[-1])
    metadata["rollout_filenames"] = {"calibration": filenames, "test": []}
    metadata["available_tensors"] = ["obs_embeddings", "action_preds"]
    original_action_dim = int(metadata["actions"]["action_dim"])
    metadata["actions"]["action_dim"] = 6  # Explicit all-six-column A0 view only.
    metadata["actions"]["action_mapping"] = converted["metadata"]["actions"]["action_mapping"]
    source = ProcessedRolloutDataset(str(root), str(repo / "configs"),
                                     required_tensors=["obs_embeddings", "action_preds"])
    source.data = {"metadata": metadata, "obs_embeddings": torch.cat(observations, dim=0),
                   "action_preds": augmented_actions}
    source.dataset_loaded = True
    source.normalize_tensors = {"obs_embeddings": False, "action_preds": False}
    require(source.data["obs_embeddings"].shape[0] == metadata["num_steps"], "Step count mismatch")

    def forbidden(*args, **kwargs):
        raise RuntimeError("This check must not normalize, fit, load merged data, or save dataset files")
    for method in ("normalize", "_save_dataset", "load_dataset", "init_dataset"):
        setattr(source, method, forbidden)

    representation = SortingA0Dataset(source, "representation", task="sorting")
    normalizer = A0FeatureNormalizer.fit(representation)
    norm_state = normalizer.state_dict()
    restored_normalizer = A0FeatureNormalizer.from_state_dict(norm_state)
    print(f"normalization: representation_only, obs_rows={norm_state['obs_count']}, "
          f"action_rows={norm_state['action_count']}")

    model = ObservationActionTemporalModel(
        obs_dim=128, action_horizon=8, action_dim=6, d_model=16,
        nhead=2, num_layers=1, dim_feedforward=32,
    ).cpu().eval()
    results = {}
    for role, indices in ROLE_INDICES.items():
        loader = make_a0_dataloader(source, role, task="sorting", batch_size=8)
        lengths = metadata["episode_lengths"][list(indices)]
        expected_count = int(lengths.sum()) if role == "threshold" else int(2 * (lengths - 1).sum())
        require(len(loader.dataset) == expected_count, f"{role}: incorrect sample count")
        raw_batch = next(iter(loader))
        batch = normalizer.transform(raw_batch)
        restored_batch = restored_normalizer.transform(raw_batch)
        for key in ("obs_embeddings", "action_preds"):
            require(torch.equal(batch[key], restored_batch[key]), "Normalization state reload mismatch")
        require(tuple(batch["action_preds"].shape[1:]) == (8, 32, 8, 6),
                "Iterator truncated augmented action columns")
        with torch.no_grad():
            logits = model(batch["obs_embeddings"], batch["action_preds"], batch["padding_mask"])
            scores = torch.nn.functional.softplus(-logits)
        require(torch.isfinite(logits).all().item() and torch.isfinite(scores).all().item(),
                f"{role}: non-finite random-model forward output")
        results[role] = {"rollouts": len(indices), "samples": len(loader.dataset),
                         "indices": list(indices), "first_batch_forward_finite": True}
        print(f"{role}: rollouts={len(indices)}, samples={len(loader.dataset)}, forward=OK")

    for key in ("obs_mean", "obs_scale", "action_mean", "action_scale"):
        require(torch.equal(norm_state[key], normalizer.state_dict()[key]), "Frozen statistics changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    report = {
        "status": "PASS", "scope": "real_calibration_constructor_and_first_batch_forward_only",
        "commit": commit, "device": "cpu", "seed": 20260909, "history": 8,
        "working_tree_dirty": bool(dirty),
        "normalization": {"fit_role": "representation", "obs_count": norm_state["obs_count"],
                          "action_count": norm_state["action_count"],
                          "rollout_indices": norm_state["rollout_indices"],
                          "frozen": True, "state_reload_equal": True},
        "model": "random_untrained_small_backbone",
        "integration_ts": ts, "position_origin": "zero_no_states",
        "raw_action_shape": [32, 8, 3], "a0_action_shape": [32, 8, 6],
        "original_metadata_action_dim": original_action_dim, "in_memory_action_dim": 6,
        "calibration_files_read": opened, "test_payload_files_read": 0,
        "merged_tensor_files_read": 0,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "roles": results,
        "limits": ["No saved-tensor numerical equivalence check", "No training or threshold fitting",
                   "Only first batch per role receives random-model forward"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"actions: raw=3, A0=6, original_metadata_action_dim={original_action_dim}, ts={ts}")
    print("PASS: calibration_files=50, test_payload_files=0, device=cpu, training=none")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
