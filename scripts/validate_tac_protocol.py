#!/usr/bin/env python3
"""Validate the TAC-FIPER protocol against rollout metadata."""

from __future__ import annotations

import argparse
import pickle
import sys
from itertools import combinations
from numbers import Integral
from pathlib import Path

import numpy as np
import yaml


STAGES = ("representation", "selection", "threshold")
LABEL_KEYS = (
    "successful_rollout_labels",
    "failed_rollout_labels",
    "calibration_rollout_labels",
    "test_rollout_labels",
)


def load_metadata(path: Path) -> dict:
    with path.open("rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    return metadata


def get_bool_mask(metadata: dict, key: str, task: str) -> np.ndarray:
    if key not in metadata:
        raise ValueError(f"{task}: metadata is missing {key}")

    values = np.asarray(metadata[key])
    if values.ndim != 1 or values.dtype != np.bool_:
        raise ValueError(
            f"{task}: {key} must be a one-dimensional bool array, "
            f"got shape={values.shape}, dtype={values.dtype}"
        )
    return values


def check_expected_counts(
    errors: list[str],
    task: str,
    task_config: dict,
    successful: np.ndarray,
    failed: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
) -> None:
    expected = task_config.get("expected")
    if not isinstance(expected, dict):
        errors.append(f"{task}: missing expected counts")
        return

    for split_name, split_mask in (
        ("calibration", calibration),
        ("test", test),
    ):
        split_config = expected.get(split_name)
        if not isinstance(split_config, dict):
            errors.append(f"{task}: missing expected.{split_name}")
            continue

        for label_name, label_mask in (
            ("successful", successful),
            ("failed", failed),
        ):
            expected_count = split_config.get(label_name)
            if (
                isinstance(expected_count, bool)
                or not isinstance(expected_count, Integral)
                or expected_count < 0
            ):
                errors.append(
                    f"{task}: expected.{split_name}.{label_name} "
                    "must be a non-negative integer"
                )
                continue

            actual_count = int(np.sum(split_mask & label_mask))
            if actual_count != int(expected_count):
                errors.append(
                    f"{task}: {split_name} {label_name} count mismatch: "
                    f"expected {expected_count}, got {actual_count}"
                )


def validate_resolved_stage(
    errors: list[str],
    task: str,
    stage: str,
    stage_config: dict,
    eligible_mask: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
) -> set[int]:
    requested = stage_config.get("requested_rollouts")
    indices = stage_config.get("rollout_indices")

    if (
        isinstance(requested, bool)
        or not isinstance(requested, Integral)
        or requested < 0
    ):
        errors.append(
            f"{task}: tac.{stage}.requested_rollouts must be "
            "a non-negative integer"
        )
        requested = None

    if not isinstance(indices, list):
        errors.append(f"{task}: tac.{stage}.rollout_indices must be a list")
        return set()

    valid_indices: list[int] = []
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, Integral):
            errors.append(
                f"{task}: tac.{stage}.rollout_indices contains "
                f"a non-integer value: {value!r}"
            )
            continue

        index = int(value)
        if index < 0 or index >= len(eligible_mask):
            errors.append(
                f"{task}: tac.{stage} rollout index {index} is out of range"
            )
            continue

        valid_indices.append(index)

        if test[index]:
            errors.append(
                f"{task}: tac.{stage} rollout index {index} belongs to test"
            )
        if not calibration[index]:
            errors.append(
                f"{task}: tac.{stage} rollout index {index} "
                "is not calibration"
            )
        if not eligible_mask[index]:
            errors.append(
                f"{task}: tac.{stage} rollout index {index} "
                "is not successful calibration"
            )

    unique_indices = set(valid_indices)
    if len(unique_indices) != len(valid_indices):
        errors.append(f"{task}: tac.{stage}.rollout_indices has duplicates")

    if requested is not None and requested != len(valid_indices):
        errors.append(
            f"{task}: tac.{stage} requests {requested} rollouts, "
            f"but lists {len(valid_indices)} indices"
        )

    available = int(np.sum(eligible_mask))
    if requested is not None and requested > available:
        errors.append(
            f"{task}: tac.{stage} requests {requested} rollouts, "
            f"but only {available} successful calibration rollouts exist"
        )

    return unique_indices


def validate_task(
    task: str,
    task_config: dict,
    data_root: Path,
    metadata_relative_path: str,
    require_resolved: bool,
) -> tuple[list[str], bool]:
    errors: list[str] = []
    metadata_path = data_root / task / metadata_relative_path

    if not metadata_path.exists():
        return [f"{task}: metadata does not exist: {metadata_path}"], False

    try:
        metadata = load_metadata(metadata_path)
        masks = {
            key: get_bool_mask(metadata, key, task)
            for key in LABEL_KEYS
        }
    except (OSError, ValueError, pickle.PickleError) as exc:
        return [f"{task}: cannot read metadata: {exc}"], False

    lengths = {len(mask) for mask in masks.values()}
    if len(lengths) != 1:
        return [f"{task}: metadata masks have different lengths"], False

    successful = masks["successful_rollout_labels"]
    failed = masks["failed_rollout_labels"]
    calibration = masks["calibration_rollout_labels"]
    test = masks["test_rollout_labels"]

    if np.any(successful & failed):
        errors.append(f"{task}: successful and failed masks overlap")
    if not np.all(successful | failed):
        errors.append(f"{task}: successful and failed masks do not cover all rollouts")
    if np.any(calibration & test):
        errors.append(f"{task}: calibration and test masks overlap")
    if not np.all(calibration | test):
        errors.append(f"{task}: calibration and test masks do not cover all rollouts")

    num_rollouts = metadata.get("num_rollouts")
    if isinstance(num_rollouts, Integral):
        if int(num_rollouts) != len(successful):
            errors.append(
                f"{task}: num_rollouts={num_rollouts}, "
                f"but masks have length {len(successful)}"
            )

    check_expected_counts(
        errors,
        task,
        task_config,
        successful,
        failed,
        calibration,
        test,
    )

    tac_config = task_config.get("tac")
    if not isinstance(tac_config, dict):
        errors.append(f"{task}: missing tac configuration")
        return errors, False

    allocation_status = tac_config.get("allocation_status")
    if allocation_status not in ("unresolved", "resolved"):
        errors.append(
            f"{task}: tac.allocation_status must be "
            "'unresolved' or 'resolved'"
        )
        return errors, False

    unresolved = allocation_status == "unresolved"
    eligible_mask = successful & calibration
    role_indices: dict[str, set[int]] = {}

    for stage in STAGES:
        stage_config = tac_config.get(stage)
        if not isinstance(stage_config, dict):
            errors.append(f"{task}: missing tac.{stage}")
            continue

        if stage_config.get("source") != "successful_calibration":
            errors.append(
                f"{task}: tac.{stage}.source must be "
                "'successful_calibration'"
            )

        if unresolved:
            if stage_config.get("requested_rollouts") is not None:
                errors.append(
                    f"{task}: unresolved tac.{stage}.requested_rollouts "
                    "must be null"
                )
            if stage_config.get("rollout_indices") != []:
                errors.append(
                    f"{task}: unresolved tac.{stage}.rollout_indices "
                    "must be empty"
                )
        else:
            role_indices[stage] = validate_resolved_stage(
                errors,
                task,
                stage,
                stage_config,
                eligible_mask,
                calibration,
                test,
            )

    if unresolved and require_resolved:
        errors.append(f"{task}: TAC role allocation remains unresolved")

    if not unresolved:
        for first, second in combinations(STAGES, 2):
            overlap = role_indices.get(first, set()) & role_indices.get(
                second, set()
            )
            if overlap:
                errors.append(
                    f"{task}: tac.{first} and tac.{second} overlap "
                    f"at rollout indices {sorted(overlap)}"
                )

    return errors, unresolved


def validate_config(
    config_path: Path,
    selected_tasks: list[str] | None,
    require_resolved: bool = False,
) -> int:
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1

    if not isinstance(config, dict):
        print("CONFIG ERROR: top-level YAML value must be a mapping")
        return 1

    errors: list[str] = []

    if config.get("status") != "draft":
        errors.append("config status must currently be 'draft'")

    data_config = config.get("data")
    if not isinstance(data_config, dict):
        errors.append("missing data configuration")
        data_config = {}

    if data_config.get("split_unit") != "rollout":
        errors.append("data.split_unit must be 'rollout'")
    if data_config.get("rollout_identity") != "metadata_index":
        errors.append("data.rollout_identity must be 'metadata_index'")

    metadata_relative_path = data_config.get("metadata")
    if not isinstance(metadata_relative_path, str):
        errors.append("data.metadata must be a string")
        metadata_relative_path = "processed_rollouts/metadata.pkl"

    rules = config.get("protocol_rules")
    if not isinstance(rules, dict):
        errors.append("missing protocol_rules")
        rules = {}

    required_true_rules = (
        "tac_successful_calibration_only",
        "require_explicit_rollout_indices",
        "require_disjoint_tac_roles",
        "allow_unresolved_allocation_in_draft",
    )
    for key in required_true_rules:
        if rules.get(key) is not True:
            errors.append(f"protocol_rules.{key} must be true")

    required_false_rules = (
        "allow_failed_calibration_for_representation",
        "allow_failed_calibration_for_selection",
        "allow_failed_calibration_for_threshold",
        "allow_calibration_unused_auto_fill",
    )
    for key in required_false_rules:
        if rules.get(key) is not False:
            errors.append(f"protocol_rules.{key} must be false")

    leakage = config.get("data_leakage_rules")
    if not isinstance(leakage, dict):
        errors.append("missing data_leakage_rules")
        leakage = {}

    for key in (
        "use_test_for_training",
        "use_test_for_hyperparameter_selection",
        "use_test_for_epoch_selection",
        "use_test_for_threshold_selection",
    ):
        if leakage.get(key) is not False:
            errors.append(f"data_leakage_rules.{key} must be false")

    conformal = config.get("conformal")
    if not isinstance(conformal, dict):
        errors.append("missing conformal configuration")
    elif conformal.get("finite_sample_guarantee_claimed") is not False:
        errors.append("conformal finite-sample guarantee must not be claimed")

    tasks = config.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        errors.append("tasks must be a non-empty mapping")
        tasks = {}

    unknown_tasks = set(selected_tasks or ()) - set(tasks)
    for task in sorted(unknown_tasks):
        errors.append(f"requested task is not configured: {task}")

    tasks_to_validate = selected_tasks or list(tasks)
    data_root = Path(data_config.get("root", "data"))
    unresolved_tasks: list[str] = []

    for task in tasks_to_validate:
        if task not in tasks:
            continue
        task_config = tasks[task]
        if not isinstance(task_config, dict):
            errors.append(f"{task}: task configuration must be a mapping")
            continue

        task_errors, unresolved = validate_task(
            task,
            task_config,
            data_root,
            metadata_relative_path,
            require_resolved,
        )
        errors.extend(task_errors)
        if unresolved:
            unresolved_tasks.append(task)

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"METADATA VALIDATION PASSED: {', '.join(tasks_to_validate)}")
    if unresolved_tasks:
        print(
            "DRAFT ONLY: TAC role allocation remains unresolved for "
            + ", ".join(unresolved_tasks)
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eval/tac_fiper_protocol.yaml",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="validate only this configured task; may be repeated",
    )
    parser.add_argument(
        "--require-resolved",
        action="store_true",
        help="fail if any TAC role allocation remains unresolved",
    )
    args = parser.parse_args()

    return validate_config(
        Path(args.config),
        args.tasks,
        args.require_resolved,
    )


if __name__ == "__main__":
    sys.exit(main())