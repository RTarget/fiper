import contextlib
import copy
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_tac_protocol import validate_config


class TacProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.metadata_dir = (
            self.root / "data" / "sorting" / "processed_rollouts"
        )
        self.metadata_dir.mkdir(parents=True)

        self.metadata = {
            "num_rollouts": 8,
            "successful_rollout_labels": np.array(
                [True, True, True, False, False, True, False, False],
                dtype=bool,
            ),
            "failed_rollout_labels": np.array(
                [False, False, False, True, True, False, True, True],
                dtype=bool,
            ),
            "calibration_rollout_labels": np.array(
                [True, True, True, True, True, False, False, False],
                dtype=bool,
            ),
            "test_rollout_labels": np.array(
                [False, False, False, False, False, True, True, True],
                dtype=bool,
            ),
        }

        self.config = {
            "name": "tac_fiper_protocol",
            "version": 1,
            "status": "draft",
            "data": {
                "root": str(self.root / "data"),
                "metadata": "processed_rollouts/metadata.pkl",
                "split_unit": "rollout",
                "rollout_identity": "metadata_index",
            },
            "tasks": {
                "sorting": {
                    "expected": {
                        "calibration": {"successful": 3, "failed": 2},
                        "test": {"successful": 1, "failed": 2},
                    },
                    "tac": {
                        "allocation_status": "resolved",
                        "representation": {
                            "source": "successful_calibration",
                            "requested_rollouts": 1,
                            "rollout_indices": [0],
                        },
                        "selection": {
                            "source": "successful_calibration",
                            "requested_rollouts": 1,
                            "rollout_indices": [1],
                        },
                        "threshold": {
                            "source": "successful_calibration",
                            "requested_rollouts": 1,
                            "rollout_indices": [2],
                        },
                    },
                }
            },
            "protocol_rules": {
                "tac_successful_calibration_only": True,
                "require_explicit_rollout_indices": True,
                "require_disjoint_tac_roles": True,
                "allow_unresolved_allocation_in_draft": True,
                "allow_failed_calibration_for_representation": False,
                "allow_failed_calibration_for_selection": False,
                "allow_failed_calibration_for_threshold": False,
                "allow_calibration_unused_auto_fill": False,
            },
            "data_leakage_rules": {
                "use_test_for_training": False,
                "use_test_for_hyperparameter_selection": False,
                "use_test_for_epoch_selection": False,
                "use_test_for_threshold_selection": False,
            },
            "conformal": {
                "method": "unresolved",
                "finite_sample_guarantee_claimed": False,
            },
        }

        self.config_path = self.root / "protocol.yaml"
        self.write_metadata(self.metadata)
        self.write_config(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_metadata(self, metadata):
        with (self.metadata_dir / "metadata.pkl").open("wb") as handle:
            pickle.dump(metadata, handle)

    def write_config(self, config):
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def run_validation(self, require_resolved=False):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = validate_config(
                self.config_path,
                None,
                require_resolved=require_resolved,
            )
        return result, output.getvalue()

    def test_valid_resolved_protocol_passes(self):
        result, output = self.run_validation(require_resolved=True)
        self.assertEqual(result, 0)
        self.assertIn("METADATA VALIDATION PASSED", output)

    def test_valid_unresolved_draft_passes_metadata_validation(self):
        config = copy.deepcopy(self.config)
        tac = config["tasks"]["sorting"]["tac"]
        tac["allocation_status"] = "unresolved"

        for stage in ("representation", "selection", "threshold"):
            tac[stage]["requested_rollouts"] = None
            tac[stage]["rollout_indices"] = []

        self.write_config(config)
        result, output = self.run_validation()

        self.assertEqual(result, 0)
        self.assertIn("DRAFT ONLY", output)

    def test_require_resolved_rejects_unresolved_draft(self):
        config = copy.deepcopy(self.config)
        tac = config["tasks"]["sorting"]["tac"]
        tac["allocation_status"] = "unresolved"

        for stage in ("representation", "selection", "threshold"):
            tac[stage]["requested_rollouts"] = None
            tac[stage]["rollout_indices"] = []

        self.write_config(config)
        result, output = self.run_validation(require_resolved=True)

        self.assertEqual(result, 1)
        self.assertIn("allocation remains unresolved", output)

    def test_rejects_too_many_requested_rollouts(self):
        config = copy.deepcopy(self.config)
        stage = config["tasks"]["sorting"]["tac"]["representation"]
        stage["requested_rollouts"] = 4
        stage["rollout_indices"] = [0, 1, 2, 3]
        self.write_config(config)

        result, output = self.run_validation()
        self.assertEqual(result, 1)
        self.assertIn("requests 4 rollouts", output)

    def test_rejects_failed_calibration_rollout(self):
        config = copy.deepcopy(self.config)
        config["tasks"]["sorting"]["tac"]["threshold"][
            "rollout_indices"
        ] = [3]
        self.write_config(config)

        result, output = self.run_validation()
        self.assertEqual(result, 1)
        self.assertIn("not successful calibration", output)

    def test_rejects_test_rollout(self):
        config = copy.deepcopy(self.config)
        config["tasks"]["sorting"]["tac"]["threshold"][
            "rollout_indices"
        ] = [5]
        self.write_config(config)

        result, output = self.run_validation()
        self.assertEqual(result, 1)
        self.assertIn("belongs to test", output)

    def test_rejects_overlapping_tac_roles(self):
        config = copy.deepcopy(self.config)
        config["tasks"]["sorting"]["tac"]["selection"][
            "rollout_indices"
        ] = [0]
        self.write_config(config)

        result, output = self.run_validation()
        self.assertEqual(result, 1)
        self.assertIn("representation and tac.selection overlap", output)

    def test_rejects_overlapping_calibration_and_test_masks(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["calibration_rollout_labels"][5] = True
        self.write_metadata(metadata)

        result, output = self.run_validation()
        self.assertEqual(result, 1)
        self.assertIn("calibration and test masks overlap", output)

    def test_rejects_test_training_rule(self):
        config = copy.deepcopy(self.config)
        config["data_leakage_rules"]["use_test_for_training"] = True
        self.write_config(config)

        result, output = self.run_validation()
        self.assertEqual(result, 1)
        self.assertIn("use_test_for_training must be false", output)


if __name__ == "__main__":
    unittest.main()