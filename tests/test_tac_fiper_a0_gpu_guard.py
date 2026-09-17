import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.tac_fiper_a0_training import run_cpu_smoke


class GpuGuardTest(unittest.TestCase):
    def test_mismatched_visibility_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-other"}):
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    run_cpu_smoke(None, output, {}, gpu_uuid="GPU-approved")
            self.assertFalse(output.exists())

    def test_occupied_gpu_rejected_without_cuda_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-approved"}), \
                    patch("evaluation.tac_fiper_a0_training.subprocess.check_output",
                          return_value="GPU-approved, 1234\n"), \
                    patch("torch.cuda.is_available") as cuda:
                with self.assertRaisesRegex(RuntimeError, "compute process"):
                    run_cpu_smoke(None, output, {}, gpu_uuid="GPU-approved")
                cuda.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
