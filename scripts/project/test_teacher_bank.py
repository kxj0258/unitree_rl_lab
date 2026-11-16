"""Focused regression tests for contract boundaries, without Isaac Sim or checkpoints."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/unitree_rl_lab"))
import torch
from unitree_rl_lab.utils.teacher_bank import (
    ObservationAdapter, gait_clock, load_registry, read_snapshot, sha256_file, verify_artifacts,
)


class TeacherContractTests(unittest.TestCase):
    def test_checkpoint_or_config_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config = root / "model.pt", root / "env.yaml"
            checkpoint.write_bytes(b"frozen checkpoint")
            config.write_text("scale: 0.25\n")
            entry = {"checkpoint": "model.pt", "checkpoint_sha256": sha256_file(checkpoint),
                     "config_artifacts": {"env": {"path": "env.yaml", "sha256": sha256_file(config)}}}
            verify_artifacts(entry, root)
            config.write_text("scale: 0.50\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_artifacts(entry, root)

    def test_yaml_python_tags_are_data_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            marker = Path(directory) / "must_not_exist"
            path.write_text(f'!!python/object/apply:os.system ["touch {marker}"]')
            self.assertIsInstance(read_snapshot(path), list)
            self.assertFalse(marker.exists())

    def test_named_terms_clip_before_scale_and_leave_input_unchanged(self):
        entry = {"observations": {"policy": [
            {"name": "a", "dim": 1, "clip": [-1., 1.], "scale": 2.},
            {"name": "b", "dim": 1, "clip": None, "scale": None}]}}
        raw = {"b": torch.tensor([[3.]]), "a": torch.tensor([[10.]])}
        actual = ObservationAdapter(entry).build(raw)
        torch.testing.assert_close(actual, torch.tensor([[2., 3.]]))
        self.assertEqual(raw["a"].item(), 10.)

    def test_mismatched_batches_and_nonfinite_input_are_rejected(self):
        entry = {"observations": {"policy": [
            {"name": "a", "dim": 1, "clip": None, "scale": None},
            {"name": "b", "dim": 1, "clip": None, "scale": None}]}}
        adapter = ObservationAdapter(entry)
        with self.assertRaises(ValueError):
            adapter.build({"a": torch.zeros(1, 1), "b": torch.zeros(2, 1)})
        with self.assertRaises(ValueError):
            adapter.build({"a": torch.zeros(1, 1), "b": torch.tensor([[float("inf")]])})

    def test_gait_clock_resets_and_speed_changes_period(self):
        params = {"minimum_speed": .15, "maximum_speed": .50, "period": .75, "minimum_period": .55}
        result = gait_clock(torch.tensor([0., .375 / .02, .275 / .02]), .02,
                            torch.tensor([.3, .15, .5]), params)
        torch.testing.assert_close(result, torch.tensor([[0., 1.], [0., -1.], [0., -1.]]), atol=1e-6, rtol=0)

    def test_registry_rejects_proxy_reuse_and_missing_terrain(self):
        registry = json.loads((ROOT / "config/seven_teacher_registry_v1.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            invalid = copy.deepcopy(registry)
            invalid["teachers"]["boxes"]["checkpoint"] = invalid["teachers"]["flat"]["checkpoint"]
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "independent"):
                load_registry(path, ROOT)
            del registry["teachers"]["flat"]
            path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError, "seven"):
                load_registry(path, ROOT)


if __name__ == "__main__":
    unittest.main()
