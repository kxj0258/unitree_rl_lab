#!/usr/bin/env python3
"""Export an explicit-state GRU actor and validate it on a real Isaac rollout."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/unitree_rl_lab"))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from unitree_rl_lab.utils.teacher_bank import load_registry, load_teacher, read_snapshot, sha256_file


class ExportActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.normalizer = policy.actor_obs_normalizer
        self.gru = policy.memory_a.rnn
        self.actor = policy.actor

    def forward(self, observation, hidden_in, reset):
        hidden = hidden_in * (1.0 - reset.reshape(1, -1, 1))
        latent, hidden_out = self.gru(self.normalizer(observation).unsqueeze(0), hidden)
        return self.actor(latent.squeeze(0)), hidden_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=ROOT / "config/seven_teacher_registry_v1.json")
    p.add_argument("--teacher", default="flat")
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output already exists")
    entry = load_registry(args.registry, ROOT)["teachers"][args.teacher]
    sequence = np.load(args.sequence, allow_pickle=False)
    if str(sequence["checkpoint_sha256"]) != entry["checkpoint_sha256"]:
        p.error("sequence belongs to a different checkpoint")
    if sequence["observations"].shape[-1] != entry["policy_dim"]:
        p.error("sequence observation dimension mismatch")
    torch.set_num_threads(2)
    policy = load_teacher(entry, ROOT)
    model = ExportActor(policy).eval()
    args.output.mkdir(parents=True)
    onnx_path = args.output / "policy.onnx"
    layers, hidden_size = entry["network"]["rnn_num_layers"], entry["network"]["rnn_hidden_dim"]
    torch.onnx.export(model, (torch.zeros(1, entry["policy_dim"]), torch.zeros(layers, 1, hidden_size), torch.ones(1)),
                      str(onnx_path), input_names=["observation", "hidden_in", "reset"],
                      output_names=["action", "hidden_out"], opset_version=17, dynamo=False,
                      dynamic_axes={"observation": {0: "batch"}, "hidden_in": {1: "batch"},
                                    "reset": {0: "batch"}, "action": {0: "batch"}, "hidden_out": {1: "batch"}})
    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    reports = []
    with torch.no_grad():
        for batch in sorted({1, sequence["observations"].shape[1]}):
            h_pt = torch.zeros(layers, batch, hidden_size)
            h_ort = h_pt.numpy().copy()
            errors = {"action_pt_ort": 0., "hidden_pt_ort": 0., "action_isaac_pt": 0.}
            reset_count = 0
            for step, obs in enumerate(sequence["observations"][:, :batch]):
                reset = np.ones(batch, np.float32) if step == 0 else sequence["dones"][step-1, :batch].astype(np.float32)
                reset_count += int(reset.sum())
                a_pt, h_pt = model(torch.from_numpy(obs), h_pt, torch.from_numpy(reset))
                a_ort, h_ort = session.run(None, {"observation": obs, "hidden_in": h_ort, "reset": reset})
                expected = sequence["actions"][step, :batch]
                for name, a, b in (("action_pt_ort", a_pt.numpy(), a_ort),
                                   ("hidden_pt_ort", h_pt.numpy(), h_ort), ("action_isaac_pt", expected, a_pt.numpy())):
                    errors[name] = max(errors[name], float(np.max(np.abs(a-b))))
                    np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
            reports.append({"batch": batch, "steps": len(sequence["observations"]), "resets_including_initial": reset_count,
                            "max_abs_error": errors})
    env = read_snapshot(ROOT / entry["config_artifacts"]["env"]["path"])
    sdk_names = env["scene"]["robot"]["joint_sdk_names"]
    joint_names = [sdk_names[i] for i in entry["control"]["joint_ids_map"]]
    if joint_names != sequence["joint_names"].tolist():
        raise ValueError("captured articulation joint order disagrees with frozen mapping")
    manifest = {"schema_version": 1, "teacher": args.teacher, "teacher_contract": entry,
                "onnx_sha256": sha256_file(onnx_path), "registry_sha256": sha256_file(args.registry),
                "joint_names": joint_names, "actuator": env["scene"]["robot"]["actuators"]["GO2HV"],
                "initial_base_height": env["scene"]["robot"]["init_state"]["pos"][2],
                "interface": {"observation": ["batch", entry["policy_dim"]], "hidden_in": [layers, "batch", hidden_size],
                              "reset": ["batch"], "action": ["batch", 12], "hidden_out": [layers, "batch", hidden_size],
                              "reset_semantics": "1 resets that batch slot BEFORE consuming current observation",
                              "normalization": "embedded in ONNX; input already clipped/scaled once"},
                "validation": {"status": "pass", "sequence": str(args.sequence.resolve()),
                               "sequence_sha256": sha256_file(args.sequence), "reports": reports}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["validation"], indent=2))


if __name__ == "__main__":
    main()
