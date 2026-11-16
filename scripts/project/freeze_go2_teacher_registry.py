#!/usr/bin/env python3
"""Freeze explicitly selected local teachers; never train or choose a best model."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/unitree_rl_lab"))
import torch
from unitree_rl_lab.utils.teacher_bank import TERM_DIMS, read_snapshot, sha256_file, load_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite registry")
    selection = json.loads(args.selection.read_text())
    teachers = {}
    for terrain, chosen in selection.items():
        checkpoint = ROOT / chosen["checkpoint"]
        paths = {name: checkpoint.parent / "params" / f"{name}.yaml" for name in ("env", "agent", "deploy")}
        env, agent, deploy = (read_snapshot(paths[k]) for k in ("env", "agent", "deploy"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        observations = {}
        for group in ("policy", "critic"):
            observations[group] = []
            for name, cfg in env["observations"][group].items():
                if not isinstance(cfg, dict) or "func" not in cfg:
                    continue
                if cfg.get("history_length") != 0 or cfg.get("modifiers"):
                    raise ValueError("unsupported observation history/modifier")
                observations[group].append({"name": name, "dim": TERM_DIMS[name],
                                            **{k: cfg[k] for k in ("func", "params", "clip", "scale")}})
        dimensions = {"policy": state["model_state_dict"]["memory_a.rnn.weight_ih_l0"].shape[1],
                      "critic": state["model_state_dict"]["memory_c.rnn.weight_ih_l0"].shape[1]}
        for group, dimension in dimensions.items():
            if sum(t["dim"] for t in observations[group]) != dimension:
                raise ValueError(f"snapshot/checkpoint dimensions disagree: {terrain}/{group}")
        scanner = env["scene"]["height_scanner"]
        teachers[terrain] = {
            **chosen, "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_iteration": int(state["iter"]), "training_seed": agent["seed"],
            "config_artifacts": {k: {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}
                                 for k, p in paths.items()},
            "network": agent["policy"], "policy_dim": dimensions["policy"],
            "critic_dim": dimensions["critic"], "action_dim": 12,
            "observations": observations,
            "control": {k: deploy[k] for k in ("joint_ids_map", "step_dt", "stiffness", "damping", "default_joint_pos", "actions")},
            "command": env["commands"]["base_velocity"],
            "height_scanner": {k: scanner[k] for k in ("offset", "ray_alignment", "pattern_cfg", "update_period", "mesh_prim_paths")},
            "terrain": env["scene"]["terrain"]["terrain_generator"]["sub_terrains"],
            "acceptance_status": "frozen_selection_pending_registry_closed_loop_acceptance",
        }
    registry = {"schema_version": 1, "kind": "go2_seven_teacher_bank",
                "selection_source": str(args.selection.resolve().relative_to(ROOT)),
                "selection_sha256": sha256_file(args.selection),
                "freeze_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "scope": "independent teachers in their own tasks; no student or sim2sim claim",
                "raw_observation_contract": "named tensors before clipping/scaling/normalization; robot articulation joint order",
                "teachers": teachers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(registry, indent=2) + "\n")
    load_registry(args.output, ROOT)
    print(f"Frozen seven teachers: {args.output}")


if __name__ == "__main__":
    main()
