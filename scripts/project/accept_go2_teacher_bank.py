#!/usr/bin/env python3
"""Audit or evaluate the frozen seven-teacher bank. Evaluation defaults to dry-run."""

import argparse
import contextlib
import csv
import io
import json
import math
import os
import shlex
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/unitree_rl_lab"))
import torch
from tensordict import TensorDict
from unitree_rl_lab.utils.teacher_bank import (
    TERRAINS, ObservationAdapter, gait_clock, load_registry, load_teacher, sha256_file,
)


def write_json(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def audit(registry):
    """Exercise real weights and state isolation; synthetic inputs are not locomotion evidence."""
    reports = {}
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for terrain in TERRAINS:
        entry = registry["teachers"][terrain]
        adapter = ObservationAdapter(entry)
        raw = {term["name"]: torch.randn(3, term["dim"])
               for group in entry["observations"].values() for term in group}
        original = {name: value.clone() for name, value in raw.items()}
        obs = TensorDict({group: adapter.build(raw, group) for group in ("policy", "critic")}, batch_size=[3])
        # Input mapping order must not affect the canonical network input order.
        torch.testing.assert_close(adapter.build(dict(reversed(list(raw.items())))), obs["policy"])
        for name in raw:
            torch.testing.assert_close(raw[name], original[name])
        for bad in ({k: v for k, v in raw.items() if k != "height_scan"},
                    {**raw, "height_scan": torch.zeros(3, 116)},
                    {**raw, "height_scan": torch.full((3, 117), float("nan"))}):
            try:
                adapter.build(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("adapter accepted a missing/invalid observation")
        extreme = {**raw, "base_ang_vel": torch.full((3, 3), 200.0)}
        torch.testing.assert_close(adapter.build(extreme)[:, :3], torch.full((3, 3), 20.0))
        with contextlib.redirect_stdout(io.StringIO()):
            policy = load_teacher(entry, ROOT)
        with torch.no_grad():
            first = policy.act_inference(obs).clone()
            policy.act_inference(obs)
            before = policy.memory_a.hidden_state.clone()
            policy.reset(torch.tensor([True, False, False]))
            if torch.count_nonzero(policy.memory_a.hidden_state[:, 0]) != 0:
                raise AssertionError("done environment state was not cleared")
            torch.testing.assert_close(policy.memory_a.hidden_state[:, 1:], before[:, 1:], atol=0, rtol=0)
            policy.reset()
            replay = policy.act_inference(obs)
            torch.testing.assert_close(first, replay, atol=0, rtol=0)
        if first.shape != (3, 12) or not torch.isfinite(first).all():
            raise AssertionError("invalid teacher inference")
        if any(t["name"] == "gait_clock" for t in entry["observations"]["policy"]):
            params = next(t["params"] for t in entry["observations"]["policy"] if t["name"] == "gait_clock")
            clock = gait_clock(torch.tensor([0., 0.75 / 0.02]), .02, torch.tensor([.15, .15]), params)
            torch.testing.assert_close(clock, torch.tensor([[0., 1.], [0., 1.]]), atol=1e-6, rtol=0)
        reports[terrain] = {"status": "pass", "checkpoint": entry["checkpoint"],
                            "checkpoint_sha256": entry["checkpoint_sha256"],
                            "policy_dim": entry["policy_dim"], "critic_dim": entry["critic_dim"],
                            "strict_weight_load": True, "finite_inference": True,
                            "reset_isolation": True, "reset_replay": True,
                            "named_order_and_invalid_input_checks": True}
        print(f"[AUDIT] {terrain}: pass ({entry['policy_dim']}/{entry['critic_dim']})", flush=True)
    return {"status": "pass", "scope": "artifact, adapter and inference acceptance only", "teachers": reports}


def build_plan(args, registry):
    plans = []
    envs = args.num_envs or (10 if args.mode == "smoke" else 100)
    episodes = args.episodes_per_env or (1 if args.mode == "smoke" else 2)
    seeds = args.seeds if args.seeds is not None else ([42] if args.mode == "smoke" else [7, 42, 123])
    if envs < 10 or envs % 10 or episodes <= 0 or any(seed < 0 for seed in seeds):
        raise ValueError("require num-envs multiple of ten, positive episodes, nonnegative seeds")
    if len(seeds) != len(set(seeds)) or len(args.profiles) != len(set(args.profiles)):
        raise ValueError("duplicate seeds/profiles would duplicate evidence")
    for terrain in args.terrains:
        entry = registry["teachers"][terrain]
        for seed in seeds:
            for profile in args.profiles:
                run_id = f"{terrain}_seed{seed}_{profile}"
                cmd = [sys.executable, str(ROOT / "scripts/project/evaluate_go2.py"),
                       "--task", entry["task"], "--checkpoint", str(ROOT / entry["checkpoint"]),
                       "--terrain_protocol", entry["terrain_protocol"], "--num_envs", str(envs),
                       "--episodes_per_env", str(episodes), "--seed", str(seed),
                       "--device", args.device, "--headless", "--output_dir", str(args.output / run_id),
                       "--teacher_registry", str(args.registry), "--teacher_id", terrain]
                if profile != "random":
                    vx = .35
                    heading = {"forward_mid": 0.0, "turn_left": math.pi / 4, "turn_right": -math.pi / 4}[profile]
                    cmd += ["--fixed_command_x", str(vx), "--fixed_command_y", "0", "--fixed_command_yaw", "0",
                            "--fixed_relative_heading", str(heading), "--fixed_initial_yaw", "0"]
                plans.append({"run_id": run_id, "terrain": terrain, "scenario_seed": seed, "profile": profile,
                              "expected_episodes": envs * episodes, "command": cmd})
    return plans


def summarize(output, plans, registry):
    groups = defaultdict(list)
    rows = []
    metrics = ("mean_xy_velocity_error_mps", "mean_abs_heading_error_rad", "mean_target_heading_progress_mps",
               "stall_fraction", "time_out", "base_contact", "bad_orientation")
    for plan in plans:
        folder = output / plan["run_id"]
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["teacher_bank"]["adapter_validation"]["status"] != "pass":
            raise ValueError("native/adapter comparison did not pass")
        data = list(csv.DictReader((folder / "episodes.csv").open()))
        if len(data) != plan["expected_episodes"]:
            raise ValueError("incomplete evaluation episodes")
        for row in data:
            for key, value in row.items():
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    raise ValueError(f"non-finite CSV value: {plan['run_id']}/{key}")
            enriched = {"run_id": plan["run_id"], "teacher_id": plan["terrain"], "profile": plan["profile"],
                        "training_seed": registry["teachers"][plan["terrain"]]["training_seed"],
                        "scenario_seed": plan["scenario_seed"], **row}
            rows.append(enriched)
            groups[(plan["terrain"], plan["profile"], "all")].append(row)
            groups[(plan["terrain"], plan["profile"], row["terrain_level"])].append(row)
    with (output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for (terrain, profile, level), data in groups.items():
        summaries.append({"terrain": terrain, "profile": profile, "level": level, "episodes": len(data),
                          **{key: sum(float(r[key]) for r in data) / len(data) for key in metrics}})
    with (output / "baseline.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return {"status": "pass", "scope": "registry loading, native parity and complete closed-loop collection",
            "locomotion_quality_gate": "not imposed; see baseline.csv including every failure",
            "runs": len(plans), "episodes": len(rows), "terrains": sorted({p["terrain"] for p in plans})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "config/seven_teacher_registry_v1.json")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; never reused")
    parser.add_argument("--mode", choices=("audit", "smoke", "formal"), default="audit")
    parser.add_argument("--execute", action="store_true", help="Execute simulation plan; otherwise only print/save it")
    parser.add_argument("--terrains", nargs="+", choices=TERRAINS, default=list(TERRAINS))
    parser.add_argument("--profiles", nargs="+", choices=("random", "forward_mid", "turn_left", "turn_right"), default=["random"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--episodes-per-env", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path,
                        default=Path(os.environ.get("UNITREE_MODEL_DIR", ROOT.parent / "unitree_model")))
    args = parser.parse_args()
    args.registry, args.output = args.registry.resolve(), args.output.resolve()
    args.model_dir = args.model_dir.resolve()
    if args.mode != "audit" and not (args.model_dir / "Go2/usd/go2.usd").is_file():
        parser.error("--model-dir must contain Go2/usd/go2.usd")
    if args.output.exists():
        parser.error(f"refusing to reuse output directory: {args.output}")
    if len(args.terrains) != len(set(args.terrains)):
        parser.error("duplicate terrains")
    registry = load_registry(args.registry, ROOT)
    plans = [] if args.mode == "audit" else build_plan(args, registry)
    args.output.mkdir(parents=True)
    try:
        provenance = {"registry_sha256": sha256_file(args.registry),
                      "unitree_model_dir": str(args.model_dir),
                      "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                      "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in [
                          Path(__file__), ROOT / "scripts/project/evaluate_go2.py",
                          ROOT / "source/unitree_rl_lab/unitree_rl_lab/utils/teacher_bank.py"]}}
        write_json(args.output / "provenance.json", provenance)
        write_json(args.output / "registry.json", registry)
        write_json(args.output / "plan.json", plans)
        result = audit(registry)
        write_json(args.output / "audit.json", result)
        if args.mode != "audit":
            for plan in plans:
                print(shlex.join(plan["command"]), flush=True)
            if args.execute:
                for i, plan in enumerate(plans, 1):
                    print(f"[MATRIX] {i}/{len(plans)} {plan['run_id']}", flush=True)
                    with (args.output / f"{plan['run_id']}.console.log").open("x") as stream:
                        child_env = {**os.environ, "UNITREE_MODEL_DIR": str(args.model_dir), "PYTHONUNBUFFERED": "1"}
                        subprocess.run(plan["command"], cwd=ROOT, env=child_env,
                                       stdout=stream, stderr=subprocess.STDOUT, check=True)
                    required = [args.output / plan["run_id"] / name for name in ("metadata.json", "episodes.csv", "summary.json")]
                    if not all(p.is_file() and p.stat().st_size for p in required):
                        raise RuntimeError(f"child exited without complete evidence: {plan['run_id']}; inspect its console log")
                result = summarize(args.output, plans, registry)
            else:
                result = {"status": "planned", "runs": len(plans), "simulation_executed": False}
        write_json(args.output / "result.json", result)
        manifest = {str(p.relative_to(args.output)): sha256_file(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
        write_json(args.output / "artifacts.sha256.json", manifest)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        write_json(args.output / "failure.json", {"status": "fail", "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
