#!/usr/bin/env python3
"""Run frozen explicit-state ONNX policy in MuJoCo, or verify observation replay."""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/unitree_rl_lab"))
import mujoco
import numpy as np
from unitree_rl_lab.utils.go2_mujoco import Go2Mujoco, PolicyRuntime, wrap_angle
from unitree_rl_lab.utils.teacher_bank import sha256_file


def verify_replay(sim, sequence_path):
    sequence = np.load(sequence_path, allow_pickle=False)
    if sequence["joint_names"].tolist() != sim.names:
        raise ValueError("replay joint order mismatch")
    maximum = np.zeros(sim.entry["policy_dim"])
    count = 0
    indices = np.unique(np.linspace(0, len(sequence["observations"])-1, 200).astype(int))
    for step in indices:
        for batch in range(sequence["observations"].shape[1]):
            sim.reset()
            sim.data.qpos[:3] = sequence["root_pos_w"][step, batch]
            sim.data.qpos[3:7] = sequence["root_quat_w"][step, batch]
            sim.data.qpos[sim.qadr] = sequence["joint_pos"][step, batch]
            sim.data.qvel[3:6] = sequence["root_ang_vel_b"][step, batch]
            sim.data.qvel[sim.vadr] = sequence["joint_vel"][step, batch]
            sim.last_action = sequence["last_action"][step, batch].copy()
            sim.steps = int(sequence["episode_steps"][step, batch])
            mujoco.mj_forward(sim.model, sim.data)
            actual = sim.observation(sequence["commands"][step, batch])[0]
            expected = sequence["observations"][step, batch]
            maximum = np.maximum(maximum, np.abs(actual-expected))
            count += 1
    offset = 0
    errors = {}
    for term in sim.entry["observations"]["policy"]:
        errors[term["name"]] = float(maximum[offset:offset+term["dim"]].max())
        offset += term["dim"]
    return {"status": "pass" if maximum.max() < 1e-5 else "fail", "state_samples": count,
            "max_abs_error": float(maximum.max()), "per_term_max_abs_error": errors,
            "scope": "same pose/joints/commands replay on flat ground; not dynamics equivalence",
            "sequence_sha256": sha256_file(sequence_path)}


def rollout(sim, runtime, profile, seed, duration):
    rng = np.random.default_rng(seed)
    sim.reset(yaw=float(rng.uniform(-.1, .1)), joint_jitter=rng.uniform(-.01, .01, 12))
    initial_xy = sim.data.xpos[sim.base_id, :2].copy()
    initial_yaw = sim.yaw
    delta = {"forward": 0., "speed_steps": 0., "turn_left": np.pi/4, "turn_right": -np.pi/4}[profile]
    heading_target = wrap_angle(initial_yaw+delta)
    rows, latencies = [], []
    reason = "time_out"
    for step in range(round(duration/sim.dt)):
        t = step*sim.dt
        vx = (.2 if t < 5 else .5 if t < 10 else .35) if profile == "speed_steps" else .35
        command = sim.command(vx, heading_target)
        velocity, omega = sim.velocity_body()
        begin = time.perf_counter()
        observation = sim.observation(command)
        inference_start = time.perf_counter()
        action = runtime.act(observation, reset=step == 0)
        end = time.perf_counter()
        latencies.append(((end-inference_start)*1000, (end-begin)*1000))
        rows.append({"step": step, "time_s": t, "command_vx": vx, "command_yaw_rate": float(command[2]),
                     "actual_vx": float(velocity[0]), "actual_vy": float(velocity[1]),
                     "xy_error_mps": float(np.linalg.norm(command[:2]-velocity[:2])),
                     "yaw_error_radps": float(abs(command[2]-omega[2])),
                     "heading_error_rad": float(abs(wrap_angle(heading_target-sim.yaw))),
                     "base_x": float(sim.data.xpos[sim.base_id, 0]), "base_y": float(sim.data.xpos[sim.base_id, 1]),
                     "base_z": float(sim.data.xpos[sim.base_id, 2])})
        sim.advance(action)
        failure = sim.termination()
        if failure:
            reason = failure
            break
    latency = np.asarray(latencies)
    result = {"seed": seed, "profile": profile, "duration_s": len(rows)*sim.dt, "termination": reason,
              "mean_xy_error_mps": float(np.mean([r["xy_error_mps"] for r in rows])),
              "mean_heading_error_rad": float(np.mean([r["heading_error_rad"] for r in rows])),
              "final_heading_error_rad": float(abs(wrap_angle(heading_target-sim.yaw))),
              "net_displacement_m": float(np.linalg.norm(sim.data.xpos[sim.base_id, :2]-initial_xy)),
              "inference_ms_p50_p95_p99": np.percentile(latency[:, 0], [50, 95, 99]).tolist(),
              "scan_observation_inference_ms_p50_p95_p99": np.percentile(latency[:, 1], [50, 95, 99]).tolist(),
              "observation_inference_deadline_misses": int((latency[:, 1] > sim.dt*1000).sum())}
    return rows, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=ROOT.parent / "mujoco_menagerie/unitree_go2/scene.xml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, help="Verify same-state Isaac observation replay before rollouts")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=("forward", "speed_steps", "turn_left", "turn_right"),
                        default=["forward", "speed_steps", "turn_left", "turn_right"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--duration", type=float, default=20.)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("require a new output directory and positive finite duration")
    if args.replay_only and not args.sequence:
        parser.error("--replay-only requires --sequence")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.profiles)) != len(args.profiles):
        parser.error("duplicate seeds/profiles")
    args.output.mkdir(parents=True)
    runtime = PolicyRuntime(args.policy)
    sim = Go2Mujoco(args.scene, runtime.manifest)
    mujoco.mj_saveLastXML(str(args.output / "effective_model.xml"), sim.model)
    try:
        replay = verify_replay(sim, args.sequence) if args.sequence else None
        if replay:
            (args.output / "observation_replay.json").write_text(json.dumps(replay, indent=2)+"\n")
            if replay["status"] != "pass":
                raise RuntimeError(f"observation replay failed: {replay}")
        results = []
        if not args.replay_only:
            for profile in args.profiles:
                for seed in args.seeds:
                    rows, result = rollout(sim, runtime, profile, seed, args.duration)
                    with (args.output / f"{profile}_seed{seed}.csv").open("x", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    results.append(result)
                    print(json.dumps(result), flush=True)
        passed = all(r["termination"] == "time_out" and r["mean_xy_error_mps"] <= .12
                     and r["final_heading_error_rad"] <= .15 for r in results)
        report = {"status": "pass" if passed else "quality_gate_failed", "teacher": runtime.manifest["teacher"],
                  "policy_sha256": runtime.manifest["onnx_sha256"], "mujoco_version": mujoco.__version__,
                  "scene": str(args.scene.resolve()), "scene_sha256": sha256_file(args.scene),
                  "menagerie_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.scene.parent, text=True).strip(),
                  "control_dt": sim.dt, "physics_dt": sim.model.opt.timestep,
                  "runtime_model_changes": {"joint_passive_damping": 0., "joint_frictionloss": .01,
                                            "actuator": "frozen Unitree torque-speed curve and external PD",
                                            "armature": "Menagerie values retained"},
                  "gate": {"no_safety_termination": True, "max_mean_xy_error_mps": .12, "max_final_heading_error_rad": .15},
                  "scope": "flat-ground frozen-weight transfer; random initial yaw/joints, no observation noise",
                  "observation_replay": replay, "episodes": results}
        (args.output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        files = {str(p.relative_to(args.output)): sha256_file(p) for p in args.output.rglob('*') if p.is_file()}
        (args.output / "artifacts.sha256.json").write_text(json.dumps(files, indent=2)+"\n")
        if not passed:
            raise SystemExit(2)
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({"error": str(exc)}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
