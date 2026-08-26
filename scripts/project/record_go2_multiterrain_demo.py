#!/usr/bin/env python3

"""Record one auditable fixed-terrain Go2 MultiTerrain demonstration clip."""

import argparse
import csv
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

RSL_RL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))
import cli_args  # noqa: E402


TERRAIN_NAMES = (
    "flat",
    "random_rough",
    "pyramid_slope",
    "inverted_pyramid_slope",
    "pyramid_stairs",
    "inverted_pyramid_stairs",
    "boxes",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Unitree-Go2-Velocity-MultiTerrain",
    choices=("Unitree-Go2-Velocity-MultiTerrain", "Unitree-Go2-Velocity-Rough"),
)
parser.add_argument("--terrain_type", required=True, choices=TERRAIN_NAMES)
parser.add_argument("--terrain_level", type=int, default=5)
parser.add_argument("--training_seed", type=int, required=True)
parser.add_argument("--seed", type=int, default=42, help="Evaluation scenario seed, not training seed.")
parser.add_argument("--video_length", type=int, default=1000, help="Clip length in simulation steps.")
parser.add_argument("--warmup_steps", type=int, default=100, help="Steps excluded from steady-state metrics.")
parser.add_argument("--output_dir", required=True, help="New per-terrain output directory.")
parser.add_argument("--command_x", type=float, default=0.35)
parser.add_argument("--command_y", type=float, default=0.0)
parser.add_argument("--command_yaw", type=float, default=0.0)
parser.add_argument(
    "--camera_mode",
    choices=("terrain_fixed", "asset_follow"),
    default="terrain_fixed",
    help="Fixed camera centered on the assigned terrain origin, or guaranteed robot-follow camera.",
)
parser.add_argument("--disable_fabric", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.task == "Unitree-Go2-Velocity-Rough" and args_cli.terrain_type not in {"flat", "random_rough"}:
    parser.error("the rough task only supports flat or random_rough")
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if not 0 <= args_cli.terrain_level <= 9:
    parser.error("--terrain_level must be in [0, 9]")
if args_cli.seed < 0:
    parser.error("--seed must be non-negative")
if args_cli.training_seed < 0:
    parser.error("--training_seed must be non-negative")
if args_cli.video_length <= 0:
    parser.error("--video_length must be positive")
if not 0 <= args_cli.warmup_steps < args_cli.video_length:
    parser.error("--warmup_steps must be in [0, video_length)")

output_dir = Path(args_cli.output_dir).expanduser().resolve()
if output_dir.exists():
    parser.error(f"output directory already exists: {output_dir}")

args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terrain_column_names(generator_cfg) -> list[str]:
    names = list(generator_cfg.sub_terrains)
    proportions = [float(cfg.proportion) for cfg in generator_cfg.sub_terrains.values()]
    total = sum(proportions)
    cumulative = []
    running = 0.0
    for value in proportions:
        running += value / total
        cumulative.append(running)
    result = []
    for column in range(int(generator_cfg.num_cols)):
        sample = column / int(generator_cfg.num_cols) + 0.001
        index = next(i for i, upper in enumerate(cumulative) if sample < upper)
        result.append(names[index])
    return result


def main() -> None:
    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None
    generator_cfg = env_cfg.scene.terrain.terrain_generator
    generator_cfg.seed = args_cli.seed
    generator_cfg.curriculum = True
    generator_cfg.num_rows = 10
    generator_cfg.num_cols = 20
    env_cfg.scene.terrain.max_init_terrain_level = 9
    ranges = env_cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
    ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
    ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
    ranges.heading = None
    env_cfg.viewer.origin_type = "asset_root" if args_cli.camera_mode == "asset_follow" else "world"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (5.5, 5.5, 3.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.35)

    output_dir.mkdir(parents=True)
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw_env = base_env.unwrapped
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)

    column_names = terrain_column_names(generator_cfg)
    matching_columns = [i for i, name in enumerate(column_names) if name == args_cli.terrain_type]
    if not matching_columns:
        raise RuntimeError(f"no generated column for {args_cli.terrain_type}")
    terrain_column = matching_columns[len(matching_columns) // 2]
    terrain = raw_env.scene.terrain
    terrain.terrain_types[:] = terrain_column
    terrain.terrain_levels[:] = args_cli.terrain_level
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    base_env.reset()

    robot_start = raw_env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
    if args_cli.camera_mode == "terrain_fixed":
        camera_eye = [robot_start[0] + 3.0, robot_start[1] + 5.0, robot_start[2] + 2.3]
        camera_target = [robot_start[0] + 1.5, robot_start[1], robot_start[2] - 0.15]
        raw_env.sim.set_camera_view(eye=camera_eye, target=camera_target)
    else:
        camera_eye = list(env_cfg.viewer.eye)
        camera_target = list(env_cfg.viewer.lookat)
    print(f"[DEMO] robot_start_world={robot_start}")
    print(f"[DEMO] camera_mode={args_cli.camera_mode} eye={camera_eye} target={camera_target}")

    video_dir = output_dir / "video"
    base_env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        name_prefix=f"{args_cli.terrain_type}_level{args_cli.terrain_level}",
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=raw_env.device)

    observations = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        observations, _ = env.get_observations()

    reward_sum = 0.0
    xy_error_sum = 0.0
    yaw_error_sum = 0.0
    termination_counts = {"time_out": 0, "base_contact": 0, "bad_orientation": 0}
    samples = []
    cumulative_xy_distance = 0.0
    start = time.perf_counter()
    for step in range(args_cli.video_length):
        actual_column = int(terrain.terrain_types[0].item())
        actual_level = int(terrain.terrain_levels[0].item())
        if actual_column != terrain_column or actual_level != args_cli.terrain_level:
            raise RuntimeError(
                f"terrain assignment changed: expected {terrain_column}/{args_cli.terrain_level}, "
                f"got {actual_column}/{actual_level}"
            )
        command = raw_env.command_manager.get_command("base_velocity")
        velocity = raw_env.scene["robot"].data.root_lin_vel_b
        angular_velocity = raw_env.scene["robot"].data.root_ang_vel_b
        xy_error = float(torch.linalg.vector_norm(command[0, :2] - velocity[0, :2]).item())
        yaw_error = float(torch.abs(command[0, 2] - angular_velocity[0, 2]).item())
        xy_error_sum += xy_error
        yaw_error_sum += yaw_error
        position_before = raw_env.scene["robot"].data.root_pos_w[0, :2].clone()
        with torch.inference_mode():
            actions = policy(observations)
            observations, rewards, dones, _ = env.step(actions)
        position_after = raw_env.scene["robot"].data.root_pos_w[0, :2]
        reward_sum += float(rewards[0].item())
        done = bool(dones[0].item())
        if not done:
            cumulative_xy_distance += float(torch.linalg.vector_norm(position_after - position_before).item())
        row = {
            "step": step,
            "time_seconds": step * float(raw_env.step_dt),
            "command_x_mps": float(command[0, 0].item()),
            "command_y_mps": float(command[0, 1].item()),
            "command_yaw_radps": float(command[0, 2].item()),
            "actual_x_mps": float(velocity[0, 0].item()),
            "actual_y_mps": float(velocity[0, 1].item()),
            "actual_yaw_radps": float(angular_velocity[0, 2].item()),
            "xy_error_mps": xy_error,
            "yaw_error_radps": yaw_error,
            "cumulative_xy_distance_m": cumulative_xy_distance,
            "done": int(done),
        }
        samples.append(row)
        if done:
            terminations = raw_env.termination_manager
            for name in termination_counts:
                termination_counts[name] += int(terminations.get_term(name)[0].item())

    elapsed = time.perf_counter() - start
    env.close()
    videos = sorted(video_dir.glob("*.mp4"))
    if len(videos) != 1 or videos[0].stat().st_size == 0:
        raise RuntimeError(f"expected one non-empty video, found {videos}")

    samples_path = output_dir / "tracking_timeseries.csv"
    with samples_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    steady_samples = samples[args_cli.warmup_steps :]

    metadata = {
        "task": args_cli.task,
        "training_seed": args_cli.training_seed,
        "evaluation_scenario_seed": args_cli.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "terrain_type": args_cli.terrain_type,
        "terrain_column": terrain_column,
        "terrain_level": args_cli.terrain_level,
        "robot_start_world": robot_start,
        "camera": {"mode": args_cli.camera_mode, "eye": camera_eye, "target": camera_target},
        "command": {"lin_vel_x": args_cli.command_x, "lin_vel_y": args_cli.command_y, "ang_vel_z": args_cli.command_yaw},
        "video_steps": args_cli.video_length,
        "warmup_steps": args_cli.warmup_steps,
        "step_dt": float(raw_env.step_dt),
        "simulated_seconds": args_cli.video_length * float(raw_env.step_dt),
        "wall_time_seconds": elapsed,
        "reward_sum": reward_sum,
        "mean_xy_velocity_error_mps": xy_error_sum / args_cli.video_length,
        "mean_yaw_velocity_error_radps": yaw_error_sum / args_cli.video_length,
        "post_warmup_mean_xy_velocity_error_mps": sum(row["xy_error_mps"] for row in steady_samples)
        / len(steady_samples),
        "post_warmup_mean_yaw_velocity_error_radps": sum(row["yaw_error_radps"] for row in steady_samples)
        / len(steady_samples),
        "post_warmup_mean_actual_x_velocity_mps": sum(row["actual_x_mps"] for row in steady_samples)
        / len(steady_samples),
        "cumulative_xy_distance_m": cumulative_xy_distance,
        "terminations": termination_counts,
        "tracking_timeseries": str(samples_path),
        "tracking_timeseries_sha256": sha256_file(samples_path),
        "video": str(videos[0]),
        "video_size_bytes": videos[0].stat().st_size,
        "video_sha256": sha256_file(videos[0]),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "video.sha256").write_text(f"{metadata['video_sha256']}  {videos[0].relative_to(output_dir)}\n")
    print("[DEMO] completed successfully")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
