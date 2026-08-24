#!/usr/bin/env python3

"""Deterministically evaluate an RSL-RL Go2 checkpoint in Isaac Lab."""

import argparse
import csv
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

# The repository's RSL-RL CLI helpers live beside play.py.
RSL_RL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))
import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-Go2-Velocity", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel evaluation environments.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric and use USD I/O.")
parser.add_argument(
    "--episodes_per_env", type=int, default=2, help="Completed episodes required from every environment."
)
parser.add_argument("--output_dir", help="New directory for CSV, JSON, and console output.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Safety limit for vector steps. Default: 3 * episodes_per_env * max_episode_length.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if not args_cli.output_dir:
    parser.error("--output_dir is required")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.episodes_per_env <= 0:
    parser.error("--episodes_per_env must be positive")

output_dir = Path(args_cli.output_dir).expanduser().resolve()
if output_dir.exists():
    parser.error(f"output directory already exists: {output_dir}")

# Evaluation never launches a GUI or renderer.
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


EPISODE_FIELDS = [
    "checkpoint",
    "seed",
    "env_id",
    "episode_index",
    "episode_steps",
    "episode_seconds",
    "reward_sum",
    "mean_xy_velocity_error_mps",
    "rmse_xy_velocity_error_mps",
    "mean_yaw_velocity_error_radps",
    "rmse_yaw_velocity_error_radps",
    "mean_command_xy_speed_mps",
    "mean_command_abs_yaw_rate_radps",
    "mean_action_delta_l2",
    "mean_mechanical_power_w",
    "mechanical_energy_j",
    "time_out",
    "base_contact",
    "bad_orientation",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_zeros(num_envs: int, device: str) -> torch.Tensor:
    return torch.zeros(num_envs, dtype=torch.float64, device=device)


def summarize(rows: list[dict], vector_steps: int, elapsed_s: float, metadata: dict) -> dict:
    def average(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows)

    summary = dict(metadata)
    summary.update(
        {
            "completed_episodes": len(rows),
            "vector_steps": vector_steps,
            "simulated_transitions": vector_steps * metadata["num_envs"],
            "wall_time_seconds": elapsed_s,
            "transitions_per_second": vector_steps * metadata["num_envs"] / elapsed_s,
            "mean_episode_steps": average("episode_steps"),
            "mean_episode_seconds": average("episode_seconds"),
            "mean_reward_sum": average("reward_sum"),
            "mean_xy_velocity_error_mps": average("mean_xy_velocity_error_mps"),
            "rmse_xy_velocity_error_mps": average("rmse_xy_velocity_error_mps"),
            "mean_yaw_velocity_error_radps": average("mean_yaw_velocity_error_radps"),
            "rmse_yaw_velocity_error_radps": average("rmse_yaw_velocity_error_radps"),
            "mean_command_xy_speed_mps": average("mean_command_xy_speed_mps"),
            "mean_command_abs_yaw_rate_radps": average("mean_command_abs_yaw_rate_radps"),
            "mean_action_delta_l2": average("mean_action_delta_l2"),
            "mean_mechanical_power_w": average("mean_mechanical_power_w"),
            "mean_mechanical_energy_j": average("mechanical_energy_j"),
            "time_out_rate": average("time_out"),
            "base_contact_rate": average("base_contact"),
            "bad_orientation_rate": average("bad_orientation"),
        }
    )
    return summary


def main() -> None:
    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed

    # Evaluation must not adapt command or terrain difficulty while comparing checkpoints.
    if hasattr(env_cfg, "curriculum"):
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
        if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
            env_cfg.curriculum.lin_vel_cmd_levels = None
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is not None:
        terrain_generator.curriculum = False

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    print(f"[EVAL] checkpoint: {checkpoint}")
    print(f"[EVAL] num_envs: {env.num_envs}")
    print(f"[EVAL] seed: {args_cli.seed}")
    print(f"[EVAL] episodes_per_env: {args_cli.episodes_per_env}")
    print(f"[EVAL] max_episode_length: {env.max_episode_length}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    observations = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        observations, _ = env.get_observations()

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    terminations = raw_env.termination_manager
    command_manager = raw_env.command_manager
    device = raw_env.device
    num_envs = env.num_envs
    step_dt = float(raw_env.step_dt)

    max_steps = args_cli.max_steps
    if max_steps is None:
        max_steps = 3 * args_cli.episodes_per_env * env.max_episode_length
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    completed = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    reward_sum = tensor_zeros(num_envs, device)
    xy_error_sum = tensor_zeros(num_envs, device)
    xy_error_sq_sum = tensor_zeros(num_envs, device)
    yaw_error_sum = tensor_zeros(num_envs, device)
    yaw_error_sq_sum = tensor_zeros(num_envs, device)
    command_xy_sum = tensor_zeros(num_envs, device)
    command_yaw_sum = tensor_zeros(num_envs, device)
    action_delta_sum = tensor_zeros(num_envs, device)
    action_delta_count = tensor_zeros(num_envs, device)
    power_sum = tensor_zeros(num_envs, device)
    previous_actions: torch.Tensor | None = None
    rows: list[dict] = []

    output_dir.mkdir(parents=True)
    metadata = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": args_cli.seed,
        "num_envs": num_envs,
        "episodes_per_env": args_cli.episodes_per_env,
        "max_episode_length": env.max_episode_length,
        "step_dt_seconds": step_dt,
        "curriculum_updates_enabled": False,
        "rsl_rl_version": version("rsl-rl-lib"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    start = time.perf_counter()
    vector_steps = 0
    with torch.inference_mode():
        while vector_steps < max_steps and torch.any(completed < args_cli.episodes_per_env):
            active = completed < args_cli.episodes_per_env
            commands = command_manager.get_command("base_velocity")
            xy_error = torch.linalg.vector_norm(
                commands[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1
            )
            yaw_error = torch.abs(commands[:, 2] - robot.data.root_ang_vel_b[:, 2])
            command_xy = torch.linalg.vector_norm(commands[:, :2], dim=1)
            command_yaw = torch.abs(commands[:, 2])
            mechanical_power = torch.sum(
                torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1
            )

            actions = policy(observations)
            if previous_actions is not None:
                delta = torch.linalg.vector_norm(actions - previous_actions, dim=1)
                valid_delta = active & (episode_steps > 0)
                action_delta_sum[valid_delta] += delta[valid_delta].double()
                action_delta_count[valid_delta] += 1.0
            previous_actions = actions.clone()

            xy_error_sum[active] += xy_error[active].double()
            xy_error_sq_sum[active] += torch.square(xy_error[active]).double()
            yaw_error_sum[active] += yaw_error[active].double()
            yaw_error_sq_sum[active] += torch.square(yaw_error[active]).double()
            command_xy_sum[active] += command_xy[active].double()
            command_yaw_sum[active] += command_yaw[active].double()
            power_sum[active] += mechanical_power[active].double()
            episode_steps[active] += 1

            observations, rewards, dones, _ = env.step(actions)
            reward_sum[active] += rewards[active].double()
            vector_steps += 1

            done_ids = torch.nonzero((dones > 0) & active, as_tuple=False).flatten()
            if done_ids.numel() == 0:
                continue

            time_out = terminations.get_term("time_out")
            base_contact = terminations.get_term("base_contact")
            bad_orientation = terminations.get_term("bad_orientation")

            for env_id_tensor in done_ids:
                env_id = int(env_id_tensor.item())
                steps = int(episode_steps[env_id].item())
                delta_count = float(action_delta_count[env_id].item())
                rows.append(
                    {
                        "checkpoint": checkpoint.name,
                        "seed": args_cli.seed,
                        "env_id": env_id,
                        "episode_index": int(completed[env_id].item()) + 1,
                        "episode_steps": steps,
                        "episode_seconds": steps * step_dt,
                        "reward_sum": float(reward_sum[env_id].item()),
                        "mean_xy_velocity_error_mps": float(xy_error_sum[env_id].item() / steps),
                        "rmse_xy_velocity_error_mps": float(
                            (xy_error_sq_sum[env_id].item() / steps) ** 0.5
                        ),
                        "mean_yaw_velocity_error_radps": float(yaw_error_sum[env_id].item() / steps),
                        "rmse_yaw_velocity_error_radps": float(
                            (yaw_error_sq_sum[env_id].item() / steps) ** 0.5
                        ),
                        "mean_command_xy_speed_mps": float(command_xy_sum[env_id].item() / steps),
                        "mean_command_abs_yaw_rate_radps": float(command_yaw_sum[env_id].item() / steps),
                        "mean_action_delta_l2": float(
                            action_delta_sum[env_id].item() / delta_count if delta_count else 0.0
                        ),
                        "mean_mechanical_power_w": float(power_sum[env_id].item() / steps),
                        "mechanical_energy_j": float(power_sum[env_id].item() * step_dt),
                        "time_out": int(time_out[env_id].item()),
                        "base_contact": int(base_contact[env_id].item()),
                        "bad_orientation": int(bad_orientation[env_id].item()),
                    }
                )
                completed[env_id] += 1

            episode_steps[done_ids] = 0
            reward_sum[done_ids] = 0
            xy_error_sum[done_ids] = 0
            xy_error_sq_sum[done_ids] = 0
            yaw_error_sum[done_ids] = 0
            yaw_error_sq_sum[done_ids] = 0
            command_xy_sum[done_ids] = 0
            command_yaw_sum[done_ids] = 0
            action_delta_sum[done_ids] = 0
            action_delta_count[done_ids] = 0

    elapsed = time.perf_counter() - start
    env.close()

    expected_episodes = num_envs * args_cli.episodes_per_env
    if len(rows) != expected_episodes:
        raise RuntimeError(
            f"Evaluation stopped with {len(rows)}/{expected_episodes} episodes after {vector_steps} vector steps"
        )

    with (output_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, vector_steps, elapsed, metadata)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("[EVAL] completed successfully")
    print(json.dumps(summary, indent=2))
    print(f"[EVAL] episodes_csv: {output_dir / 'episodes.csv'}")
    print(f"[EVAL] summary_json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
