#!/usr/bin/env python3

"""Deterministically evaluate an RSL-RL Go2 checkpoint in Isaac Lab."""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
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
parser.add_argument(
    "--terrain_protocol",
    choices=("legacy", "rough_stratified"),
    default="legacy",
    help=(
        "Terrain evaluation protocol. 'legacy' preserves the original randomized-terrain "
        "evaluation behavior; 'rough_stratified' uses the fixed 10x20 Go2 rough grid."
    ),
)
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
if args_cli.terrain_protocol == "rough_stratified":
    if args_cli.task != "Unitree-Go2-Velocity-Rough":
        parser.error("--terrain_protocol rough_stratified requires --task Unitree-Go2-Velocity-Rough")
    if args_cli.num_envs < 20 or args_cli.num_envs % 2:
        parser.error("rough_stratified requires an even --num_envs of at least 20")
    if args_cli.seed < 0:
        parser.error("rough_stratified requires a non-negative --seed")

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
    "terrain_protocol",
    "terrain_type",
    "terrain_column",
    "terrain_level",
    "terrain_difficulty_lower",
    "terrain_difficulty_upper",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_zeros(num_envs: int, device: str) -> torch.Tensor:
    return torch.zeros(num_envs, dtype=torch.float64, device=device)


def configure_terrain_protocol(env_cfg) -> None:
    """Freeze runtime curricula and configure the requested terrain generation protocol."""

    if hasattr(env_cfg, "curriculum"):
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
        if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
            env_cfg.curriculum.lin_vel_cmd_levels = None

    terrain_cfg = env_cfg.scene.terrain
    generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
    if generator_cfg is None:
        if args_cli.terrain_protocol == "rough_stratified":
            raise ValueError("rough_stratified requires a terrain generator")
        return

    if args_cli.terrain_protocol == "legacy":
        generator_cfg.curriculum = False
        return

    sub_terrains = generator_cfg.sub_terrains
    if list(sub_terrains) != ["flat", "random_rough"]:
        raise ValueError(
            "rough_stratified requires ordered sub-terrains: ['flat', 'random_rough']; "
            f"got {list(sub_terrains)}"
        )
    proportions = [float(cfg.proportion) for cfg in sub_terrains.values()]
    proportion_sum = sum(proportions)
    normalized = [value / proportion_sum for value in proportions]
    if any(abs(value - 0.5) > 1.0e-9 for value in normalized):
        raise ValueError(f"rough_stratified requires a 50/50 terrain mix; got {normalized}")

    generator_cfg.seed = args_cli.seed
    generator_cfg.curriculum = True
    generator_cfg.num_rows = 10
    generator_cfg.num_cols = 20
    terrain_cfg.max_init_terrain_level = generator_cfg.num_rows - 1


def terrain_column_names(generator_cfg) -> list[str]:
    """Mirror TerrainGenerator's column-to-sub-terrain assignment."""

    if generator_cfg is None:
        return []
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


def apply_rough_stratification(raw_env, num_envs: int) -> bool:
    """Assign paired flat/rough environments across all difficulty levels."""

    if args_cli.terrain_protocol != "rough_stratified":
        return False

    terrain = raw_env.scene.terrain
    generator_cfg = raw_env.cfg.scene.terrain.terrain_generator
    num_rows = int(generator_cfg.num_rows)
    num_cols = int(generator_cfg.num_cols)
    columns_per_type = num_cols // 2
    envs_per_type = num_envs // 2
    device = terrain.terrain_levels.device

    local_ids = torch.arange(envs_per_type, device=device, dtype=torch.long)
    local_columns = torch.div(
        local_ids * columns_per_type, envs_per_type, rounding_mode="floor"
    )
    paired_levels = torch.remainder(local_ids + args_cli.seed, num_rows)
    terrain.terrain_types[:] = torch.cat((local_columns, local_columns + columns_per_type))
    terrain.terrain_levels[:] = torch.cat((paired_levels, paired_levels))
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return True


def build_terrain_assignment(env_cfg, raw_env, num_envs: int) -> tuple[list[dict], dict]:
    """Capture each environment's fixed terrain assignment and its protocol metadata."""

    generator_cfg = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if args_cli.terrain_protocol != "rough_stratified":
        assignments = [
            {
                "terrain_protocol": args_cli.terrain_protocol,
                "terrain_type": "",
                "terrain_column": "",
                "terrain_level": "",
                "terrain_difficulty_lower": "",
                "terrain_difficulty_upper": "",
            }
            for _ in range(num_envs)
        ]
        return assignments, {
            "protocol": args_cli.terrain_protocol,
            "protocol_version": 1,
            "generator_present": generator_cfg is not None,
            "generation_curriculum_enabled": bool(generator_cfg.curriculum)
            if generator_cfg is not None
            else False,
            "runtime_curriculum_updates_enabled": False,
        }

    terrain = raw_env.scene.terrain
    column_names = terrain_column_names(generator_cfg)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if generator_cfg is None or terrain_types is None or terrain_levels is None:
        raise RuntimeError("rough_stratified terrain assignments are unavailable")

    type_indices = [int(value) for value in terrain_types.detach().cpu().tolist()]
    levels = [int(value) for value in terrain_levels.detach().cpu().tolist()]
    if len(type_indices) != num_envs or len(levels) != num_envs:
        raise RuntimeError("terrain assignment size does not match num_envs")

    num_rows = int(generator_cfg.num_rows)
    difficulty_min, difficulty_max = (float(value) for value in generator_cfg.difficulty_range)
    difficulty_span = difficulty_max - difficulty_min
    assignments = []
    for type_index, level in zip(type_indices, levels):
        if not 0 <= type_index < len(column_names):
            raise RuntimeError(f"invalid terrain type index: {type_index}")
        lower = difficulty_min + difficulty_span * level / num_rows
        upper = difficulty_min + difficulty_span * (level + 1) / num_rows
        assignments.append(
            {
                "terrain_protocol": args_cli.terrain_protocol,
                "terrain_type": column_names[type_index],
                "terrain_column": type_index,
                "terrain_level": level,
                "terrain_difficulty_lower": lower,
                "terrain_difficulty_upper": upper,
            }
        )

    type_counts = Counter(item["terrain_type"] for item in assignments)
    level_counts = Counter(str(item["terrain_level"]) for item in assignments)
    type_level_counts: dict[str, Counter] = defaultdict(Counter)
    for item in assignments:
        type_level_counts[item["terrain_type"]][str(item["terrain_level"])] += 1

    sub_terrain_proportions = {
        name: float(cfg.proportion) for name, cfg in generator_cfg.sub_terrains.items()
    }
    sub_terrain_parameters = {}
    for name, cfg in generator_cfg.sub_terrains.items():
        parameters = {
            "class": f"{type(cfg).__module__}.{type(cfg).__name__}",
            "proportion": float(cfg.proportion),
        }
        for attribute in ("noise_range", "noise_step", "border_width"):
            if hasattr(cfg, attribute):
                value = getattr(cfg, attribute)
                parameters[attribute] = list(value) if isinstance(value, tuple) else value
        sub_terrain_parameters[name] = parameters
    terrain_metadata = {
        "protocol": args_cli.terrain_protocol,
        "protocol_version": 1,
        "generator_present": True,
        "generator_seed": generator_cfg.seed,
        "generation_curriculum_enabled": bool(generator_cfg.curriculum),
        "runtime_curriculum_updates_enabled": False,
        "num_rows": num_rows,
        "num_cols": int(generator_cfg.num_cols),
        "max_init_terrain_level": int(env_cfg.scene.terrain.max_init_terrain_level),
        "difficulty_range": [difficulty_min, difficulty_max],
        "difficulty_value_note": (
            "Each episode records the row's inclusive-lower/exclusive-upper difficulty interval; "
            "the generated tile uses a seeded uniform sample inside that interval."
        ),
        "env_assignment_rule": (
            "First half flat and second half random_rough; paired environment offsets share the "
            "same deterministic level. All levels 0-9 occur in each type, with seed selecting "
            "which levels receive the two extra environments when num_envs=64."
        ),
        "sub_terrain_proportions": sub_terrain_proportions,
        "sub_terrain_parameters": sub_terrain_parameters,
        "column_terrain_types": column_names,
        "initial_env_distribution": {
            "by_type": dict(sorted(type_counts.items())),
            "by_level": dict(sorted(level_counts.items(), key=lambda item: int(item[0]))),
            "by_type_and_level": {
                terrain_type: dict(sorted(counts.items(), key=lambda item: int(item[0])))
                for terrain_type, counts in sorted(type_level_counts.items())
            },
        },
    }
    return assignments, terrain_metadata


def aggregate_rows(rows: list[dict]) -> dict:
    """Compute episode-weighted metrics for one non-empty set of episodes."""

    def average(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows)

    return {
        "completed_episodes": len(rows),
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


def summarize(rows: list[dict], vector_steps: int, elapsed_s: float, metadata: dict) -> dict:
    summary = dict(metadata)
    summary.update(aggregate_rows(rows))
    summary.update(
        {
            "vector_steps": vector_steps,
            "simulated_transitions": vector_steps * metadata["num_envs"],
            "wall_time_seconds": elapsed_s,
            "transitions_per_second": vector_steps * metadata["num_envs"] / elapsed_s,
        }
    )

    if any(row["terrain_type"] for row in rows):
        by_type: dict[str, list[dict]] = defaultdict(list)
        by_level: dict[int, list[dict]] = defaultdict(list)
        by_type_and_level: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            terrain_type = str(row["terrain_type"])
            terrain_level = int(row["terrain_level"])
            by_type[terrain_type].append(row)
            by_level[terrain_level].append(row)
            by_type_and_level[terrain_type][terrain_level].append(row)

        summary["terrain_metrics"] = {
            "by_type": {
                name: aggregate_rows(group) for name, group in sorted(by_type.items())
            },
            "by_level": {
                str(level): aggregate_rows(group) for level, group in sorted(by_level.items())
            },
            "by_type_and_level": {
                name: {
                    str(level): aggregate_rows(group)
                    for level, group in sorted(level_groups.items())
                }
                for name, level_groups in sorted(by_type_and_level.items())
            },
        }
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
    configure_terrain_protocol(env_cfg)

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if apply_rough_stratification(base_env.unwrapped, args_cli.num_envs):
        # The importer initially sampled origins before stratification. Reset once so all robot
        # and sensor states use the explicit paired terrain origins before evaluation starts.
        base_env.reset()
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    print(f"[EVAL] checkpoint: {checkpoint}")
    print(f"[EVAL] num_envs: {env.num_envs}")
    print(f"[EVAL] seed: {args_cli.seed}")
    print(f"[EVAL] terrain_protocol: {args_cli.terrain_protocol}")
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
    terrain_assignments, terrain_metadata = build_terrain_assignment(env_cfg, raw_env, num_envs)

    if args_cli.terrain_protocol == "rough_stratified":
        distribution = terrain_metadata["initial_env_distribution"]
        type_counts = distribution["by_type"]
        expected_per_type = num_envs // 2
        if type_counts != {"flat": expected_per_type, "random_rough": expected_per_type}:
            raise RuntimeError(
                "rough_stratified did not produce the required 50/50 environment assignment: "
                f"{type_counts}"
            )
        expected_levels = {str(level) for level in range(10)}
        for terrain_type, level_counts in distribution["by_type_and_level"].items():
            if set(level_counts) != expected_levels:
                raise RuntimeError(
                    f"rough_stratified did not cover levels 0-9 for {terrain_type}: "
                    f"{sorted(level_counts, key=int)}"
                )

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
        "terrain_protocol": args_cli.terrain_protocol,
        "terrain": terrain_metadata,
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
                if args_cli.terrain_protocol == "rough_stratified":
                    actual_column = int(raw_env.scene.terrain.terrain_types[env_id].item())
                    actual_level = int(raw_env.scene.terrain.terrain_levels[env_id].item())
                    expected_assignment = terrain_assignments[env_id]
                    if (
                        actual_column != expected_assignment["terrain_column"]
                        or actual_level != expected_assignment["terrain_level"]
                    ):
                        raise RuntimeError(
                            f"terrain assignment changed for env {env_id}: "
                            f"expected column/level "
                            f"{expected_assignment['terrain_column']}/{expected_assignment['terrain_level']}, "
                            f"got {actual_column}/{actual_level}"
                        )
                steps = int(episode_steps[env_id].item())
                delta_count = float(action_delta_count[env_id].item())
                rows.append(
                    {
                        "checkpoint": checkpoint.name,
                        "seed": args_cli.seed,
                        **terrain_assignments[env_id],
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
