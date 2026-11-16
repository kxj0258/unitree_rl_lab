#!/usr/bin/env python3

"""Deterministically evaluate an RSL-RL Go2 checkpoint in Isaac Lab."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
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
    choices=(
        "legacy",
        "rough_stratified",
        "multiterrain_stratified",
        "v5_foundation_stratified",
        "v5_stairs_stratified",
        "v5_boxes_stratified",
        "teacher_v1_stratified",
        "teacher_v2_stratified",
        "teacher_v3_stratified",
    ),
    default="legacy",
    help=(
        "Terrain evaluation protocol. 'legacy' preserves the original randomized-terrain "
        "evaluation behavior; 'rough_stratified' uses the fixed 10x20 Go2 rough grid; "
        "'multiterrain_stratified' covers all levels of all seven Go2 MultiTerrain types; "
        "'v5_foundation_stratified' covers all eight levels of the four v5 foundation terrains; "
        "'v5_stairs_stratified' covers all eight levels of the two rehearsal and two stairs terrains; "
        "'v5_boxes_stratified' evaluates only boxes while covering all eight levels."
        " 'teacher_v1_stratified' covers all ten levels of one Terrain Teacher v1 task."
    ),
)
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric and use USD I/O.")
parser.add_argument(
    "--episodes_per_env", type=int, default=2, help="Completed episodes required from every environment."
)
parser.add_argument("--output_dir", help="New directory for CSV, JSON, and console output.")
parser.add_argument("--teacher_registry", help="Optional frozen seven-teacher registry for adapter acceptance.")
parser.add_argument("--teacher_id", help="Terrain key in --teacher_registry; both options must be supplied.")
parser.add_argument("--capture_policy_sequence", action="store_true", help="Save first two environments' input/action/reset sequence and states for Sim2Sim replay.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Safety limit for vector steps. Default: 3 * episodes_per_env * max_episode_length.",
)
parser.add_argument("--fixed_command_x", type=float, default=None, help="Fix commanded body-frame x velocity.")
parser.add_argument("--fixed_command_y", type=float, default=None, help="Fix commanded body-frame y velocity.")
parser.add_argument("--fixed_command_yaw", type=float, default=None, help="Fix commanded yaw rate.")
parser.add_argument(
    "--fixed_heading",
    type=float,
    default=None,
    help="Use heading feedback toward this fixed world-frame yaw instead of a fixed yaw-rate command.",
)
parser.add_argument(
    "--fixed_relative_heading",
    type=float,
    default=None,
    help="Lock a Teacher v2 heading target relative to yaw at command resampling.",
)
parser.add_argument(
    "--fixed_initial_yaw",
    type=float,
    default=None,
    help="Fix the reset yaw for every evaluation episode.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if not args_cli.output_dir:
    parser.error("--output_dir is required")
if bool(args_cli.teacher_registry) != bool(args_cli.teacher_id):
    parser.error("--teacher_registry and --teacher_id must be supplied together")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.episodes_per_env <= 0:
    parser.error("--episodes_per_env must be positive")
fixed_command_values = (args_cli.fixed_command_x, args_cli.fixed_command_y, args_cli.fixed_command_yaw)
if any(value is not None for value in fixed_command_values) and not all(
    value is not None for value in fixed_command_values
):
    parser.error("--fixed_command_x, --fixed_command_y, and --fixed_command_yaw must be provided together")
if any(value is not None and not math.isfinite(value) for value in fixed_command_values):
    parser.error("fixed command values must be finite")
if args_cli.fixed_heading is not None and not math.isfinite(args_cli.fixed_heading):
    parser.error("--fixed_heading must be finite")
if args_cli.fixed_initial_yaw is not None and not math.isfinite(args_cli.fixed_initial_yaw):
    parser.error("--fixed_initial_yaw must be finite")
if args_cli.fixed_relative_heading is not None and not math.isfinite(args_cli.fixed_relative_heading):
    parser.error("--fixed_relative_heading must be finite")
if args_cli.fixed_heading is not None and args_cli.fixed_relative_heading is not None:
    parser.error("--fixed_heading and --fixed_relative_heading are mutually exclusive")
if args_cli.fixed_heading is not None and args_cli.fixed_command_yaw not in {None, 0.0}:
    parser.error("--fixed_command_yaw must be 0 when --fixed_heading is used")
if args_cli.terrain_protocol == "rough_stratified":
    if args_cli.task != "Unitree-Go2-Velocity-Rough":
        parser.error("--terrain_protocol rough_stratified requires --task Unitree-Go2-Velocity-Rough")
    if args_cli.num_envs < 20 or args_cli.num_envs % 2:
        parser.error("rough_stratified requires an even --num_envs of at least 20")
    if args_cli.seed < 0:
        parser.error("rough_stratified requires a non-negative --seed")
if args_cli.terrain_protocol == "multiterrain_stratified":
    if args_cli.task not in {
        "Unitree-Go2-Velocity-MultiTerrain",
        "Unitree-Go2-Velocity-MultiTerrain-HeightScan",
        "Unitree-Go2-Velocity-ProxyStudent-HeightScan",
    }:
        parser.error(
            "--terrain_protocol multiterrain_stratified requires "
            "--task Unitree-Go2-Velocity-MultiTerrain or "
            "Unitree-Go2-Velocity-MultiTerrain-HeightScan or "
            "Unitree-Go2-Velocity-ProxyStudent-HeightScan"
        )
    if args_cli.num_envs < 70 or args_cli.num_envs % 70:
        parser.error("multiterrain_stratified requires --num_envs to be a positive multiple of 70")
    if args_cli.seed < 0:
        parser.error("multiterrain_stratified requires a non-negative --seed")
if args_cli.terrain_protocol == "v5_foundation_stratified":
    if args_cli.task != "Unitree-Go2-Velocity-V5-Foundation":
        parser.error(
            "--terrain_protocol v5_foundation_stratified requires "
            "--task Unitree-Go2-Velocity-V5-Foundation"
        )
    if args_cli.num_envs < 32 or args_cli.num_envs % 32:
        parser.error("v5_foundation_stratified requires --num_envs to be a positive multiple of 32")
    if args_cli.seed < 0:
        parser.error("v5_foundation_stratified requires a non-negative --seed")
if args_cli.terrain_protocol == "v5_stairs_stratified":
    if args_cli.task != "Unitree-Go2-Velocity-V5-Stairs-Specialist":
        parser.error(
            "--terrain_protocol v5_stairs_stratified requires "
            "--task Unitree-Go2-Velocity-V5-Stairs-Specialist"
        )
    if args_cli.num_envs < 32 or args_cli.num_envs % 32:
        parser.error("v5_stairs_stratified requires --num_envs to be a positive multiple of 32")
    if args_cli.seed < 0:
        parser.error("v5_stairs_stratified requires a non-negative --seed")
if args_cli.terrain_protocol == "v5_boxes_stratified":
    if args_cli.task not in {
        "Unitree-Go2-Velocity-V5-Boxes-Specialist",
        "Unitree-Go2-Velocity-V5-Boxes-Mixed-Command-Remediation",
        "Unitree-Go2-Velocity-V5-Boxes-Safety-Remediation",
    }:
        parser.error(
            "--terrain_protocol v5_boxes_stratified requires "
            "--task Unitree-Go2-Velocity-V5-Boxes-Specialist or "
            "Unitree-Go2-Velocity-V5-Boxes-Mixed-Command-Remediation or "
            "Unitree-Go2-Velocity-V5-Boxes-Safety-Remediation"
        )
    if args_cli.num_envs < 8 or args_cli.num_envs % 8:
        parser.error("v5_boxes_stratified requires --num_envs to be a positive multiple of 8")
    if args_cli.seed < 0:
        parser.error("v5_boxes_stratified requires a non-negative --seed")
if args_cli.terrain_protocol == "teacher_v1_stratified":
    teacher_v1_tasks = {
        "Unitree-Go2-Velocity-TeacherV1-Flat",
        "Unitree-Go2-Velocity-TeacherV1-RandomRough",
        "Unitree-Go2-Velocity-TeacherV1-PyramidSlope",
        "Unitree-Go2-Velocity-TeacherV1-InvertedPyramidSlope",
        "Unitree-Go2-Velocity-TeacherV1-PyramidStairs",
        "Unitree-Go2-Velocity-TeacherV1-InvertedPyramidStairs",
        "Unitree-Go2-Velocity-TeacherV1-Boxes",
    }
    if args_cli.task not in teacher_v1_tasks:
        parser.error("--terrain_protocol teacher_v1_stratified requires a Terrain Teacher v1 task")
    if args_cli.num_envs < 10 or args_cli.num_envs % 10:
        parser.error("teacher_v1_stratified requires --num_envs to be a positive multiple of 10")
    if args_cli.seed < 0:
        parser.error("teacher_v1_stratified requires a non-negative --seed")
if args_cli.terrain_protocol == "teacher_v2_stratified":
    teacher_v2_tasks = {
        "Unitree-Go2-Velocity-TeacherV2-Flat",
        "Unitree-Go2-Velocity-TeacherV2-RandomRough",
        "Unitree-Go2-Velocity-TeacherV2-PyramidSlope",
        "Unitree-Go2-Velocity-TeacherV2-InvertedPyramidSlope",
        "Unitree-Go2-Velocity-TeacherV2-PyramidStairs",
        "Unitree-Go2-Velocity-TeacherV2-InvertedPyramidStairs",
        "Unitree-Go2-Velocity-TeacherV2-Boxes",
    }
    if args_cli.task not in teacher_v2_tasks:
        parser.error("--terrain_protocol teacher_v2_stratified requires a Terrain Teacher v2 task")
    if args_cli.num_envs < 10 or args_cli.num_envs % 10:
        parser.error("teacher_v2_stratified requires --num_envs to be a positive multiple of 10")
    if args_cli.seed < 0:
        parser.error("teacher_v2_stratified requires a non-negative --seed")
if args_cli.terrain_protocol == "teacher_v3_stratified":
    teacher_v3_tasks = {
        "Unitree-Go2-Velocity-TeacherV3-PyramidStairs",
        "Unitree-Go2-Velocity-TeacherV3-InvertedPyramidStairs",
        "Unitree-Go2-Velocity-TeacherV3-Boxes",
    }
    if args_cli.task not in teacher_v3_tasks:
        parser.error("--terrain_protocol teacher_v3_stratified requires a Terrain Teacher v3 task")
    if args_cli.num_envs < 10 or args_cli.num_envs % 10:
        parser.error("teacher_v3_stratified requires --num_envs to be a positive multiple of 10")
    if args_cli.seed < 0:
        parser.error("teacher_v3_stratified requires a non-negative --seed")

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
    "mean_abs_heading_error_rad",
    "rmse_heading_error_rad",
    "final_abs_heading_error_rad",
    "heading_within_5deg_fraction",
    "heading_within_10deg_fraction",
    "heading_within_15deg_fraction",
    "mean_target_heading_progress_mps",
    "target_heading_distance_m",
    "mean_command_aligned_progress_mps",
    "command_aligned_distance_m",
    "integrated_planar_displacement_m",
    "planar_path_length_m",
    "stall_fraction",
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
        if args_cli.terrain_protocol in {
            "rough_stratified",
            "multiterrain_stratified",
            "v5_foundation_stratified",
            "v5_stairs_stratified",
            "v5_boxes_stratified",
            "teacher_v1_stratified",
            "teacher_v2_stratified",
            "teacher_v3_stratified",
        }:
            raise ValueError(f"{args_cli.terrain_protocol} requires a terrain generator")
        return

    if args_cli.terrain_protocol == "legacy":
        generator_cfg.curriculum = False
        return

    sub_terrains = generator_cfg.sub_terrains
    if args_cli.terrain_protocol == "rough_stratified":
        expected_names = ["flat", "random_rough"]
        expected_proportions = [0.5, 0.5]
    elif args_cli.terrain_protocol == "multiterrain_stratified":
        expected_names = [
            "flat",
            "random_rough",
            "pyramid_slope",
            "inverted_pyramid_slope",
            "pyramid_stairs",
            "inverted_pyramid_stairs",
            "boxes",
        ]
        expected_proportions = [0.10, 0.20, 0.10, 0.10, 0.15, 0.15, 0.20]
    elif args_cli.terrain_protocol == "v5_foundation_stratified":
        expected_names = [
            "flat",
            "random_rough",
            "pyramid_slope",
            "inverted_pyramid_slope",
        ]
        expected_proportions = [0.25, 0.25, 0.25, 0.25]
    elif args_cli.terrain_protocol == "v5_stairs_stratified":
        expected_names = [
            "flat",
            "random_rough",
            "pyramid_stairs",
            "inverted_pyramid_stairs",
        ]
        expected_proportions = [0.15, 0.15, 0.35, 0.35]
    elif args_cli.terrain_protocol == "v5_boxes_stratified":
        expected_names = ["flat", "random_rough", "boxes"]
        expected_proportions = (
            [0.05, 0.05, 0.90]
            if args_cli.task == "Unitree-Go2-Velocity-V5-Boxes-Safety-Remediation"
            else [0.15, 0.15, 0.70]
        )
    elif args_cli.terrain_protocol in {
        "teacher_v1_stratified",
        "teacher_v2_stratified",
        "teacher_v3_stratified",
    }:
        if len(sub_terrains) != 1:
            raise ValueError(
                f"{args_cli.terrain_protocol} requires exactly one terrain type; "
                f"got {list(sub_terrains)}"
            )
        expected_names = list(sub_terrains)
        expected_proportions = [1.0]
    if list(sub_terrains) != expected_names:
        raise ValueError(
            f"{args_cli.terrain_protocol} requires ordered sub-terrains {expected_names}; "
            f"got {list(sub_terrains)}"
        )
    proportions = [float(cfg.proportion) for cfg in sub_terrains.values()]
    proportion_sum = sum(proportions)
    normalized = [value / proportion_sum for value in proportions]
    if any(abs(actual - expected) > 1.0e-9 for actual, expected in zip(normalized, expected_proportions)):
        raise ValueError(
            f"{args_cli.terrain_protocol} requires terrain proportions {expected_proportions}; "
            f"got {normalized}"
        )

    generator_cfg.seed = args_cli.seed
    generator_cfg.curriculum = True
    if args_cli.terrain_protocol in {
        "v5_foundation_stratified",
        "v5_stairs_stratified",
        "v5_boxes_stratified",
    }:
        generator_cfg.num_rows = 8
        generator_cfg.num_cols = 16 if args_cli.terrain_protocol == "v5_foundation_stratified" else 20
    else:
        generator_cfg.num_rows = 10
        generator_cfg.num_cols = 20
    terrain_cfg.max_init_terrain_level = generator_cfg.num_rows - 1


def configure_fixed_command(env_cfg) -> dict | None:
    """Apply one constant body-frame velocity command for every evaluation episode."""

    if args_cli.fixed_command_x is None:
        return None
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.rel_standing_envs = 0.0
    # Keep one immutable command/heading target for the entire fixed-command
    # episode.  A 20 s resampling period coincides with the default 20 s
    # episode boundary and can expose the next target in final-step metrics.
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.ranges.lin_vel_x = (args_cli.fixed_command_x, args_cli.fixed_command_x)
    command_cfg.ranges.lin_vel_y = (args_cli.fixed_command_y, args_cli.fixed_command_y)
    metadata = {
        "lin_vel_x_mps": args_cli.fixed_command_x,
        "lin_vel_y_mps": args_cli.fixed_command_y,
        "resampling_time_seconds": 1.0e9,
    }
    if args_cli.fixed_relative_heading is not None:
        if not hasattr(command_cfg, "relative_heading_range"):
            raise ValueError("--fixed_relative_heading requires a relative-heading command task")
        command_cfg.heading_command = True
        command_cfg.rel_heading_envs = 1.0
        command_cfg.relative_heading_range = (
            args_cli.fixed_relative_heading,
            args_cli.fixed_relative_heading,
        )
        metadata.update(
            {
                "mode": "fixed_relative_heading",
                "relative_heading_target_rad": args_cli.fixed_relative_heading,
                "ang_vel_z_limit_radps": list(command_cfg.ranges.ang_vel_z),
            }
        )
    elif args_cli.fixed_heading is None:
        command_cfg.ranges.ang_vel_z = (args_cli.fixed_command_yaw, args_cli.fixed_command_yaw)
        command_cfg.ranges.heading = None
        command_cfg.heading_command = False
        metadata.update({"mode": "fixed_yaw_rate", "ang_vel_z_radps": args_cli.fixed_command_yaw})
    else:
        command_cfg.heading_command = True
        command_cfg.rel_heading_envs = 1.0
        command_cfg.heading_control_stiffness = 1.0
        command_cfg.ranges.heading = (args_cli.fixed_heading, args_cli.fixed_heading)
        command_cfg.ranges.ang_vel_z = (-1.0, 1.0)
        metadata.update(
            {
                "mode": "fixed_world_heading_feedback",
                "heading_target_rad": args_cli.fixed_heading,
                "heading_control_stiffness": command_cfg.heading_control_stiffness,
                "ang_vel_z_limit_radps": list(command_cfg.ranges.ang_vel_z),
            }
        )
    if args_cli.fixed_initial_yaw is not None:
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (
            args_cli.fixed_initial_yaw,
            args_cli.fixed_initial_yaw,
        )
        metadata["initial_yaw_rad"] = args_cli.fixed_initial_yaw
    return metadata


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


def apply_terrain_stratification(raw_env, num_envs: int) -> bool:
    """Assign deterministic terrain types and levels for a stratified protocol."""

    if args_cli.terrain_protocol not in {
        "rough_stratified",
        "multiterrain_stratified",
        "v5_foundation_stratified",
        "v5_stairs_stratified",
        "v5_boxes_stratified",
        "teacher_v1_stratified",
        "teacher_v2_stratified",
        "teacher_v3_stratified",
    }:
        return False

    terrain = raw_env.scene.terrain
    generator_cfg = raw_env.cfg.scene.terrain.terrain_generator
    num_rows = int(generator_cfg.num_rows)
    num_cols = int(generator_cfg.num_cols)
    device = terrain.terrain_levels.device
    column_names = terrain_column_names(generator_cfg)

    if args_cli.terrain_protocol == "rough_stratified":
        columns_per_type = num_cols // 2
        envs_per_type = num_envs // 2
        local_ids = torch.arange(envs_per_type, device=device, dtype=torch.long)
        local_columns = torch.div(local_ids * columns_per_type, envs_per_type, rounding_mode="floor")
        paired_levels = torch.remainder(local_ids + args_cli.seed, num_rows)
        terrain.terrain_types[:] = torch.cat((local_columns, local_columns + columns_per_type))
        terrain.terrain_levels[:] = torch.cat((paired_levels, paired_levels))
    elif args_cli.terrain_protocol == "v5_boxes_stratified":
        matching_columns = [index for index, name in enumerate(column_names) if name == "boxes"]
        if not matching_columns:
            raise RuntimeError("no generated columns found for boxes terrain")
        envs_per_level = num_envs // num_rows
        local_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        terrain.terrain_types[:] = torch.tensor(
            [matching_columns[(index + args_cli.seed) % len(matching_columns)] for index in range(num_envs)],
            device=device,
            dtype=torch.long,
        )
        terrain.terrain_levels[:] = torch.remainder(
            torch.div(local_ids, envs_per_level, rounding_mode="floor") + args_cli.seed,
            num_rows,
        )
    else:
        terrain_names = list(generator_cfg.sub_terrains)
        envs_per_type = num_envs // len(terrain_names)
        assigned_columns = []
        assigned_levels = []
        for terrain_name in terrain_names:
            matching_columns = [index for index, name in enumerate(column_names) if name == terrain_name]
            if not matching_columns:
                raise RuntimeError(f"no generated columns found for terrain type {terrain_name}")
            for local_id in range(envs_per_type):
                assigned_columns.append(matching_columns[local_id % len(matching_columns)])
                assigned_levels.append((local_id + args_cli.seed) % num_rows)
        terrain.terrain_types[:] = torch.tensor(assigned_columns, device=device, dtype=torch.long)
        terrain.terrain_levels[:] = torch.tensor(assigned_levels, device=device, dtype=torch.long)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return True


def build_terrain_assignment(env_cfg, raw_env, num_envs: int) -> tuple[list[dict], dict]:
    """Capture each environment's fixed terrain assignment and its protocol metadata."""

    generator_cfg = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if args_cli.terrain_protocol not in {
        "rough_stratified",
        "multiterrain_stratified",
        "v5_foundation_stratified",
        "v5_stairs_stratified",
        "v5_boxes_stratified",
        "teacher_v1_stratified",
        "teacher_v2_stratified",
        "teacher_v3_stratified",
    }:
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
        raise RuntimeError(f"{args_cli.terrain_protocol} terrain assignments are unavailable")

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
        for attribute in (
            "noise_range",
            "noise_step",
            "slope_range",
            "step_height_range",
            "step_width",
            "grid_height_range",
            "grid_width",
            "platform_width",
            "border_width",
            "holes",
        ):
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
            if args_cli.terrain_protocol == "rough_stratified"
            else (
                "Equal environment blocks for all four ordered v5 terrain types. Within every type, "
                "environments cycle through that type's generated columns and cover levels 0-7; "
                "the evaluation scenario seed rotates the level assignment."
                if args_cli.terrain_protocol in {"v5_foundation_stratified", "v5_stairs_stratified"}
                else (
                    "All environments use boxes columns. Environments are divided equally across "
                    "levels 0-7; the evaluation scenario seed rotates both level order and boxes "
                    "column selection. Commands retain the task's seeded random distribution."
                    if args_cli.terrain_protocol == "v5_boxes_stratified"
                    else (
                        "All environments use the task's sole terrain type and cover "
                        "levels 0-9; the evaluation scenario seed rotates level and generated-column "
                        "assignment. Commands retain the task distribution unless fixed-command options "
                        "are provided."
                        if args_cli.terrain_protocol
                        in {"teacher_v1_stratified", "teacher_v2_stratified", "teacher_v3_stratified"}
                        else "Equal environment blocks for all seven ordered terrain types. Within every "
                        "type, environments cycle through that type's generated columns and cover levels "
                        "0-9; the evaluation scenario seed rotates the level assignment."
                    )
                )
            )
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
        "mean_abs_heading_error_rad": average("mean_abs_heading_error_rad"),
        "rmse_heading_error_rad": average("rmse_heading_error_rad"),
        "mean_final_abs_heading_error_rad": average("final_abs_heading_error_rad"),
        "mean_heading_within_5deg_fraction": average("heading_within_5deg_fraction"),
        "mean_heading_within_10deg_fraction": average("heading_within_10deg_fraction"),
        "mean_heading_within_15deg_fraction": average("heading_within_15deg_fraction"),
        "mean_target_heading_progress_mps": average("mean_target_heading_progress_mps"),
        "mean_target_heading_distance_m": average("target_heading_distance_m"),
        "mean_command_aligned_progress_mps": average("mean_command_aligned_progress_mps"),
        "mean_command_aligned_distance_m": average("command_aligned_distance_m"),
        "mean_integrated_planar_displacement_m": average("integrated_planar_displacement_m"),
        "mean_planar_path_length_m": average("planar_path_length_m"),
        "mean_stall_fraction": average("stall_fraction"),
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
    teacher_entry = None
    if args_cli.teacher_registry:
        from unitree_rl_lab.utils.teacher_bank import load_registry, load_teacher, IsaacObservationBridge
        project_root = Path(__file__).resolve().parents[2]
        registry = load_registry(args_cli.teacher_registry, project_root)
        teacher_entry = registry["teachers"][args_cli.teacher_id]
        if (project_root / teacher_entry["checkpoint"]).resolve() != checkpoint:
            raise ValueError("checkpoint differs from frozen teacher selection")
        if teacher_entry["task"] != args_cli.task or teacher_entry["terrain_protocol"] != args_cli.terrain_protocol:
            raise ValueError("task/protocol differs from frozen teacher selection")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed
    if teacher_entry is not None:
        # Parity acceptance needs identical raw states, not two independent
        # draws of actor observation noise. Legacy evaluation is unchanged.
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.observations.critic.enable_corruption = False
    configure_terrain_protocol(env_cfg)
    fixed_command = configure_fixed_command(env_cfg)
    # Evaluations are always headless.  Command arrows are unnecessary and
    # otherwise make Isaac Sim probe a remote USD asset during every process
    # launch, which can stall an offline matrix run for up to 300 seconds.
    env_cfg.commands.base_velocity.debug_vis = False

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if apply_terrain_stratification(base_env.unwrapped, args_cli.num_envs):
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
    print(f"[EVAL] fixed_command: {fixed_command}")
    print(f"[EVAL] max_episode_length: {env.max_episode_length}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Evaluation never updates parameters; loading optimizer state is unnecessary
    # and would reject valid distillation checkpoints produced by a non-PPO optimizer.
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    adapter_report = None
    if teacher_entry is not None:
        adapter_policy = load_teacher(teacher_entry, project_root, env.unwrapped.device)
        bridge = IsaacObservationBridge(env.unwrapped, teacher_entry)
        adapter_report = {"status": "running", "steps_compared": 0,
                          "max_observation_abs_error": 0.0, "max_action_abs_error": 0.0}

    observations = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        observations, _ = env.get_observations()

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    terminations = raw_env.termination_manager
    command_manager = raw_env.command_manager
    command_term = command_manager.get_term("base_velocity")
    if not hasattr(command_term, "heading_target"):
        raise RuntimeError(
            "evaluation requires base_velocity to expose heading_target; "
            "use a heading-command task"
        )
    device = raw_env.device
    num_envs = env.num_envs
    step_dt = float(raw_env.step_dt)
    terrain_assignments, terrain_metadata = build_terrain_assignment(env_cfg, raw_env, num_envs)

    if args_cli.terrain_protocol in {
        "rough_stratified",
        "multiterrain_stratified",
        "v5_foundation_stratified",
        "v5_stairs_stratified",
        "v5_boxes_stratified",
        "teacher_v1_stratified",
        "teacher_v2_stratified",
        "teacher_v3_stratified",
    }:
        distribution = terrain_metadata["initial_env_distribution"]
        type_counts = distribution["by_type"]
        if args_cli.terrain_protocol == "v5_boxes_stratified":
            expected_counts = {"boxes": num_envs}
        else:
            expected_names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains)
            expected_per_type = num_envs // len(expected_names)
            expected_counts = {name: expected_per_type for name in expected_names}
        if type_counts != expected_counts:
            raise RuntimeError(
                f"{args_cli.terrain_protocol} did not produce the required equal assignment: "
                f"{type_counts}"
            )
        num_rows = int(env_cfg.scene.terrain.terrain_generator.num_rows)
        expected_levels = {str(level) for level in range(num_rows)}
        for terrain_type, level_counts in distribution["by_type_and_level"].items():
            if set(level_counts) != expected_levels:
                raise RuntimeError(
                    f"{args_cli.terrain_protocol} did not cover levels 0-{num_rows - 1} "
                    f"for {terrain_type}: "
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
    heading_error_sum = tensor_zeros(num_envs, device)
    heading_error_sq_sum = tensor_zeros(num_envs, device)
    heading_within_5deg_count = tensor_zeros(num_envs, device)
    heading_within_10deg_count = tensor_zeros(num_envs, device)
    heading_within_15deg_count = tensor_zeros(num_envs, device)
    target_heading_progress_sum = tensor_zeros(num_envs, device)
    command_progress_sum = tensor_zeros(num_envs, device)
    world_displacement_sum = torch.zeros((num_envs, 2), dtype=torch.float64, device=device)
    planar_path_length_sum = tensor_zeros(num_envs, device)
    stall_count = tensor_zeros(num_envs, device)
    action_delta_sum = tensor_zeros(num_envs, device)
    action_delta_count = tensor_zeros(num_envs, device)
    power_sum = tensor_zeros(num_envs, device)
    previous_actions: torch.Tensor | None = None
    rows: list[dict] = []
    replay = defaultdict(list)

    output_dir.mkdir(parents=True)
    metadata = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": args_cli.seed,
        "evaluation_scenario_seed": args_cli.seed,
        "num_envs": num_envs,
        "episodes_per_env": args_cli.episodes_per_env,
        "max_episode_length": env.max_episode_length,
        "step_dt_seconds": step_dt,
        "curriculum_updates_enabled": False,
        "terrain_protocol": args_cli.terrain_protocol,
        "fixed_command": fixed_command,
        "terrain": terrain_metadata,
        "rsl_rl_version": version("rsl-rl-lib"),
    }
    if teacher_entry is not None:
        metadata["teacher_bank"] = {
            "teacher_id": args_cli.teacher_id,
            "registry_sha256": sha256_file(Path(args_cli.teacher_registry)),
            "training_seed": teacher_entry["training_seed"],
            "config_artifacts": teacher_entry["config_artifacts"],
            "selection_reason": teacher_entry["selection_reason"],
            "observation_corruption_enabled": False,
            "adapter_validation": adapter_report,
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
            signed_heading_error = torch.atan2(
                torch.sin(command_term.heading_target - robot.data.heading_w),
                torch.cos(command_term.heading_target - robot.data.heading_w),
            )
            abs_heading_error = torch.abs(signed_heading_error)
            command_direction = commands[:, :2] / torch.clamp(command_xy.unsqueeze(1), min=1.0e-6)
            command_progress = torch.sum(
                robot.data.root_lin_vel_b[:, :2] * command_direction, dim=1
            )
            moving_command = command_xy >= 0.2
            stalled = moving_command & (command_progress < 0.05)
            world_velocity_xy = robot.data.root_lin_vel_w[:, :2]
            target_heading_direction = torch.stack(
                (torch.cos(command_term.heading_target), torch.sin(command_term.heading_target)), dim=1
            )
            target_heading_progress = torch.sum(world_velocity_xy * target_heading_direction, dim=1)
            mechanical_power = torch.sum(
                torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1
            )

            actions = policy(observations)
            if teacher_entry is not None:
                adapted = bridge.build()
                for group in ("policy", "critic"):
                    torch.testing.assert_close(adapted[group], observations[group], atol=1e-6, rtol=1e-6)
                    error = float((adapted[group] - observations[group]).abs().max())
                    adapter_report["max_observation_abs_error"] = max(adapter_report["max_observation_abs_error"], error)
                adapted_actions = adapter_policy.act_inference(adapted)
                torch.testing.assert_close(adapted_actions, actions, atol=1e-5, rtol=1e-5)
                error = float((adapted_actions - actions).abs().max())
                adapter_report["max_action_abs_error"] = max(adapter_report["max_action_abs_error"], error)
                adapter_report["steps_compared"] += 1
                # The adapted teacher actually drives the environment; native
                # inference is the independently stateful parity reference.
                actions = adapted_actions
            if args_cli.capture_policy_sequence:
                capture = {
                    "observations": observations["policy"], "actions": actions,
                    "root_pos_w": robot.data.root_pos_w, "root_quat_w": robot.data.root_quat_w,
                    "root_lin_vel_b": robot.data.root_lin_vel_b, "root_ang_vel_b": robot.data.root_ang_vel_b,
                    "joint_pos": robot.data.joint_pos, "joint_vel": robot.data.joint_vel,
                    "commands": commands, "last_action": raw_env.action_manager.action,
                    "episode_steps": raw_env.episode_length_buf,
                }
                for key, value in capture.items():
                    replay[key].append(value[:2].detach().cpu().numpy().copy())
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
            heading_error_sum[active] += abs_heading_error[active].double()
            heading_error_sq_sum[active] += torch.square(abs_heading_error[active]).double()
            heading_within_5deg_count[active] += (
                abs_heading_error[active] <= math.radians(5.0)
            ).double()
            heading_within_10deg_count[active] += (
                abs_heading_error[active] <= math.radians(10.0)
            ).double()
            heading_within_15deg_count[active] += (
                abs_heading_error[active] <= math.radians(15.0)
            ).double()
            target_heading_progress_sum[active] += target_heading_progress[active].double()
            command_progress_sum[active] += command_progress[active].double()
            world_displacement_sum[active] += world_velocity_xy[active].double() * step_dt
            planar_path_length_sum[active] += (
                torch.linalg.vector_norm(world_velocity_xy[active], dim=1).double() * step_dt
            )
            stall_count[active] += stalled[active].double()
            power_sum[active] += mechanical_power[active].double()
            episode_steps[active] += 1

            observations, rewards, dones, _ = env.step(actions)
            if args_cli.capture_policy_sequence:
                replay["dones"].append(dones[:2].detach().cpu().numpy().copy())
            # RSL-RL inference keeps recurrent hidden state internally.  Clear
            # only completed environments so a new episode never inherits the
            # previous episode's GRU/LSTM state.  This is a no-op for MLP policies.
            if getattr(runner.alg.policy, "is_recurrent", False):
                runner.alg.policy.reset(dones)
            if teacher_entry is not None:
                adapter_policy.reset(dones)
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
                if args_cli.terrain_protocol in {
                    "rough_stratified",
                    "multiterrain_stratified",
                    "v5_foundation_stratified",
                    "v5_stairs_stratified",
                    "v5_boxes_stratified",
                    "teacher_v1_stratified",
                    "teacher_v2_stratified",
                    "teacher_v3_stratified",
                }:
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
                        "mean_abs_heading_error_rad": float(heading_error_sum[env_id].item() / steps),
                        "rmse_heading_error_rad": float(
                            (heading_error_sq_sum[env_id].item() / steps) ** 0.5
                        ),
                        "final_abs_heading_error_rad": float(abs_heading_error[env_id].item()),
                        "heading_within_5deg_fraction": float(
                            heading_within_5deg_count[env_id].item() / steps
                        ),
                        "heading_within_10deg_fraction": float(
                            heading_within_10deg_count[env_id].item() / steps
                        ),
                        "heading_within_15deg_fraction": float(
                            heading_within_15deg_count[env_id].item() / steps
                        ),
                        "mean_target_heading_progress_mps": float(
                            target_heading_progress_sum[env_id].item() / steps
                        ),
                        "target_heading_distance_m": float(
                            target_heading_progress_sum[env_id].item() * step_dt
                        ),
                        "mean_command_aligned_progress_mps": float(
                            command_progress_sum[env_id].item() / steps
                        ),
                        "command_aligned_distance_m": float(
                            command_progress_sum[env_id].item() * step_dt
                        ),
                        "integrated_planar_displacement_m": float(
                            torch.linalg.vector_norm(world_displacement_sum[env_id]).item()
                        ),
                        "planar_path_length_m": float(planar_path_length_sum[env_id].item()),
                        "stall_fraction": float(stall_count[env_id].item() / steps),
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
            heading_error_sum[done_ids] = 0
            heading_error_sq_sum[done_ids] = 0
            heading_within_5deg_count[done_ids] = 0
            heading_within_10deg_count[done_ids] = 0
            heading_within_15deg_count[done_ids] = 0
            target_heading_progress_sum[done_ids] = 0
            command_progress_sum[done_ids] = 0
            world_displacement_sum[done_ids] = 0
            planar_path_length_sum[done_ids] = 0
            stall_count[done_ids] = 0
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

    if adapter_report is not None:
        adapter_report["status"] = "pass"
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args_cli.capture_policy_sequence:
        import numpy as np
        np.savez_compressed(output_dir / "policy_sequence.npz", **{k: np.stack(v) for k, v in replay.items()},
                            joint_names=np.asarray(robot.data.joint_names),
                            checkpoint_sha256=np.asarray(metadata["checkpoint_sha256"]),
                            control_dt=np.asarray(step_dt))
    summary = summarize(rows, vector_steps, elapsed, metadata)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("[EVAL] completed successfully")
    print(json.dumps(summary, indent=2))
    print(f"[EVAL] episodes_csv: {output_dir / 'episodes.csv'}")
    print(f"[EVAL] summary_json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        # Print before Kit shutdown, which may otherwise swallow the traceback.
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
