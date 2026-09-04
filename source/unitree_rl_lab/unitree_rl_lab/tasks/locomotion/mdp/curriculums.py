from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def boxes_command_ranges_tracking_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracking_reward_term: str = "track_lin_vel_xy",
    progress_reward_term: str = "command_progress",
    tracking_fraction_threshold: float = 0.65,
    minimum_progress_speed: float = 0.20,
    reverse_step: float = 0.05,
    lateral_step: float = 0.025,
) -> torch.Tensor:
    """Gradually add reverse and lateral boxes commands after competent episodes.

    The source specialist is reliable only for forward commands. Expansion is
    therefore gated by both velocity tracking and command-aligned progress,
    and changes only at full-episode boundaries. The returned scalar is the
    currently exposed reverse speed magnitude for TensorBoard auditing.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_term)
    tracking_average = (
        torch.mean(env.reward_manager._episode_sums[tracking_reward_term][env_ids])
        / env.max_episode_length_s
    )
    progress_cfg = env.reward_manager.get_term_cfg(progress_reward_term)
    progress_average = (
        torch.mean(env.reward_manager._episode_sums[progress_reward_term][env_ids])
        / env.max_episode_length_s
    )
    tracks_well = tracking_average >= tracking_cfg.weight * tracking_fraction_threshold
    makes_progress = progress_average >= progress_cfg.weight * minimum_progress_speed

    if env.common_step_counter % env.max_episode_length == 0 and tracks_well and makes_progress:
        ranges.lin_vel_x = [
            max(float(limit_ranges.lin_vel_x[0]), float(ranges.lin_vel_x[0]) - reverse_step),
            float(ranges.lin_vel_x[1]),
        ]
        ranges.lin_vel_y = [
            max(float(limit_ranges.lin_vel_y[0]), float(ranges.lin_vel_y[0]) - lateral_step),
            min(float(limit_ranges.lin_vel_y[1]), float(ranges.lin_vel_y[1]) + lateral_step),
        ]

    return torch.tensor(max(0.0, -float(ranges.lin_vel_x[0])), device=env.device)


def multiterrain_levels_survival(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """Adjust complex-terrain difficulty from episode survival.

    The upstream distance curriculum requires a four-metre net displacement
    on the current eight-metre tiles. That criterion is poorly matched to
    velocity commands that may be resampled or reverse direction within an
    episode. MultiTerrain instead advances environments that reach the time
    limit, lowers environments that fail in the first half of an episode, and
    leaves intermediate outcomes at their current level.
    """
    terrain = env.scene.terrain
    episode_length = env.episode_length_buf[env_ids]
    move_up = episode_length >= env.max_episode_length - 1
    move_down = episode_length < env.max_episode_length * 0.5
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def multiterrain_levels_tracking_survival(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    tracking_threshold: float = 0.65,
) -> torch.Tensor:
    """Adjust terrain difficulty using both survival and velocity tracking.

    An environment advances only after surviving the full episode while its
    average linear-velocity tracking reward reaches ``tracking_threshold`` of
    the term's maximum weighted reward. Full episodes with weaker tracking and
    failures in the first half are moved down; intermediate failures stay at
    their current level.
    """
    terrain = env.scene.terrain
    episode_length = env.episode_length_buf[env_ids]
    reached_time_limit = episode_length >= env.max_episode_length - 1

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    tracking_reward = env.reward_manager._episode_sums[reward_term_name][env_ids] / env.max_episode_length_s
    tracks_well = tracking_reward >= reward_term.weight * tracking_threshold

    move_up = reached_time_limit & tracks_well
    move_down = (episode_length < env.max_episode_length * 0.5) | (reached_time_limit & ~tracks_well)
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def multiterrain_levels_progress_tracking_survival(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    tracking_threshold: float = 0.45,
    minimum_displacement_m: float = 0.75,
    standing_command_threshold: float = 0.1,
) -> torch.Tensor:
    """Advance only episodes that survive, track, and make non-trivial progress.

    The velocity command may be resampled during an episode, so net displacement
    alone cannot establish command alignment. The integrated tracking reward is
    used for alignment, while displacement rejects stationary survival. Standing
    commands are exempt from the displacement requirement.
    """
    terrain = env.scene.terrain
    asset = env.scene["robot"]
    episode_length = env.episode_length_buf[env_ids]
    reached_time_limit = episode_length >= env.max_episode_length - 1

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    tracking_reward = env.reward_manager._episode_sums[reward_term_name][env_ids] / env.max_episode_length_s
    tracks_well = tracking_reward >= reward_term.weight * tracking_threshold

    displacement = torch.linalg.vector_norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    command_speed = torch.linalg.vector_norm(
        env.command_manager.get_command("base_velocity")[env_ids, :2], dim=1
    )
    makes_progress = (displacement >= minimum_displacement_m) | (command_speed < standing_command_threshold)

    move_up = reached_time_limit & tracks_well & makes_progress
    early_failure = episode_length < env.max_episode_length * 0.5
    failed_success_gate = reached_time_limit & (~tracks_well | ~makes_progress)
    move_down = (early_failure | failed_success_gate) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_tracking_command_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracking_reward_term: str = "track_lin_vel_xy",
    progress_reward_term: str = "command_progress",
    tracking_fraction_threshold: float = 0.60,
    minimum_progress_speed: float = 0.12,
) -> torch.Tensor:
    """Advance terrain only after sustained tracking and command-aligned motion.

    Both metrics come from episode-integrated reward terms, so command direction
    changes cannot cancel as they do with final net displacement.  The v5
    foundation task keeps commands fixed for an entire episode, but retaining
    the integrated definition makes this curriculum safe for later stages.
    """
    terrain = env.scene.terrain
    episode_length = env.episode_length_buf[env_ids]
    reached_time_limit = episode_length >= env.max_episode_length - 1

    tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_term)
    tracking_average = env.reward_manager._episode_sums[tracking_reward_term][env_ids] / env.max_episode_length_s
    tracks_well = tracking_average >= tracking_cfg.weight * tracking_fraction_threshold

    progress_cfg = env.reward_manager.get_term_cfg(progress_reward_term)
    progress_average = env.reward_manager._episode_sums[progress_reward_term][env_ids] / env.max_episode_length_s
    makes_progress = progress_average >= progress_cfg.weight * minimum_progress_speed

    move_up = reached_time_limit & tracks_well & makes_progress
    early_failure = episode_length < env.max_episode_length * 0.5
    failed_full_episode = reached_time_limit & (~tracks_well | ~makes_progress)
    move_down = (early_failure | failed_full_episode) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_tracking_progress_yaw(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracking_reward_term: str = "track_lin_vel_xy",
    progress_reward_term: str = "command_progress",
    yaw_reward_term: str = "track_ang_vel_z",
    tracking_fraction_threshold: float = 0.55,
    minimum_progress_speed: float = 0.15,
    yaw_fraction_threshold: float = 0.70,
) -> torch.Tensor:
    """Gate terrain promotion on linear tracking, progress, and yaw tracking.

    The yaw gate prevents a policy from earning body-frame command progress by
    circling on a stair tile.  Full-length episodes that miss any gate move
    down; early failures retain the same behavior as the v5 curriculum.
    """
    terrain = env.scene.terrain
    episode_length = env.episode_length_buf[env_ids]
    reached_time_limit = episode_length >= env.max_episode_length - 1

    tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_term)
    tracking_average = env.reward_manager._episode_sums[tracking_reward_term][env_ids] / env.max_episode_length_s
    tracks_well = tracking_average >= tracking_cfg.weight * tracking_fraction_threshold

    progress_cfg = env.reward_manager.get_term_cfg(progress_reward_term)
    progress_average = env.reward_manager._episode_sums[progress_reward_term][env_ids] / env.max_episode_length_s
    makes_progress = progress_average >= progress_cfg.weight * minimum_progress_speed

    yaw_cfg = env.reward_manager.get_term_cfg(yaw_reward_term)
    yaw_average = env.reward_manager._episode_sums[yaw_reward_term][env_ids] / env.max_episode_length_s
    tracks_yaw = yaw_average >= yaw_cfg.weight * yaw_fraction_threshold

    move_up = reached_time_limit & tracks_well & makes_progress & tracks_yaw
    early_failure = episode_length < env.max_episode_length * 0.5
    failed_full_episode = reached_time_limit & (~tracks_well | ~makes_progress | ~tracks_yaw)
    move_down = (early_failure | failed_full_episode) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_tracking_world_progress_heading(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracking_reward_term: str = "track_lin_vel_xy",
    progress_reward_term: str = "command_progress",
    heading_error_term: str = "heading_error",
    tracking_fraction_threshold: float = 0.60,
    minimum_progress_speed: float = 0.12,
    maximum_mean_heading_error: float = 0.22,
) -> torch.Tensor:
    """Promote terrain only with tracking, world progress, and heading control."""
    terrain = env.scene.terrain
    episode_length = env.episode_length_buf[env_ids]
    reached_time_limit = episode_length >= env.max_episode_length - 1

    tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_term)
    tracking_average = env.reward_manager._episode_sums[tracking_reward_term][env_ids] / env.max_episode_length_s
    tracks_well = tracking_average >= tracking_cfg.weight * tracking_fraction_threshold

    progress_cfg = env.reward_manager.get_term_cfg(progress_reward_term)
    progress_average = env.reward_manager._episode_sums[progress_reward_term][env_ids] / env.max_episode_length_s
    makes_progress = progress_average >= progress_cfg.weight * minimum_progress_speed

    heading_cfg = env.reward_manager.get_term_cfg(heading_error_term)
    weighted_heading_average = (
        env.reward_manager._episode_sums[heading_error_term][env_ids] / env.max_episode_length_s
    )
    mean_heading_error = weighted_heading_average / heading_cfg.weight
    tracks_heading = mean_heading_error <= maximum_mean_heading_error

    move_up = reached_time_limit & tracks_well & makes_progress & tracks_heading
    early_failure = episode_length < env.max_episode_length * 0.5
    failed_full_episode = reached_time_limit & (~tracks_well | ~makes_progress | ~tracks_heading)
    move_down = (early_failure | failed_full_episode) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_tracking_world_progress_heading_gait(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracking_reward_term: str = "track_lin_vel_xy",
    progress_reward_term: str = "command_progress",
    heading_error_term: str = "heading_error",
    long_contact_term: str = "excessive_contact_time",
    all_contact_term: str = "all_feet_contact",
    tracking_fraction_threshold: float = 0.60,
    minimum_progress_speed: float = 0.10,
    maximum_mean_heading_error: float = 0.22,
    maximum_mean_long_contact_penalty: float = 0.08,
    maximum_all_feet_contact_fraction: float = 0.45,
) -> torch.Tensor:
    """Promote stair difficulty only after tracking, progress, heading, and gait gates pass."""
    terrain = env.scene.terrain
    episode_length = env.episode_length_buf[env_ids]
    reached_time_limit = episode_length >= env.max_episode_length - 1

    def episode_average(term_name: str) -> torch.Tensor:
        return env.reward_manager._episode_sums[term_name][env_ids] / env.max_episode_length_s

    tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_term)
    tracks_well = episode_average(tracking_reward_term) >= tracking_cfg.weight * tracking_fraction_threshold

    progress_cfg = env.reward_manager.get_term_cfg(progress_reward_term)
    makes_progress = episode_average(progress_reward_term) >= progress_cfg.weight * minimum_progress_speed

    heading_cfg = env.reward_manager.get_term_cfg(heading_error_term)
    mean_heading_error = episode_average(heading_error_term) / heading_cfg.weight
    tracks_heading = mean_heading_error <= maximum_mean_heading_error

    long_contact_cfg = env.reward_manager.get_term_cfg(long_contact_term)
    mean_long_contact = episode_average(long_contact_term) / long_contact_cfg.weight
    avoids_long_contact = mean_long_contact <= maximum_mean_long_contact_penalty

    all_contact_cfg = env.reward_manager.get_term_cfg(all_contact_term)
    all_contact_fraction = episode_average(all_contact_term) / all_contact_cfg.weight
    avoids_support_freeze = all_contact_fraction <= maximum_all_feet_contact_fraction

    gait_ok = avoids_long_contact & avoids_support_freeze
    move_up = reached_time_limit & tracks_well & makes_progress & tracks_heading & gait_ok
    early_failure = episode_length < env.max_episode_length * 0.5
    failed_full_episode = reached_time_limit & (~tracks_well | ~makes_progress | ~tracks_heading | ~gait_ok)
    move_down = (early_failure | failed_full_episode) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_tracking_world_progress_heading_gait_v2(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracking_reward_term: str = "track_lin_vel_xy",
    progress_reward_term: str = "command_progress",
    heading_error_term: str = "heading_error",
    long_contact_term: str = "excessive_contact_time",
    long_air_term: str = "excessive_foot_air_time",
    participation_term: str = "contact_participation",
    phase_term: str = "trot_phase",
    tracking_fraction_threshold: float = 0.60,
    minimum_progress_speed: float = 0.10,
    maximum_mean_heading_error: float = 0.22,
    maximum_mean_long_contact_penalty: float = 0.06,
    maximum_mean_long_air_penalty: float = 0.05,
    maximum_mean_participation_penalty: float = 0.02,
    minimum_phase_match: float = 0.35,
) -> torch.Tensor:
    """Promote stairs only when both long-contact and long-air gait failures are absent."""
    terrain = env.scene.terrain
    episode_length = env.episode_length_buf[env_ids]
    reached_time_limit = episode_length >= env.max_episode_length - 1

    def raw_average(term_name: str) -> torch.Tensor:
        cfg = env.reward_manager.get_term_cfg(term_name)
        weighted = env.reward_manager._episode_sums[term_name][env_ids] / env.max_episode_length_s
        return weighted / cfg.weight

    tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_term)
    tracking_weighted = env.reward_manager._episode_sums[tracking_reward_term][env_ids] / env.max_episode_length_s
    tracks_well = tracking_weighted >= tracking_cfg.weight * tracking_fraction_threshold

    progress_cfg = env.reward_manager.get_term_cfg(progress_reward_term)
    progress_weighted = env.reward_manager._episode_sums[progress_reward_term][env_ids] / env.max_episode_length_s
    makes_progress = progress_weighted >= progress_cfg.weight * minimum_progress_speed

    tracks_heading = raw_average(heading_error_term) <= maximum_mean_heading_error
    avoids_long_contact = raw_average(long_contact_term) <= maximum_mean_long_contact_penalty
    avoids_long_air = raw_average(long_air_term) <= maximum_mean_long_air_penalty
    participates = raw_average(participation_term) <= maximum_mean_participation_penalty
    phase_matches = raw_average(phase_term) >= minimum_phase_match

    gait_ok = avoids_long_contact & avoids_long_air & participates & phase_matches
    move_up = reached_time_limit & tracks_well & makes_progress & tracks_heading & gait_ok
    early_failure = episode_length < env.max_episode_length * 0.5
    failed_full_episode = reached_time_limit & (~tracks_well | ~makes_progress | ~tracks_heading | ~gait_ok)
    move_down = (early_failure | failed_full_episode) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_tracking_world_progress_heading_gait_v3(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracking_reward_term: str = "track_lin_vel_xy",
    progress_reward_term: str = "command_progress",
    heading_error_term: str = "heading_error",
    lateral_drift_term: str = "heading_lateral_drift",
    long_contact_term: str = "excessive_contact_time",
    long_air_term: str = "excessive_foot_air_time",
    participation_term: str = "contact_participation",
    duty_symmetry_term: str | None = None,
    phase_term: str = "trot_phase",
    stall_term: str | None = None,
    elevation_progress_term: str | None = None,
    support_edge_term: str | None = None,
    tracking_fraction_threshold: float = 0.60,
    minimum_progress_speed: float = 0.10,
    maximum_mean_heading_error: float = 0.22,
    maximum_mean_lateral_drift: float = 0.025,
    maximum_mean_long_contact_penalty: float = 0.06,
    maximum_mean_long_air_penalty: float = 0.05,
    maximum_mean_participation_penalty: float = 0.02,
    maximum_mean_duty_symmetry_penalty: float = 0.03,
    minimum_phase_match: float = 0.30,
    maximum_mean_stall_fraction: float = 0.08,
    minimum_mean_elevation_progress: float = 0.005,
    maximum_mean_support_edge_penalty: float = 0.02,
) -> torch.Tensor:
    """Gait-safe promotion with optional ascent and box-support success gates."""
    terrain = env.scene.terrain
    episode_length = env.episode_length_buf[env_ids]
    reached_time_limit = episode_length >= env.max_episode_length - 1

    def raw_average(term_name: str) -> torch.Tensor:
        cfg = env.reward_manager.get_term_cfg(term_name)
        weighted = env.reward_manager._episode_sums[term_name][env_ids] / env.max_episode_length_s
        return weighted / cfg.weight

    tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_term)
    tracking_weighted = env.reward_manager._episode_sums[tracking_reward_term][env_ids] / env.max_episode_length_s
    tracks_well = tracking_weighted >= tracking_cfg.weight * tracking_fraction_threshold
    progress_cfg = env.reward_manager.get_term_cfg(progress_reward_term)
    progress_weighted = env.reward_manager._episode_sums[progress_reward_term][env_ids] / env.max_episode_length_s
    makes_progress = progress_weighted >= progress_cfg.weight * minimum_progress_speed

    task_ok = (
        (raw_average(heading_error_term) <= maximum_mean_heading_error)
        & (raw_average(lateral_drift_term) <= maximum_mean_lateral_drift)
        & (raw_average(long_contact_term) <= maximum_mean_long_contact_penalty)
        & (raw_average(long_air_term) <= maximum_mean_long_air_penalty)
        & (raw_average(participation_term) <= maximum_mean_participation_penalty)
        & (raw_average(phase_term) >= minimum_phase_match)
    )
    if elevation_progress_term is not None:
        task_ok &= raw_average(elevation_progress_term) >= minimum_mean_elevation_progress
    if support_edge_term is not None:
        task_ok &= raw_average(support_edge_term) <= maximum_mean_support_edge_penalty
    if duty_symmetry_term is not None:
        task_ok &= raw_average(duty_symmetry_term) <= maximum_mean_duty_symmetry_penalty
    if stall_term is not None:
        task_ok &= raw_average(stall_term) <= maximum_mean_stall_fraction

    move_up = reached_time_limit & tracks_well & makes_progress & task_ok
    early_failure = episode_length < env.max_episode_length * 0.5
    failed_full_episode = reached_time_limit & (~tracks_well | ~makes_progress | ~task_ok)
    move_down = (early_failure | failed_full_episode) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
