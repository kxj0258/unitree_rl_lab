from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def command_direction_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    standing_command_threshold: float = 0.1,
    max_progress_speed: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward planar velocity projected onto the commanded travel direction.

    Unlike a world-frame forward-velocity reward, this term remains correct for
    forward, backward, and lateral commands.  Standing commands receive zero so
    they cannot accumulate progress reward by drifting.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=1)
    command_direction = command_xy / torch.clamp(command_speed.unsqueeze(1), min=1.0e-6)
    projected_speed = torch.sum(asset.data.root_lin_vel_b[:, :2] * command_direction, dim=1)
    projected_speed = torch.clamp(projected_speed, min=0.0, max=max_progress_speed)
    return projected_speed * (command_speed >= standing_command_threshold)


def commanded_motion_stall(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_speed_threshold: float = 0.2,
    progress_speed_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one when a moving command produces almost no aligned motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=1)
    command_direction = command_xy / torch.clamp(command_speed.unsqueeze(1), min=1.0e-6)
    projected_speed = torch.sum(asset.data.root_lin_vel_b[:, :2] * command_direction, dim=1)
    return ((command_speed >= command_speed_threshold) & (projected_speed < progress_speed_threshold)).float()


def target_heading_world_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    standing_command_threshold: float = 0.1,
    max_progress_speed: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward world velocity projected toward the locked heading target.

    This closes the body-frame loophole that lets a robot circle while earning
    positive forward progress.  It requires a command term exposing
    ``heading_target``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "heading_target"):
        raise AttributeError(f"{command_name} does not expose heading_target")
    heading = command_term.heading_target
    target_direction_w = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    projected_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * target_direction_w, dim=1)
    command_vx = env.command_manager.get_command(command_name)[:, 0]
    return torch.clamp(projected_speed, min=0.0, max=max_progress_speed) * (
        command_vx >= standing_command_threshold
    )


def heading_error_abs(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    minimum_turning_speed: float = 0.1,
) -> torch.Tensor:
    """Absolute wrapped heading error, zeroed for standing commands."""
    command_term = env.command_manager.get_term(command_name)
    error = torch.abs(torch.atan2(
        torch.sin(command_term.heading_target - env.scene["robot"].data.heading_w),
        torch.cos(command_term.heading_target - env.scene["robot"].data.heading_w),
    ))
    moving = env.command_manager.get_command(command_name)[:, 0] >= minimum_turning_speed
    return error * moving


def target_heading_lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    minimum_turning_speed: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world velocity perpendicular to the locked heading target.

    Body-frame forward tracking alone can be satisfied while circling.  This
    term measures lateral drift in the immutable target-heading frame instead.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    heading = command_term.heading_target
    lateral_direction_w = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=1)
    lateral_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * lateral_direction_w, dim=1)
    moving = env.command_manager.get_command(command_name)[:, 0] >= minimum_turning_speed
    return torch.square(lateral_speed) * moving


def target_heading_aligned_vertical_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    minimum_forward_progress: float = 0.05,
    maximum_abs_vertical_speed: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed elevation gain only while progressing along target heading.

    Positive and negative vertical motion cancel over a jump cycle, preventing
    repeated hopping from impersonating sustained stair ascent.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    heading = command_term.heading_target
    target_direction_w = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    aligned_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * target_direction_w, dim=1)
    vertical_speed = torch.clamp(
        asset.data.root_lin_vel_w[:, 2], -maximum_abs_vertical_speed, maximum_abs_vertical_speed
    )
    return vertical_speed * (aligned_speed >= minimum_forward_progress)


def excessive_foot_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    maximum_air_time: float = 0.45,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize every foot that stays continuously airborne too long."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    excess = torch.clamp(
        sensor.data.current_air_time[:, sensor_cfg.body_ids] - maximum_air_time,
        min=0.0,
    )
    moving = env.command_manager.get_command(command_name)[:, 0] >= 0.1
    return torch.sum(excess, dim=1) * moving


def all_feet_airborne(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize flight phases for walking-speed forward locomotion."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    moving = env.command_manager.get_command(command_name)[:, 0] >= 0.1
    return (~torch.any(contacts, dim=1)).float() * moving


def all_feet_in_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    minimum_command_speed: float = 0.1,
) -> torch.Tensor:
    """Penalize a four-foot support freeze while a forward command is active."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    moving = env.command_manager.get_command(command_name)[:, 0] >= minimum_command_speed
    return torch.all(contacts, dim=1).float() * moving


def excessive_foot_contact_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    maximum_contact_time: float = 0.8,
    command_name: str = "base_velocity",
    minimum_command_speed: float = 0.1,
) -> torch.Tensor:
    """Penalize feet that remain planted continuously during commanded motion."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    excess = torch.clamp(
        sensor.data.current_contact_time[:, sensor_cfg.body_ids] - maximum_contact_time,
        min=0.0,
    )
    moving = env.command_manager.get_command(command_name)[:, 0] >= minimum_command_speed
    return torch.sum(excess, dim=1) * moving


def loaded_feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_normalization: float = 50.0,
    maximum_force_scale: float = 2.0,
    square_speed: bool = False,
) -> torch.Tensor:
    """Penalize horizontal foot motion in proportion to vertical contact load."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    vertical_force = torch.abs(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    load_scale = torch.clamp(vertical_force / force_normalization, min=0.0, max=maximum_force_scale)
    horizontal_speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    if square_speed:
        horizontal_speed = torch.square(horizontal_speed)
    return torch.sum(horizontal_speed * load_scale, dim=1)


def swing_foot_terrain_clearance(
    env: ManagerBasedRLEnv,
    height_sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_clearance: float = 0.08,
    horizontal_speed_threshold: float = 0.05,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize moving swing feet that do not clear the nearest scanned terrain height.

    The nearest ray hit in XY supplies a terrain-relative reference.  Contact
    feet are excluded so stance is governed by slip/contact-time terms instead
    of being incorrectly pulled upward by a body-frame height target.
    """
    height_sensor = env.scene.sensors[height_sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
    ray_hits = height_sensor.data.ray_hits_w
    xy_distance_sq = torch.sum(
        torch.square(foot_positions[:, :, None, :2] - ray_hits[:, None, :, :2]), dim=-1
    )
    nearest_indices = torch.argmin(xy_distance_sq, dim=-1)
    terrain_height = torch.gather(ray_hits[..., 2], 1, nearest_indices)
    clearance = foot_positions[..., 2] - terrain_height

    contacts = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0
    horizontal_speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    active_swing = (~contacts) & (horizontal_speed >= horizontal_speed_threshold)
    moving = env.command_manager.get_command(command_name)[:, 0] >= 0.1
    shortfall = torch.clamp(target_clearance - clearance, min=0.0)
    return torch.sum(shortfall * active_swing, dim=1) * moving


def speed_adaptive_trot_phase_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    maximum_period: float = 0.75,
    minimum_period: float = 0.55,
    minimum_speed: float = 0.15,
    maximum_speed: float = 0.50,
    stance_ratio: float = 0.62,
    maximum_terrain_level: int = 9,
    minimum_terrain_scale: float = 0.20,
) -> torch.Tensor:
    """Soft speed-adaptive trot contact reward with reduced weight on hard stairs."""
    command_speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    speed_fraction = torch.clamp(
        (command_speed - minimum_speed) / max(maximum_speed - minimum_speed, 1.0e-6), 0.0, 1.0
    )
    period = maximum_period + speed_fraction * (minimum_period - maximum_period)
    global_phase = torch.remainder(env.episode_length_buf * env.step_dt, period) / period
    offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=env.device)
    expected_stance = torch.remainder(global_phase.unsqueeze(1) + offsets, 1.0) < stance_ratio

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    contact_match = torch.mean((contacts == expected_stance).float(), dim=1)
    moving = command_speed >= 0.1

    terrain_level = env.scene.terrain.terrain_levels.float()
    difficulty_fraction = torch.clamp(terrain_level / max(maximum_terrain_level, 1), 0.0, 1.0)
    terrain_scale = 1.0 - (1.0 - minimum_terrain_scale) * difficulty_fraction
    return contact_match * terrain_scale * moving


class FootContactParticipationPenalty(ManagerTermBase):
    """Penalize per-foot contact EMA outside a deployable participation band."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.window_seconds = float(cfg.params["window_seconds"])
        self.contact_ema = torch.full((env.num_envs, 4), 0.5, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.contact_ema[env_ids] = 0.5

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        window_seconds: float,
        minimum_duty_factor: float,
        maximum_duty_factor: float,
        command_name: str,
    ) -> torch.Tensor:
        del window_seconds
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contacts = (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()
        alpha = min(1.0, env.step_dt / self.window_seconds)
        self.contact_ema.mul_(1.0 - alpha).add_(contacts * alpha)
        low = torch.clamp(minimum_duty_factor - self.contact_ema, min=0.0)
        high = torch.clamp(self.contact_ema - maximum_duty_factor, min=0.0)
        moving = env.command_manager.get_command(command_name)[:, 0] >= 0.1
        return torch.sum(torch.square(low) + torch.square(high), dim=1) * moving


class FootContactDutySymmetryPenalty(ManagerTermBase):
    """Penalize the EMA duty-factor spread so one leg cannot do much less work."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.window_seconds = float(cfg.params["window_seconds"])
        self.contact_ema = torch.full((env.num_envs, 4), 0.5, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.contact_ema[env_ids] = 0.5

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        window_seconds: float,
        allowed_duty_spread: float,
        command_name: str,
    ) -> torch.Tensor:
        del window_seconds
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contacts = (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()
        alpha = min(1.0, env.step_dt / self.window_seconds)
        self.contact_ema.mul_(1.0 - alpha).add_(contacts * alpha)
        duty_spread = torch.amax(self.contact_ema, dim=1) - torch.amin(self.contact_ema, dim=1)
        moving = env.command_manager.get_command(command_name)[:, 0] >= 0.1
        return torch.clamp(duty_spread - allowed_duty_spread, min=0.0) * moving


def worst_loaded_foot_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_normalization: float = 50.0,
    maximum_force_scale: float = 2.0,
) -> torch.Tensor:
    """Penalize the worst sliding loaded foot instead of hiding it in a sum."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    vertical_force = torch.abs(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    load_scale = torch.clamp(vertical_force / force_normalization, min=0.0, max=maximum_force_scale)
    horizontal_speed_sq = torch.sum(
        torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=2
    )
    return torch.amax(horizontal_speed_sq * load_scale, dim=1)


class TerrainRelativeSwingPeakPenalty(ManagerTermBase):
    """Penalize swing peak clearance outside a target band when each foot lands."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peak_clearance = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.peak_clearance[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        height_sensor_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        minimum_peak_clearance: float,
        maximum_peak_clearance: float,
        command_name: str,
    ) -> torch.Tensor:
        height_sensor = env.scene.sensors[height_sensor_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
        ray_hits = height_sensor.data.ray_hits_w
        xy_distance_sq = torch.sum(
            torch.square(foot_positions[:, :, None, :2] - ray_hits[:, None, :, :2]), dim=-1
        )
        nearest_indices = torch.argmin(xy_distance_sq, dim=-1)
        terrain_height = torch.gather(ray_hits[..., 2], 1, nearest_indices)
        clearance = foot_positions[..., 2] - terrain_height
        in_air = contact_sensor.data.current_air_time[:, contact_sensor_cfg.body_ids] > 0.0
        self.peak_clearance = torch.where(
            in_air, torch.maximum(self.peak_clearance, clearance), self.peak_clearance
        )
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, contact_sensor_cfg.body_ids]
        low = torch.clamp(minimum_peak_clearance - self.peak_clearance, min=0.0)
        high = torch.clamp(self.peak_clearance - maximum_peak_clearance, min=0.0)
        landing_penalty = torch.sum((torch.square(low) + torch.square(high)) * first_contact, dim=1)
        self.peak_clearance = torch.where(first_contact, torch.zeros_like(self.peak_clearance), self.peak_clearance)
        moving = env.command_manager.get_command(command_name)[:, 0] >= 0.1
        return landing_penalty * moving


def contact_foot_support_edge_penalty(
    env: ManagerBasedRLEnv,
    height_sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    support_radius: float = 0.14,
    maximum_support_height_span: float = 0.035,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize loaded feet whose local scan neighborhood straddles a box edge.

    A contact foot is considered poorly supported when nearby ray hits contain
    a large height discontinuity.  Force weighting focuses the penalty on
    stance feet and avoids shaping harmless swing motion.
    """
    height_sensor = env.scene.sensors[height_sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
    ray_hits = height_sensor.data.ray_hits_w
    distance_sq = torch.sum(
        torch.square(foot_positions[:, :, None, :2] - ray_hits[:, None, :, :2]), dim=-1
    )
    neighborhood = distance_sq <= support_radius * support_radius
    ray_heights = ray_hits[:, None, :, 2].expand(-1, foot_positions.shape[1], -1)
    local_min = torch.where(neighborhood, ray_heights, torch.inf).amin(dim=2)
    local_max = torch.where(neighborhood, ray_heights, -torch.inf).amax(dim=2)
    valid = torch.sum(neighborhood, dim=2) >= 2
    height_span = torch.where(valid, local_max - local_min, torch.zeros_like(local_min))
    edge_excess = torch.clamp(height_span - maximum_support_height_span, min=0.0)
    vertical_force = torch.abs(contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, 2])
    loaded = torch.clamp(vertical_force / 50.0, min=0.0, max=2.0)
    moving = env.command_manager.get_command(command_name)[:, 0] >= 0.1
    return torch.sum(edge_excess * loaded, dim=1) * moving


def diagonal_trot_contact_mismatch(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Soft contact mismatch for diagonal trot ordering FL, FR, RL, RR."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    if contacts.shape[1] != 4:
        raise ValueError(f"expected four feet, got {contacts.shape[1]}")
    same_diagonal = (contacts[:, 0] ^ contacts[:, 3]).float() + (contacts[:, 1] ^ contacts[:, 2]).float()
    opposite_diagonal = (~(contacts[:, 0] ^ contacts[:, 1])).float()
    moving = env.command_manager.get_command(command_name)[:, 0] >= 0.1
    return (same_diagonal + opposite_diagonal) * moving


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
