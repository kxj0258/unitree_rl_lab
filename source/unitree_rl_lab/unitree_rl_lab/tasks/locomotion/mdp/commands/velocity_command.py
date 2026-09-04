from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class RelativeHeadingVelocityCommand(UniformVelocityCommand):
    """Forward command with a heading target sampled relative to yaw at resampling.

    The target stays fixed for the complete command segment.  Yaw-rate output is
    limited by an absolute cap, a minimum-turn-radius constraint, and a lateral-
    acceleration constraint.  Standing commands never request in-place rotation.
    """

    cfg: "RelativeHeadingVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        relative_heading = r.uniform_(*self.cfg.relative_heading_range)
        self.heading_target[env_ids] = math_utils.wrap_to_pi(
            self.robot.data.heading_w[env_ids] + relative_heading
        )
        self.is_heading_env[env_ids] = True

    def _update_command(self):
        heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        vx = torch.clamp(self.vel_command_b[:, 0], min=0.0)
        eps = 1.0e-6
        yaw_limit = torch.full_like(vx, self.cfg.max_yaw_rate)
        yaw_limit = torch.minimum(yaw_limit, vx / self.cfg.minimum_turn_radius)
        yaw_limit = torch.minimum(
            yaw_limit,
            self.cfg.maximum_lateral_acceleration / torch.clamp(vx, min=eps),
        )
        yaw_limit = torch.where(vx >= self.cfg.minimum_turning_speed, yaw_limit, 0.0)
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = torch.clamp(
            self.cfg.heading_control_stiffness * heading_error,
            min=-yaw_limit,
            max=yaw_limit,
        )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class RelativeHeadingVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Configuration for :class:`RelativeHeadingVelocityCommand`."""

    class_type: type = RelativeHeadingVelocityCommand
    relative_heading_range: tuple[float, float] = (-0.7853981633974483, 0.7853981633974483)
    max_yaw_rate: float = 0.6
    minimum_turn_radius: float = 0.55
    maximum_lateral_acceleration: float = 0.45
    minimum_turning_speed: float = 0.10
