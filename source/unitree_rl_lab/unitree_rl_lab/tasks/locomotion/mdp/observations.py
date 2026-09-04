from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str | None = None,
    minimum_period: float | None = None,
    minimum_speed: float = 0.15,
    maximum_speed: float = 0.50,
) -> torch.Tensor:
    """Return a deployable sin/cos gait clock, optionally speed-adaptive."""
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    period_per_env = torch.full((env.num_envs,), period, device=env.device)
    if command_name is not None:
        if minimum_period is None:
            raise ValueError("minimum_period is required for a speed-adaptive gait clock")
        command_speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
        speed_fraction = torch.clamp(
            (command_speed - minimum_speed) / max(maximum_speed - minimum_speed, 1.0e-6), 0.0, 1.0
        )
        period_per_env = period + speed_fraction * (minimum_period - period)
    global_phase = torch.remainder(env.episode_length_buf * env.step_dt, period_per_env) / period_per_env

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
