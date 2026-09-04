"""Focused remediation for ascending inverted-pyramid stairs."""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .v5_stairs_specialist_env_cfg import (
    V5StairsSpecialistEnvCfg,
    V5StairsSpecialistPlayEnvCfg,
    V5StairsSpecialistRewardsCfg,
)


GO2_V5_STAIRS_ASCENDING_REMEDIATION_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05, noise_range=(0.005, 0.05), noise_step=0.005, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.03, 0.14),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "inverted_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.70,
            step_height_range=(0.03, 0.14),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class V5StairsAscendingRemediationRewardsCfg(V5StairsSpecialistRewardsCfg):
    """Make sustained heading control a first-class stair objective."""

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )


@configclass
class V5StairsAscendingRemediationEnvCfg(V5StairsSpecialistEnvCfg):
    rewards: V5StairsAscendingRemediationRewardsCfg = V5StairsAscendingRemediationRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GO2_V5_STAIRS_ASCENDING_REMEDIATION_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator.curriculum = True

        self.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_progress_yaw
        self.curriculum.terrain_levels.params = {
            "tracking_reward_term": "track_lin_vel_xy",
            "progress_reward_term": "command_progress",
            "yaw_reward_term": "track_ang_vel_z",
            "tracking_fraction_threshold": 0.55,
            "minimum_progress_speed": 0.15,
            "yaw_fraction_threshold": 0.70,
        }

        # Train one reproducible world direction. Heading feedback becomes
        # non-zero only after drift, directly targeting the observed failure.
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.40, 0.40)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class V5StairsAscendingRemediationPlayEnvCfg(V5StairsAscendingRemediationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 20
