"""Seven independent from-scratch perceptive teacher tasks.

All tasks intentionally share the same deployable actor observation, privileged
critic observation, action semantics, recurrent PPO architecture, and command
field ordering.  Only terrain geometry, feasible command envelope, and a small
number of terrain-specific reward weights differ.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .v5_foundation_env_cfg import (
    V5FoundationEnvCfg,
    V5FoundationPlayEnvCfg,
    V5FoundationRewardsCfg,
)


def _generator(sub_terrain) -> terrain_gen.TerrainGeneratorCfg:
    return terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains=sub_terrain,
    )


TEACHER_TERRAINS = {
    "flat": _generator({"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}),
    "random_rough": _generator(
        {
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(0.005, 0.08),
                noise_step=0.005,
                border_width=0.25,
            )
        }
    ),
    "pyramid_slope": _generator(
        {
            "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(0.02, 0.35),
                platform_width=2.0,
                border_width=0.25,
            )
        }
    ),
    "inverted_pyramid_slope": _generator(
        {
            "inverted_pyramid_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(0.02, 0.35),
                platform_width=2.0,
                border_width=0.25,
            )
        }
    ),
    "pyramid_stairs": _generator(
        {
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.03, 0.18),
                step_width=0.30,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            )
        }
    ),
    "inverted_pyramid_stairs": _generator(
        {
            "inverted_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.03, 0.18),
                step_width=0.30,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            )
        }
    ),
    "boxes": _generator(
        {
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=1.0,
                grid_width=0.45,
                grid_height_range=(0.02, 0.18),
                platform_width=2.0,
                holes=False,
            )
        }
    ),
}


# Body-frame feasible command domains.  These are deliberately not a Cartesian
# promise that every terrain supports the same manoeuvre.  The schema remains
# [vx, vy, yaw_rate], while hard discrete terrains emphasize traversable forward
# motion with bounded lateral correction and steering.
FEASIBLE_COMMAND_DOMAINS = {
    "flat": {"lin_vel_x": (-0.60, 0.80), "lin_vel_y": (-0.35, 0.35), "ang_vel_z": (-0.80, 0.80)},
    "random_rough": {"lin_vel_x": (-0.30, 0.65), "lin_vel_y": (-0.20, 0.20), "ang_vel_z": (-0.60, 0.60)},
    "pyramid_slope": {"lin_vel_x": (-0.20, 0.55), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-0.50, 0.50)},
    "inverted_pyramid_slope": {"lin_vel_x": (-0.20, 0.55), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-0.50, 0.50)},
    "pyramid_stairs": {"lin_vel_x": (0.15, 0.50), "lin_vel_y": (-0.08, 0.08), "ang_vel_z": (-0.30, 0.30)},
    "inverted_pyramid_stairs": {"lin_vel_x": (0.15, 0.50), "lin_vel_y": (-0.08, 0.08), "ang_vel_z": (-0.30, 0.30)},
    "boxes": {"lin_vel_x": (0.15, 0.50), "lin_vel_y": (-0.08, 0.08), "ang_vel_z": (-0.30, 0.30)},
}


@configclass
class TerrainTeacherV1DiscreteRewardsCfg(V5FoundationRewardsCfg):
    """Clearance-aware objective for stairs and boxes teachers."""

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.20)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.20,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.35,
        },
    )
    swing_foot_height = RewTerm(
        func=mdp.feet_height_body,
        weight=-0.50,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": -0.18,
            "tanh_mult": 2.0,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.75,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )


@configclass
class TerrainTeacherV1EnvCfg(V5FoundationEnvCfg):
    """Common contract; concrete subclasses set ``terrain_name``."""

    terrain_name: str = "flat"

    def __post_init__(self):
        super().__post_init__()
        name = self.terrain_name
        self.scene.terrain.terrain_generator = TEACHER_TERRAINS[name].copy()
        domain = FEASIBLE_COMMAND_DOMAINS[name]
        self.commands.base_velocity.ranges.lin_vel_x = domain["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = domain["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = domain["ang_vel_z"]
        self.commands.base_velocity.limit_ranges.lin_vel_x = domain["lin_vel_x"]
        self.commands.base_velocity.limit_ranges.lin_vel_y = domain["lin_vel_y"]
        self.commands.base_velocity.limit_ranges.ang_vel_z = domain["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.rel_standing_envs = 0.03 if name == "flat" else 0.0
        self.curriculum.lin_vel_cmd_levels = None
        self.actions.JointPositionAction.scale = 0.25

        if name == "flat":
            self.scene.terrain.max_init_terrain_level = 0
            self.scene.terrain.terrain_generator.curriculum = False
            self.curriculum.terrain_levels = None
        else:
            self.scene.terrain.max_init_terrain_level = 1
            self.scene.terrain.terrain_generator.curriculum = True
            self.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_command_progress
            self.curriculum.terrain_levels.params = {
                "tracking_reward_term": "track_lin_vel_xy",
                "progress_reward_term": "command_progress",
                "tracking_fraction_threshold": 0.55,
                "minimum_progress_speed": 0.10 if "stairs" in name or name == "boxes" else 0.12,
            }


@configclass
class TerrainTeacherV1PlayEnvCfg(TerrainTeacherV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class FlatTeacherV1EnvCfg(TerrainTeacherV1EnvCfg):
    terrain_name: str = "flat"


@configclass
class FlatTeacherV1PlayEnvCfg(TerrainTeacherV1PlayEnvCfg):
    terrain_name: str = "flat"


@configclass
class RandomRoughTeacherV1EnvCfg(TerrainTeacherV1EnvCfg):
    terrain_name: str = "random_rough"


@configclass
class RandomRoughTeacherV1PlayEnvCfg(TerrainTeacherV1PlayEnvCfg):
    terrain_name: str = "random_rough"


@configclass
class PyramidSlopeTeacherV1EnvCfg(TerrainTeacherV1EnvCfg):
    terrain_name: str = "pyramid_slope"


@configclass
class PyramidSlopeTeacherV1PlayEnvCfg(TerrainTeacherV1PlayEnvCfg):
    terrain_name: str = "pyramid_slope"


@configclass
class InvertedPyramidSlopeTeacherV1EnvCfg(TerrainTeacherV1EnvCfg):
    terrain_name: str = "inverted_pyramid_slope"


@configclass
class InvertedPyramidSlopeTeacherV1PlayEnvCfg(TerrainTeacherV1PlayEnvCfg):
    terrain_name: str = "inverted_pyramid_slope"


@configclass
class PyramidStairsTeacherV1EnvCfg(TerrainTeacherV1EnvCfg):
    terrain_name: str = "pyramid_stairs"
    rewards: TerrainTeacherV1DiscreteRewardsCfg = TerrainTeacherV1DiscreteRewardsCfg()


@configclass
class PyramidStairsTeacherV1PlayEnvCfg(TerrainTeacherV1PlayEnvCfg):
    terrain_name: str = "pyramid_stairs"
    rewards: TerrainTeacherV1DiscreteRewardsCfg = TerrainTeacherV1DiscreteRewardsCfg()


@configclass
class InvertedPyramidStairsTeacherV1EnvCfg(TerrainTeacherV1EnvCfg):
    terrain_name: str = "inverted_pyramid_stairs"
    rewards: TerrainTeacherV1DiscreteRewardsCfg = TerrainTeacherV1DiscreteRewardsCfg()


@configclass
class InvertedPyramidStairsTeacherV1PlayEnvCfg(TerrainTeacherV1PlayEnvCfg):
    terrain_name: str = "inverted_pyramid_stairs"
    rewards: TerrainTeacherV1DiscreteRewardsCfg = TerrainTeacherV1DiscreteRewardsCfg()


@configclass
class BoxesTeacherV1EnvCfg(TerrainTeacherV1EnvCfg):
    terrain_name: str = "boxes"
    rewards: TerrainTeacherV1DiscreteRewardsCfg = TerrainTeacherV1DiscreteRewardsCfg()


@configclass
class BoxesTeacherV1PlayEnvCfg(TerrainTeacherV1PlayEnvCfg):
    terrain_name: str = "boxes"
    rewards: TerrainTeacherV1DiscreteRewardsCfg = TerrainTeacherV1DiscreteRewardsCfg()
