"""V5 boxes specialist transferred from the formal perceptive foundation."""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .v5_foundation_env_cfg import V5FoundationEnvCfg, V5FoundationRewardsCfg


# Keep proportions aligned to the 20 terrain columns: 14 boxes columns and
# three rehearsal columns for each continuous terrain.  The central platform
# provides a reproducible spawn area; curriculum rows control grid height.
GO2_V5_BOXES_SPECIALIST_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.005, 0.05),
            noise_step=0.005,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.70,
            grid_width=0.45,
            grid_height_range=(0.02, 0.18),
            platform_width=2.0,
            holes=False,
        ),
    },
)


@configclass
class V5BoxesSpecialistRewardsCfg(V5FoundationRewardsCfg):
    """Encourage deliberate obstacle clearance without rewarding survival alone."""

    # Allow obstacle-crossing pitch and larger joint excursions while keeping
    # the foundation tracking/progress/stall and contact-safety objectives.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.25)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.075,
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
        weight=-0.5,
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
class V5BoxesSpecialistEnvCfg(V5FoundationEnvCfg):
    rewards: V5BoxesSpecialistRewardsCfg = V5BoxesSpecialistRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GO2_V5_BOXES_SPECIALIST_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator.curriculum = True

        self.curriculum.terrain_levels.params = {
            "tracking_reward_term": "track_lin_vel_xy",
            "progress_reward_term": "command_progress",
            "tracking_fraction_threshold": 0.55,
            "minimum_progress_speed": 0.12,
        }
        self.curriculum.lin_vel_cmd_levels = None

        # A full-episode command makes obstacle traversal attribution clear.
        # Forward motion dominates exploration, while lateral/yaw coverage is
        # retained for the future unified-student command schema.
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.60)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.40, 0.40)

        # Preserve the exact foundation action semantics during transfer.
        self.actions.JointPositionAction.scale = 0.25


@configclass
class V5BoxesSpecialistPlayEnvCfg(V5BoxesSpecialistEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 20
