"""V5 perceptive foundation task trained from random initialization.

Stage 1 intentionally contains only smooth terrain, moderate roughness, and
ascending/descending slopes.  Stairs and boxes enter later specialist stages;
this task establishes command tracking and terrain-conditioned balance without
letting discrete obstacles dominate early exploration.
"""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import ObservationsCfg, RewardsCfg, RobotEnvCfg


GO2_V5_FOUNDATION_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.005, 0.05), noise_step=0.005, border_width=0.25
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.02, 0.25), platform_width=2.0, border_width=0.25
        ),
        "inverted_pyramid_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.02, 0.25), platform_width=2.0, border_width=0.25
        ),
    },
)


@configclass
class V5FoundationObservationsCfg(ObservationsCfg):
    """Noisy perceptive actor and clean asymmetric critic observations."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class V5FoundationRewardsCfg(RewardsCfg):
    """Tracking-first objective that makes stationary survival unprofitable."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.30},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    command_progress = RewTerm(
        func=mdp.command_direction_progress,
        weight=1.0,
        params={"command_name": "base_velocity", "standing_command_threshold": 0.1},
    )
    commanded_stall = RewTerm(
        func=mdp.commanded_motion_stall,
        weight=-0.75,
        params={
            "command_name": "base_velocity",
            "command_speed_threshold": 0.2,
            "progress_speed_threshold": 0.05,
        },
    )

    # Preserve safety and smoothness, but avoid forcing a flat-ground posture
    # or tiny joint excursions on sloped terrain.
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    air_time_variance = None


@configclass
class V5FoundationEnvCfg(RobotEnvCfg):
    observations: V5FoundationObservationsCfg = V5FoundationObservationsCfg()
    rewards: V5FoundationRewardsCfg = V5FoundationRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GO2_V5_FOUNDATION_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator.curriculum = True

        # Forward-biased 13 x 9 scan.  The recurrent policy supplies temporal
        # context instead of flattening an arbitrary observation history.
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.45, 0.0, 20.0))
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(1.2, 0.8))

        self.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_command_progress
        self.curriculum.terrain_levels.params = {
            "tracking_reward_term": "track_lin_vel_xy",
            "progress_reward_term": "command_progress",
            "tracking_fraction_threshold": 0.60,
            "minimum_progress_speed": 0.12,
        }
        self.curriculum.lin_vel_cmd_levels = None

        # One command per episode makes stage-1 attribution and curriculum
        # metrics unambiguous.  Later stages may restore command transitions.
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)


@configclass
class V5FoundationPlayEnvCfg(V5FoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 16
