"""Final-command-aligned terrain teachers trained from random initialization."""

from __future__ import annotations

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .terrain_teacher_v1_env_cfg import TEACHER_TERRAINS
from .v5_foundation_env_cfg import (
    V5FoundationEnvCfg,
    V5FoundationObservationsCfg,
    V5FoundationPlayEnvCfg,
    V5FoundationRewardsCfg,
)


V2_FORWARD_SPEED_RANGES = {
    "flat": (0.0, 0.80),
    "random_rough": (0.10, 0.65),
    "pyramid_slope": (0.10, 0.55),
    "inverted_pyramid_slope": (0.10, 0.55),
    "pyramid_stairs": (0.15, 0.50),
    "inverted_pyramid_stairs": (0.15, 0.50),
    "boxes": (0.15, 0.50),
}
V2_MAX_YAW_RATES = {
    "flat": 0.60,
    "random_rough": 0.50,
    "pyramid_slope": 0.45,
    "inverted_pyramid_slope": 0.45,
    "pyramid_stairs": 0.30,
    "inverted_pyramid_stairs": 0.30,
    "boxes": 0.30,
}

FEET = SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])


@configclass
class TerrainTeacherV2RewardsCfg(V5FoundationRewardsCfg):
    """Tracking plus soft four-leg participation and locked-heading progress."""

    command_progress = RewTerm(
        func=mdp.target_heading_world_progress,
        weight=1.0,
        params={"command_name": "base_velocity", "standing_command_threshold": 0.1},
    )
    heading_error = RewTerm(
        func=mdp.heading_error_abs,
        weight=-0.25,
        params={"command_name": "base_velocity", "minimum_turning_speed": 0.1},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.02,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.35},
    )
    air_time_variance = RewTerm(func=mdp.air_time_variance_penalty, weight=-0.20, params={"sensor_cfg": FEET})
    excessive_foot_air_time = RewTerm(
        func=mdp.excessive_foot_air_time,
        weight=-1.0,
        params={"sensor_cfg": FEET, "maximum_air_time": 0.45, "command_name": "base_velocity"},
    )
    all_feet_airborne = RewTerm(
        func=mdp.all_feet_airborne,
        weight=-0.50,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )
    diagonal_trot_contact = RewTerm(
        func=mdp.diagonal_trot_contact_mismatch,
        weight=-0.05,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )


@configclass
class TerrainTeacherV2DiscreteRewardsCfg(TerrainTeacherV2RewardsCfg):
    """Obstacle rewards allow gait adaptation without permitting a tucked leg."""

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.40)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.12,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    diagonal_trot_contact = RewTerm(
        func=mdp.diagonal_trot_contact_mismatch,
        weight=-0.02,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )
    swing_foot_height = RewTerm(
        func=mdp.feet_height_body,
        weight=-0.15,
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
        params={"sensor_cfg": FEET},
    )


@configclass
class TerrainTeacherV2EnvCfg(V5FoundationEnvCfg):
    terrain_name: str = "flat"
    rewards: TerrainTeacherV2RewardsCfg = TerrainTeacherV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        name = self.terrain_name
        self.scene.terrain.terrain_generator = TEACHER_TERRAINS[name].copy()
        vx_range = V2_FORWARD_SPEED_RANGES[name]
        yaw_limit = V2_MAX_YAW_RATES[name]
        self.commands.base_velocity = mdp.RelativeHeadingVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            rel_standing_envs=0.05 if name == "flat" else 0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=True,
            ranges=mdp.RelativeHeadingVelocityCommandCfg.Ranges(
                lin_vel_x=vx_range,
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-yaw_limit, yaw_limit),
                heading=(-math.pi, math.pi),
            ),
            limit_ranges=mdp.RelativeHeadingVelocityCommandCfg.Ranges(
                lin_vel_x=vx_range,
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-yaw_limit, yaw_limit),
                heading=(-math.pi, math.pi),
            ),
            relative_heading_range=(-math.pi / 4.0, math.pi / 4.0),
            max_yaw_rate=yaw_limit,
            minimum_turn_radius=0.55,
            maximum_lateral_acceleration=0.45,
            minimum_turning_speed=0.10,
        )
        self.curriculum.lin_vel_cmd_levels = None
        if name == "flat":
            self.scene.terrain.max_init_terrain_level = 0
            self.scene.terrain.terrain_generator.curriculum = False
            self.curriculum.terrain_levels = None
        else:
            self.scene.terrain.max_init_terrain_level = 1
            self.scene.terrain.terrain_generator.curriculum = True
            self.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_world_progress_heading
            self.curriculum.terrain_levels.params = {
                "tracking_reward_term": "track_lin_vel_xy",
                "progress_reward_term": "command_progress",
                "heading_error_term": "heading_error",
                "tracking_fraction_threshold": 0.60,
                "minimum_progress_speed": 0.10 if "stairs" in name or name == "boxes" else 0.12,
                "maximum_mean_heading_error": 0.22,
            }
        # Structured terrains start approximately along the generated tile axis;
        # heading commands remain relative to the actual reset yaw.
        if "stairs" in name or name == "boxes":
            self.events.reset_base.params["pose_range"]["yaw"] = (-0.10, 0.10)


@configclass
class TerrainTeacherV2PlayEnvCfg(TerrainTeacherV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class FlatTeacherV2EnvCfg(TerrainTeacherV2EnvCfg):
    terrain_name: str = "flat"

@configclass
class FlatTeacherV2PlayEnvCfg(TerrainTeacherV2PlayEnvCfg):
    terrain_name: str = "flat"

@configclass
class RandomRoughTeacherV2EnvCfg(TerrainTeacherV2EnvCfg):
    terrain_name: str = "random_rough"

@configclass
class RandomRoughTeacherV2PlayEnvCfg(TerrainTeacherV2PlayEnvCfg):
    terrain_name: str = "random_rough"

@configclass
class PyramidSlopeTeacherV2EnvCfg(TerrainTeacherV2EnvCfg):
    terrain_name: str = "pyramid_slope"

@configclass
class PyramidSlopeTeacherV2PlayEnvCfg(TerrainTeacherV2PlayEnvCfg):
    terrain_name: str = "pyramid_slope"

@configclass
class InvertedPyramidSlopeTeacherV2EnvCfg(TerrainTeacherV2EnvCfg):
    terrain_name: str = "inverted_pyramid_slope"

@configclass
class InvertedPyramidSlopeTeacherV2PlayEnvCfg(TerrainTeacherV2PlayEnvCfg):
    terrain_name: str = "inverted_pyramid_slope"

@configclass
class PyramidStairsTeacherV2EnvCfg(TerrainTeacherV2EnvCfg):
    terrain_name: str = "pyramid_stairs"
    rewards: TerrainTeacherV2DiscreteRewardsCfg = TerrainTeacherV2DiscreteRewardsCfg()

@configclass
class PyramidStairsTeacherV2PlayEnvCfg(TerrainTeacherV2PlayEnvCfg):
    terrain_name: str = "pyramid_stairs"
    rewards: TerrainTeacherV2DiscreteRewardsCfg = TerrainTeacherV2DiscreteRewardsCfg()


@configclass
class PyramidStairsRemediationRewardsCfg(TerrainTeacherV2DiscreteRewardsCfg):
    """Terrain-relative clearance and anti-drag objective for natural stair gait."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.15,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.55},
    )
    air_time_variance = RewTerm(func=mdp.air_time_variance_penalty, weight=-0.05, params={"sensor_cfg": FEET})
    excessive_foot_air_time = RewTerm(
        func=mdp.excessive_foot_air_time,
        weight=-0.15,
        params={"sensor_cfg": FEET, "maximum_air_time": 0.75, "command_name": "base_velocity"},
    )
    all_feet_airborne = RewTerm(
        func=mdp.all_feet_airborne,
        weight=-0.25,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )
    diagonal_trot_contact = RewTerm(
        func=mdp.diagonal_trot_contact_mismatch,
        weight=-0.03,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )
    swing_foot_height = None
    terrain_clearance = RewTerm(
        func=mdp.swing_foot_terrain_clearance,
        weight=-0.75,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "target_clearance": 0.08,
            "horizontal_speed_threshold": 0.05,
            "command_name": "base_velocity",
        },
    )
    feet_slide = None
    loaded_feet_slide = RewTerm(
        func=mdp.loaded_feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "force_normalization": 50.0,
            "maximum_force_scale": 2.0,
        },
    )
    excessive_contact_time = RewTerm(
        func=mdp.excessive_foot_contact_time,
        weight=-0.25,
        params={
            "sensor_cfg": FEET,
            "maximum_contact_time": 0.80,
            "command_name": "base_velocity",
        },
    )
    all_feet_contact = RewTerm(
        func=mdp.all_feet_in_contact,
        weight=-0.20,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.25)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )


@configclass
class PyramidStairsTeacherV2RemediationEnvCfg(TerrainTeacherV2EnvCfg):
    """From-scratch stair remediation with a gait-gated 2--14 cm curriculum."""

    terrain_name: str = "pyramid_stairs"
    rewards: PyramidStairsRemediationRewardsCfg = PyramidStairsRemediationRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        stairs_cfg = self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"]
        stairs_cfg.step_height_range = (0.02, 0.14)
        self.scene.terrain.max_init_terrain_level = 0

        # Preserve the shared 13x9 teacher/student schema, but shift it rearward
        # enough to cover hind feet as well as several upcoming stair treads.
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.30, 0.0, 20.0))
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(1.2, 0.8))

        self.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_world_progress_heading_gait
        self.curriculum.terrain_levels.params = {
            "tracking_reward_term": "track_lin_vel_xy",
            "progress_reward_term": "command_progress",
            "heading_error_term": "heading_error",
            "long_contact_term": "excessive_contact_time",
            "all_contact_term": "all_feet_contact",
            "tracking_fraction_threshold": 0.60,
            "minimum_progress_speed": 0.10,
            "maximum_mean_heading_error": 0.22,
            "maximum_mean_long_contact_penalty": 0.08,
            "maximum_all_feet_contact_fraction": 0.45,
        }


@configclass
class PyramidStairsTeacherV2RemediationPlayEnvCfg(PyramidStairsTeacherV2RemediationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class PyramidStairsRemediationV2ObservationsCfg(V5FoundationObservationsCfg):
    """Perception plus an explicit speed-adaptive trot clock for actor and critic."""

    @configclass
    class PolicyCfg(V5FoundationObservationsCfg.PolicyCfg):
        gait_clock = ObsTerm(
            func=mdp.gait_phase,
            params={
                "period": 0.75,
                "command_name": "base_velocity",
                "minimum_period": 0.55,
                "minimum_speed": 0.15,
                "maximum_speed": 0.50,
            },
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(V5FoundationObservationsCfg.CriticCfg):
        gait_clock = ObsTerm(
            func=mdp.gait_phase,
            params={
                "period": 0.75,
                "command_name": "base_velocity",
                "minimum_period": 0.55,
                "minimum_speed": 0.15,
                "maximum_speed": 0.50,
            },
        )

    critic: CriticCfg = CriticCfg()


@configclass
class PyramidStairsRemediationV2RewardsCfg(PyramidStairsRemediationRewardsCfg):
    """Phase, participation and landing-aware objective that forbids a tucked leg."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.10,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.30},
    )
    air_time_variance = RewTerm(func=mdp.air_time_variance_penalty, weight=-0.10, params={"sensor_cfg": FEET})
    excessive_foot_air_time = RewTerm(
        func=mdp.excessive_foot_air_time,
        weight=-0.75,
        params={"sensor_cfg": FEET, "maximum_air_time": 0.60, "command_name": "base_velocity"},
    )
    diagonal_trot_contact = None
    trot_phase = RewTerm(
        func=mdp.speed_adaptive_trot_phase_reward,
        weight=0.20,
        params={
            "sensor_cfg": FEET,
            "command_name": "base_velocity",
            "maximum_period": 0.75,
            "minimum_period": 0.55,
            "minimum_speed": 0.15,
            "maximum_speed": 0.50,
            "stance_ratio": 0.62,
            "minimum_terrain_scale": 0.20,
        },
    )
    contact_participation = RewTerm(
        func=mdp.FootContactParticipationPenalty,
        weight=-0.50,
        params={
            "sensor_cfg": FEET,
            "command_name": "base_velocity",
            "window_seconds": 1.5,
            "minimum_duty_factor": 0.25,
            "maximum_duty_factor": 0.90,
        },
    )
    terrain_clearance = RewTerm(
        func=mdp.swing_foot_terrain_clearance,
        weight=-0.75,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "target_clearance": 0.08,
            "horizontal_speed_threshold": 0.0,
            "command_name": "base_velocity",
        },
    )
    swing_peak = RewTerm(
        func=mdp.TerrainRelativeSwingPeakPenalty,
        weight=-0.50,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "minimum_peak_clearance": 0.06,
            "maximum_peak_clearance": 0.13,
            "command_name": "base_velocity",
        },
    )
    loaded_feet_slide = RewTerm(
        func=mdp.loaded_feet_slide,
        weight=-0.20,
        params={
            "sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "force_normalization": 50.0,
            "maximum_force_scale": 2.0,
            "square_speed": True,
        },
    )
    excessive_contact_time = RewTerm(
        func=mdp.excessive_foot_contact_time,
        weight=-0.12,
        params={
            "sensor_cfg": FEET,
            "maximum_contact_time": 1.0,
            "command_name": "base_velocity",
        },
    )
    all_feet_contact = RewTerm(
        func=mdp.all_feet_in_contact,
        weight=-0.05,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )


@configclass
class PyramidStairsTeacherV2RemediationV2EnvCfg(TerrainTeacherV2EnvCfg):
    """Second from-scratch remediation with explicit bilateral gait participation."""

    terrain_name: str = "pyramid_stairs"
    observations: PyramidStairsRemediationV2ObservationsCfg = PyramidStairsRemediationV2ObservationsCfg()
    rewards: PyramidStairsRemediationV2RewardsCfg = PyramidStairsRemediationV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        stairs_cfg = self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"]
        stairs_cfg.step_height_range = (0.02, 0.14)
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.30, 0.0, 20.0))
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(1.2, 0.8))
        self.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_world_progress_heading_gait_v2
        self.curriculum.terrain_levels.params = {
            "tracking_reward_term": "track_lin_vel_xy",
            "progress_reward_term": "command_progress",
            "heading_error_term": "heading_error",
            "long_contact_term": "excessive_contact_time",
            "long_air_term": "excessive_foot_air_time",
            "participation_term": "contact_participation",
            "phase_term": "trot_phase",
            "tracking_fraction_threshold": 0.60,
            "minimum_progress_speed": 0.10,
            "maximum_mean_heading_error": 0.22,
            "maximum_mean_long_contact_penalty": 0.06,
            "maximum_mean_long_air_penalty": 0.05,
            "maximum_mean_participation_penalty": 0.02,
            "minimum_phase_match": 0.35,
        }


@configclass
class PyramidStairsTeacherV2RemediationV2PlayEnvCfg(PyramidStairsTeacherV2RemediationV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class DiscreteTerrainTeacherV2RemediationRewardsCfg(PyramidStairsRemediationV2RewardsCfg):
    """Shared anti-exploit gait objective for stairs and discontinuous boxes."""

    heading_lateral_drift = RewTerm(
        func=mdp.target_heading_lateral_velocity_l2,
        weight=-0.50,
        params={"command_name": "base_velocity", "minimum_turning_speed": 0.1},
    )


@configclass
class InvertedPyramidStairsTeacherV2RemediationRewardsCfg(
    DiscreteTerrainTeacherV2RemediationRewardsCfg
):
    """Natural-gait tracking plus signed ascent progress for inverted stairs."""

    elevation_progress = RewTerm(
        func=mdp.target_heading_aligned_vertical_velocity,
        weight=0.50,
        params={
            "command_name": "base_velocity",
            "minimum_forward_progress": 0.05,
            "maximum_abs_vertical_speed": 0.35,
        },
    )


@configclass
class BoxesTeacherV2RemediationRewardsCfg(DiscreteTerrainTeacherV2RemediationRewardsCfg):
    """Natural-gait tracking with terrain-relative box-edge support shaping."""

    terrain_clearance = RewTerm(
        func=mdp.swing_foot_terrain_clearance,
        weight=-0.75,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "target_clearance": 0.07,
            "horizontal_speed_threshold": 0.0,
            "command_name": "base_velocity",
        },
    )
    swing_peak = RewTerm(
        func=mdp.TerrainRelativeSwingPeakPenalty,
        weight=-0.50,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "minimum_peak_clearance": 0.05,
            "maximum_peak_clearance": 0.12,
            "command_name": "base_velocity",
        },
    )
    support_edge = RewTerm(
        func=mdp.contact_foot_support_edge_penalty,
        weight=-0.50,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "support_radius": 0.14,
            "maximum_support_height_span": 0.035,
            "command_name": "base_velocity",
        },
    )


@configclass
class InvertedPyramidStairsTeacherV2RemediationEnvCfg(TerrainTeacherV2EnvCfg):
    """Ascending inverted stairs with explicit ascent, anti-circle and gait gates."""

    terrain_name: str = "inverted_pyramid_stairs"
    observations: PyramidStairsRemediationV2ObservationsCfg = PyramidStairsRemediationV2ObservationsCfg()
    rewards: InvertedPyramidStairsTeacherV2RemediationRewardsCfg = (
        InvertedPyramidStairsTeacherV2RemediationRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        stairs_cfg = self.scene.terrain.terrain_generator.sub_terrains["inverted_pyramid_stairs"]
        stairs_cfg.step_height_range = (0.02, 0.14)
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.30, 0.0, 20.0))
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(1.2, 0.8))
        self.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_world_progress_heading_gait_v3
        self.curriculum.terrain_levels.params = {
            "tracking_reward_term": "track_lin_vel_xy",
            "progress_reward_term": "command_progress",
            "heading_error_term": "heading_error",
            "lateral_drift_term": "heading_lateral_drift",
            "long_contact_term": "excessive_contact_time",
            "long_air_term": "excessive_foot_air_time",
            "participation_term": "contact_participation",
            "phase_term": "trot_phase",
            "elevation_progress_term": "elevation_progress",
            "tracking_fraction_threshold": 0.60,
            "minimum_progress_speed": 0.10,
            "maximum_mean_heading_error": 0.22,
            "maximum_mean_lateral_drift": 0.025,
            "maximum_mean_long_contact_penalty": 0.06,
            "maximum_mean_long_air_penalty": 0.05,
            "maximum_mean_participation_penalty": 0.02,
            "minimum_phase_match": 0.30,
            "minimum_mean_elevation_progress": 0.005,
        }


@configclass
class InvertedPyramidStairsTeacherV2RemediationPlayEnvCfg(
    InvertedPyramidStairsTeacherV2RemediationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class BoxesTeacherV2RemediationEnvCfg(TerrainTeacherV2EnvCfg):
    """Random boxes with edge-aware support, anti-circle and bilateral gait gates."""

    terrain_name: str = "boxes"
    observations: PyramidStairsRemediationV2ObservationsCfg = PyramidStairsRemediationV2ObservationsCfg()
    rewards: BoxesTeacherV2RemediationRewardsCfg = BoxesTeacherV2RemediationRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        boxes_cfg = self.scene.terrain.terrain_generator.sub_terrains["boxes"]
        boxes_cfg.grid_height_range = (0.02, 0.18)
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.30, 0.0, 20.0))
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(1.2, 0.8))
        self.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_world_progress_heading_gait_v3
        self.curriculum.terrain_levels.params = {
            "tracking_reward_term": "track_lin_vel_xy",
            "progress_reward_term": "command_progress",
            "heading_error_term": "heading_error",
            "lateral_drift_term": "heading_lateral_drift",
            "long_contact_term": "excessive_contact_time",
            "long_air_term": "excessive_foot_air_time",
            "participation_term": "contact_participation",
            "phase_term": "trot_phase",
            "support_edge_term": "support_edge",
            "tracking_fraction_threshold": 0.60,
            "minimum_progress_speed": 0.10,
            "maximum_mean_heading_error": 0.22,
            "maximum_mean_lateral_drift": 0.025,
            "maximum_mean_long_contact_penalty": 0.06,
            "maximum_mean_long_air_penalty": 0.05,
            "maximum_mean_participation_penalty": 0.02,
            "minimum_phase_match": 0.30,
            "maximum_mean_support_edge_penalty": 0.02,
        }


@configclass
class BoxesTeacherV2RemediationPlayEnvCfg(BoxesTeacherV2RemediationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class DiscreteTerrainTeacherV3RewardsCfg(DiscreteTerrainTeacherV2RemediationRewardsCfg):
    """Stricter worst-leg objective derived from the three V2 gait-gate failures."""

    excessive_foot_air_time = RewTerm(
        func=mdp.excessive_foot_air_time,
        weight=-1.0,
        params={"sensor_cfg": FEET, "maximum_air_time": 0.45, "command_name": "base_velocity"},
    )
    excessive_contact_time = RewTerm(
        func=mdp.excessive_foot_contact_time,
        weight=-0.30,
        params={
            "sensor_cfg": FEET,
            "maximum_contact_time": 0.85,
            "command_name": "base_velocity",
        },
    )
    contact_participation = RewTerm(
        func=mdp.FootContactParticipationPenalty,
        weight=-0.75,
        params={
            "sensor_cfg": FEET,
            "command_name": "base_velocity",
            "window_seconds": 1.5,
            "minimum_duty_factor": 0.40,
            "maximum_duty_factor": 0.88,
        },
    )
    duty_symmetry = RewTerm(
        func=mdp.FootContactDutySymmetryPenalty,
        weight=-0.75,
        params={
            "sensor_cfg": FEET,
            "window_seconds": 1.5,
            "allowed_duty_spread": 0.15,
            "command_name": "base_velocity",
        },
    )
    worst_loaded_slide = RewTerm(
        func=mdp.worst_loaded_foot_slide,
        weight=-0.30,
        params={
            "sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "force_normalization": 50.0,
            "maximum_force_scale": 2.0,
        },
    )
    trot_phase = RewTerm(
        func=mdp.speed_adaptive_trot_phase_reward,
        weight=0.25,
        params={
            "sensor_cfg": FEET,
            "command_name": "base_velocity",
            "maximum_period": 0.75,
            "minimum_period": 0.55,
            "minimum_speed": 0.15,
            "maximum_speed": 0.50,
            "stance_ratio": 0.62,
            "minimum_terrain_scale": 0.30,
        },
    )


@configclass
class InvertedPyramidStairsTeacherV3RewardsCfg(DiscreteTerrainTeacherV3RewardsCfg):
    elevation_progress = RewTerm(
        func=mdp.target_heading_aligned_vertical_velocity,
        weight=0.50,
        params={
            "command_name": "base_velocity",
            "minimum_forward_progress": 0.05,
            "maximum_abs_vertical_speed": 0.35,
        },
    )


@configclass
class BoxesTeacherV3RewardsCfg(DiscreteTerrainTeacherV3RewardsCfg):
    terrain_clearance = RewTerm(
        func=mdp.swing_foot_terrain_clearance,
        weight=-0.75,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "target_clearance": 0.07,
            "horizontal_speed_threshold": 0.0,
            "command_name": "base_velocity",
        },
    )
    swing_peak = RewTerm(
        func=mdp.TerrainRelativeSwingPeakPenalty,
        weight=-0.50,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "minimum_peak_clearance": 0.05,
            "maximum_peak_clearance": 0.12,
            "command_name": "base_velocity",
        },
    )
    support_edge = RewTerm(
        func=mdp.contact_foot_support_edge_penalty,
        weight=-0.40,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": FEET,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "support_radius": 0.14,
            "maximum_support_height_span": 0.035,
            "command_name": "base_velocity",
        },
    )
    trot_phase = RewTerm(
        func=mdp.speed_adaptive_trot_phase_reward,
        weight=0.15,
        params={
            "sensor_cfg": FEET,
            "command_name": "base_velocity",
            "maximum_period": 0.75,
            "minimum_period": 0.55,
            "minimum_speed": 0.15,
            "maximum_speed": 0.50,
            "stance_ratio": 0.62,
            "minimum_terrain_scale": 0.10,
        },
    )


def _set_v3_curriculum(env, *, elevation: bool = False, support_edge: bool = False) -> None:
    env.curriculum.terrain_levels.func = mdp.terrain_levels_tracking_world_progress_heading_gait_v3
    params = {
        "tracking_reward_term": "track_lin_vel_xy",
        "progress_reward_term": "command_progress",
        "heading_error_term": "heading_error",
        "lateral_drift_term": "heading_lateral_drift",
        "long_contact_term": "excessive_contact_time",
        "long_air_term": "excessive_foot_air_time",
        "participation_term": "contact_participation",
        "duty_symmetry_term": "duty_symmetry",
        "phase_term": "trot_phase",
        "stall_term": "commanded_stall",
        "tracking_fraction_threshold": 0.58,
        "minimum_progress_speed": 0.10,
        "maximum_mean_heading_error": 0.22,
        "maximum_mean_lateral_drift": 0.025,
        "maximum_mean_long_contact_penalty": 0.05,
        "maximum_mean_long_air_penalty": 0.04,
        "maximum_mean_participation_penalty": 0.025,
        "maximum_mean_duty_symmetry_penalty": 0.03,
        "minimum_phase_match": 0.25 if support_edge else 0.32,
        "maximum_mean_stall_fraction": 0.06,
    }
    if elevation:
        params["elevation_progress_term"] = "elevation_progress"
        params["minimum_mean_elevation_progress"] = 0.005
    if support_edge:
        params["support_edge_term"] = "support_edge"
        params["maximum_mean_support_edge_penalty"] = 0.035
    env.curriculum.terrain_levels.params = params


@configclass
class PyramidStairsTeacherV3EnvCfg(PyramidStairsTeacherV2RemediationV2EnvCfg):
    rewards: DiscreteTerrainTeacherV3RewardsCfg = DiscreteTerrainTeacherV3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0
        _set_v3_curriculum(self)


@configclass
class PyramidStairsTeacherV3PlayEnvCfg(PyramidStairsTeacherV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class InvertedPyramidStairsTeacherV3EnvCfg(InvertedPyramidStairsTeacherV2RemediationEnvCfg):
    rewards: InvertedPyramidStairsTeacherV3RewardsCfg = InvertedPyramidStairsTeacherV3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0
        _set_v3_curriculum(self, elevation=True)


@configclass
class InvertedPyramidStairsTeacherV3PlayEnvCfg(InvertedPyramidStairsTeacherV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class BoxesTeacherV3EnvCfg(BoxesTeacherV2RemediationEnvCfg):
    rewards: BoxesTeacherV3RewardsCfg = BoxesTeacherV3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Seed the population across levels 0--3 so the policy cannot overfit
        # level zero while waiting for every gait gate to pass simultaneously.
        self.scene.terrain.max_init_terrain_level = 3
        _set_v3_curriculum(self, support_edge=True)


@configclass
class BoxesTeacherV3PlayEnvCfg(BoxesTeacherV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_cols = 5

@configclass
class InvertedPyramidStairsTeacherV2EnvCfg(TerrainTeacherV2EnvCfg):
    terrain_name: str = "inverted_pyramid_stairs"
    rewards: TerrainTeacherV2DiscreteRewardsCfg = TerrainTeacherV2DiscreteRewardsCfg()

@configclass
class InvertedPyramidStairsTeacherV2PlayEnvCfg(TerrainTeacherV2PlayEnvCfg):
    terrain_name: str = "inverted_pyramid_stairs"
    rewards: TerrainTeacherV2DiscreteRewardsCfg = TerrainTeacherV2DiscreteRewardsCfg()

@configclass
class BoxesTeacherV2EnvCfg(TerrainTeacherV2EnvCfg):
    terrain_name: str = "boxes"
    rewards: TerrainTeacherV2DiscreteRewardsCfg = TerrainTeacherV2DiscreteRewardsCfg()

@configclass
class BoxesTeacherV2PlayEnvCfg(TerrainTeacherV2PlayEnvCfg):
    terrain_name: str = "boxes"
    rewards: TerrainTeacherV2DiscreteRewardsCfg = TerrainTeacherV2DiscreteRewardsCfg()
