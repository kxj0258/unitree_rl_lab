"""Hard-terrain safety remediation for the v5 boxes mixed-command policy."""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .v5_boxes_mixed_command_remediation_env_cfg import (
    V5BoxesMixedCommandRemediationEnvCfg,
    V5BoxesMixedCommandRemediationPlayEnvCfg,
)
from .v5_boxes_specialist_env_cfg import V5BoxesSpecialistRewardsCfg


# Twenty columns give an exact 1/1/18 split. Levels are sampled uniformly at
# reset and then frozen, so failed hard-terrain environments cannot escape the
# safety objective by being demoted by a runtime terrain curriculum.
GO2_V5_BOXES_SAFETY_REMEDIATION_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
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
            proportion=0.05,
            noise_range=(0.005, 0.05),
            noise_step=0.005,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.90,
            grid_width=0.45,
            grid_height_range=(0.02, 0.18),
            platform_width=2.0,
            holes=False,
        ),
    },
)


@configclass
class V5BoxesSafetyRemediationRewardsCfg(V5BoxesSpecialistRewardsCfg):
    """Strengthen dense precursors of the observed terminal safety failures."""

    base_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-4.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
        },
    )

@configclass
class V5BoxesSafetyRemediationEnvCfg(V5BoxesMixedCommandRemediationEnvCfg):
    """Focus model1999 on hard boxes and its unsafe combined commands."""

    rewards: V5BoxesSafetyRemediationRewardsCfg = V5BoxesSafetyRemediationRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GO2_V5_BOXES_SAFETY_REMEDIATION_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 7
        self.scene.terrain.terrain_generator.curriculum = True
        self.curriculum.terrain_levels = None

        self.rewards.commanded_stall.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -0.40
        self.rewards.action_rate.weight = -0.06
        self.rewards.feet_stumble.weight = -1.0
        self.rewards.undesired_contacts.weight = -1.5

        # Keep the full shared command envelope from the first remediation.
        # Slight negative-side bias targets lateral-right and reverse-yaw while
        # retaining positive-direction rehearsal to guard against forgetting.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.40, 0.40)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.20)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.40, 0.35)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.40, 0.60)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.40, 0.40)
        self.curriculum.lin_vel_cmd_levels = None


@configclass
class V5BoxesSafetyRemediationPlayEnvCfg(V5BoxesSafetyRemediationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 8
        self.scene.terrain.terrain_generator.num_cols = 20
