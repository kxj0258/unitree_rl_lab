"""Mixed-command remediation for the forward-specialized v5 boxes policy."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .v5_boxes_specialist_env_cfg import V5BoxesSpecialistEnvCfg, V5BoxesSpecialistPlayEnvCfg


@configclass
class V5BoxesMixedCommandRemediationEnvCfg(V5BoxesSpecialistEnvCfg):
    """Expand model3999 from forward boxes traversal to the shared command schema."""

    def __post_init__(self):
        super().__post_init__()

        # Start mostly inside the proven forward distribution. The command
        # curriculum progressively exposes reverse and wider lateral motion.
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.60)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.40, 0.40)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.40, 0.60)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.40, 0.40)

        self.curriculum.lin_vel_cmd_levels = CurrTerm(
            func=mdp.boxes_command_ranges_tracking_progress,
            params={
                "tracking_reward_term": "track_lin_vel_xy",
                "progress_reward_term": "command_progress",
                "tracking_fraction_threshold": 0.65,
                "minimum_progress_speed": 0.20,
                "reverse_step": 0.05,
                "lateral_step": 0.025,
            },
        )


@configclass
class V5BoxesMixedCommandRemediationPlayEnvCfg(V5BoxesMixedCommandRemediationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 20
