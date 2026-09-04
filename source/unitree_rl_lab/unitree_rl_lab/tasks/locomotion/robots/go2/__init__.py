import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_velocity_env_cfg:RoughRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_velocity_env_cfg:RoughRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-MultiTerrain",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multiterrain_velocity_env_cfg:MultiTerrainRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.multiterrain_velocity_env_cfg:MultiTerrainRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-MultiTerrain-TrackingAware",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multiterrain_velocity_env_cfg:TrackingAwareMultiTerrainRobotEnvCfg"
        ),
        "play_env_cfg_entry_point": f"{__name__}.multiterrain_velocity_env_cfg:MultiTerrainRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-MultiTerrain-HeightScan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multiterrain_velocity_env_cfg:HeightScanMultiTerrainRobotEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.multiterrain_velocity_env_cfg:HeightScanMultiTerrainRobotPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:HeightScanPPORunnerCfg"
        ),
    },
)

# Evaluation-only namespace for recurrent students distilled on the seven-terrain
# HeightScan schema.  Keeping this separate avoids changing the historical
# HeightScan PPO runner architecture used by existing checkpoints.
gym.register(
    id="Unitree-Go2-Velocity-ProxyStudent-HeightScan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multiterrain_velocity_env_cfg:HeightScanMultiTerrainRobotEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.multiterrain_velocity_env_cfg:HeightScanMultiTerrainRobotPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:V5FoundationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-V5-Foundation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v5_foundation_env_cfg:V5FoundationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.v5_foundation_env_cfg:V5FoundationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:V5FoundationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-V5-Stairs-Specialist",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v5_stairs_specialist_env_cfg:V5StairsSpecialistEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.v5_stairs_specialist_env_cfg:V5StairsSpecialistPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "V5StairsSpecialistPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-V5-Boxes-Specialist",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v5_boxes_specialist_env_cfg:V5BoxesSpecialistEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.v5_boxes_specialist_env_cfg:V5BoxesSpecialistPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "V5BoxesSpecialistPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-V5-Boxes-Mixed-Command-Remediation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.v5_boxes_mixed_command_remediation_env_cfg:"
            "V5BoxesMixedCommandRemediationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.v5_boxes_mixed_command_remediation_env_cfg:"
            "V5BoxesMixedCommandRemediationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "V5BoxesMixedCommandRemediationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-V5-Boxes-Safety-Remediation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.v5_boxes_safety_remediation_env_cfg:"
            "V5BoxesSafetyRemediationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.v5_boxes_safety_remediation_env_cfg:"
            "V5BoxesSafetyRemediationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "V5BoxesSafetyRemediationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-V5-Stairs-Ascending-Remediation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.v5_stairs_ascending_remediation_env_cfg:"
            "V5StairsAscendingRemediationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.v5_stairs_ascending_remediation_env_cfg:"
            "V5StairsAscendingRemediationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "V5StairsAscendingRemediationPPORunnerCfg"
        ),
    },
)


def _register_terrain_teacher_v1(task_suffix: str, cfg_class: str, play_cfg_class: str) -> None:
    """Register one isolated from-scratch terrain-teacher task."""

    gym.register(
        id=f"Unitree-Go2-Velocity-TeacherV1-{task_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.terrain_teacher_v1_env_cfg:{cfg_class}",
            "play_env_cfg_entry_point": f"{__name__}.terrain_teacher_v1_env_cfg:{play_cfg_class}",
            "rsl_rl_cfg_entry_point": (
                "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
                "TerrainTeacherV1PPORunnerCfg"
            ),
        },
    )


_register_terrain_teacher_v1("Flat", "FlatTeacherV1EnvCfg", "FlatTeacherV1PlayEnvCfg")
_register_terrain_teacher_v1(
    "RandomRough", "RandomRoughTeacherV1EnvCfg", "RandomRoughTeacherV1PlayEnvCfg"
)
_register_terrain_teacher_v1(
    "PyramidSlope", "PyramidSlopeTeacherV1EnvCfg", "PyramidSlopeTeacherV1PlayEnvCfg"
)
_register_terrain_teacher_v1(
    "InvertedPyramidSlope",
    "InvertedPyramidSlopeTeacherV1EnvCfg",
    "InvertedPyramidSlopeTeacherV1PlayEnvCfg",
)
_register_terrain_teacher_v1(
    "PyramidStairs", "PyramidStairsTeacherV1EnvCfg", "PyramidStairsTeacherV1PlayEnvCfg"
)
_register_terrain_teacher_v1(
    "InvertedPyramidStairs",
    "InvertedPyramidStairsTeacherV1EnvCfg",
    "InvertedPyramidStairsTeacherV1PlayEnvCfg",
)
_register_terrain_teacher_v1("Boxes", "BoxesTeacherV1EnvCfg", "BoxesTeacherV1PlayEnvCfg")


def _register_terrain_teacher_v2(task_suffix: str, cfg_class: str, play_cfg_class: str) -> None:
    gym.register(
        id=f"Unitree-Go2-Velocity-TeacherV2-{task_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.terrain_teacher_v2_env_cfg:{cfg_class}",
            "play_env_cfg_entry_point": f"{__name__}.terrain_teacher_v2_env_cfg:{play_cfg_class}",
            "rsl_rl_cfg_entry_point": (
                "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
                "TerrainTeacherV2PPORunnerCfg"
            ),
        },
    )


_register_terrain_teacher_v2("Flat", "FlatTeacherV2EnvCfg", "FlatTeacherV2PlayEnvCfg")
_register_terrain_teacher_v2(
    "RandomRough", "RandomRoughTeacherV2EnvCfg", "RandomRoughTeacherV2PlayEnvCfg"
)
_register_terrain_teacher_v2(
    "PyramidSlope", "PyramidSlopeTeacherV2EnvCfg", "PyramidSlopeTeacherV2PlayEnvCfg"
)
_register_terrain_teacher_v2(
    "InvertedPyramidSlope",
    "InvertedPyramidSlopeTeacherV2EnvCfg",
    "InvertedPyramidSlopeTeacherV2PlayEnvCfg",
)
_register_terrain_teacher_v2(
    "PyramidStairs", "PyramidStairsTeacherV2EnvCfg", "PyramidStairsTeacherV2PlayEnvCfg"
)
_register_terrain_teacher_v2(
    "InvertedPyramidStairs",
    "InvertedPyramidStairsTeacherV2EnvCfg",
    "InvertedPyramidStairsTeacherV2PlayEnvCfg",
)
_register_terrain_teacher_v2("Boxes", "BoxesTeacherV2EnvCfg", "BoxesTeacherV2PlayEnvCfg")

gym.register(
    id="Unitree-Go2-Velocity-TeacherV2-PyramidStairsRemediation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.terrain_teacher_v2_env_cfg:PyramidStairsTeacherV2RemediationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.terrain_teacher_v2_env_cfg:PyramidStairsTeacherV2RemediationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "PyramidStairsTeacherV2RemediationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-TeacherV2-PyramidStairsRemediationV2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.terrain_teacher_v2_env_cfg:PyramidStairsTeacherV2RemediationV2EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.terrain_teacher_v2_env_cfg:PyramidStairsTeacherV2RemediationV2PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "PyramidStairsTeacherV2RemediationV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-TeacherV2-InvertedPyramidStairsRemediation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.terrain_teacher_v2_env_cfg:"
            "InvertedPyramidStairsTeacherV2RemediationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.terrain_teacher_v2_env_cfg:"
            "InvertedPyramidStairsTeacherV2RemediationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "InvertedPyramidStairsTeacherV2RemediationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Velocity-TeacherV2-BoxesRemediation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_teacher_v2_env_cfg:BoxesTeacherV2RemediationEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.terrain_teacher_v2_env_cfg:BoxesTeacherV2RemediationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "BoxesTeacherV2RemediationPPORunnerCfg"
        ),
    },
)


def _register_discrete_teacher_v3(name: str, env_class: str, play_class: str, runner_class: str):
    gym.register(
        id=f"Unitree-Go2-Velocity-TeacherV3-{name}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.terrain_teacher_v2_env_cfg:{env_class}",
            "play_env_cfg_entry_point": f"{__name__}.terrain_teacher_v2_env_cfg:{play_class}",
            "rsl_rl_cfg_entry_point": (
                "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:" + runner_class
            ),
        },
    )


_register_discrete_teacher_v3(
    "PyramidStairs",
    "PyramidStairsTeacherV3EnvCfg",
    "PyramidStairsTeacherV3PlayEnvCfg",
    "PyramidStairsTeacherV3PPORunnerCfg",
)
_register_discrete_teacher_v3(
    "InvertedPyramidStairs",
    "InvertedPyramidStairsTeacherV3EnvCfg",
    "InvertedPyramidStairsTeacherV3PlayEnvCfg",
    "InvertedPyramidStairsTeacherV3PPORunnerCfg",
)
_register_discrete_teacher_v3(
    "Boxes",
    "BoxesTeacherV3EnvCfg",
    "BoxesTeacherV3PlayEnvCfg",
    "BoxesTeacherV3PPORunnerCfg",
)
