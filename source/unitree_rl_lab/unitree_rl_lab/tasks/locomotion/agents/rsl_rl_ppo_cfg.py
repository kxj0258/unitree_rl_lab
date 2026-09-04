# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeightScanPPORunnerCfg(BasePPORunnerCfg):
    """Conservative fixed-rate PPO config for newly introduced scan inputs."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class V5FoundationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """From-scratch recurrent PPO for the v5 perceptive foundation stage."""

    num_steps_per_env = 32
    # 1024 envs x 32 steps x 5000 updates = 163.84M transitions.  This is
    # slightly above the stock Isaac Lab Go2 rough budget (4096 x 24 x 1500)
    # and leaves room for the recurrent perceptive policy and terrain curriculum.
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_v5_foundation"
    empirical_normalization = None
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TerrainTeacherV1PPORunnerCfg(V5FoundationPPORunnerCfg):
    """Shared from-scratch recurrent PPO contract for all seven v1 teachers."""

    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_7terrain_teacher_v1"


@configclass
class TerrainTeacherV2PPORunnerCfg(V5FoundationPPORunnerCfg):
    """From-scratch PPO for the final-command-aligned teacher set."""

    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_7terrain_teacher_v2"


@configclass
class PyramidStairsTeacherV2RemediationPPORunnerCfg(V5FoundationPPORunnerCfg):
    """Independent from-scratch runner for the gait-remediated ascending-stairs teacher."""

    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_teacher_v2_pyramid_stairs_remediation"


@configclass
class PyramidStairsTeacherV2RemediationV2PPORunnerCfg(V5FoundationPPORunnerCfg):
    """Independent runner for the phase-and-participation stair remediation."""

    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_teacher_v2_pyramid_stairs_remediation_v2"


@configclass
class InvertedPyramidStairsTeacherV2RemediationPPORunnerCfg(V5FoundationPPORunnerCfg):
    """Independent from-scratch runner for gait-safe inverted-stairs ascent."""

    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_teacher_v2_inverted_pyramid_stairs_remediation"


@configclass
class BoxesTeacherV2RemediationPPORunnerCfg(V5FoundationPPORunnerCfg):
    """Independent from-scratch runner for gait-safe edge-aware boxes locomotion."""

    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_teacher_v2_boxes_remediation"


@configclass
class PyramidStairsTeacherV3PPORunnerCfg(V5FoundationPPORunnerCfg):
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_teacher_v3_pyramid_stairs"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.num_mini_batches = 16


@configclass
class InvertedPyramidStairsTeacherV3PPORunnerCfg(V5FoundationPPORunnerCfg):
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_teacher_v3_inverted_pyramid_stairs"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.num_mini_batches = 16


@configclass
class BoxesTeacherV3PPORunnerCfg(V5FoundationPPORunnerCfg):
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_teacher_v3_boxes"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.num_mini_batches = 16


@configclass
class V5StairsSpecialistPPORunnerCfg(V5FoundationPPORunnerCfg):
    """Transfer-learning PPO for the first stairs specialist stage."""

    max_iterations = 4000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_v5_stairs_specialist"

    def __post_init__(self):
        super().__post_init__()
        # The specialist loads only policy and normalization weights from the
        # foundation checkpoint.  A fresh optimizer starts conservatively.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"


@configclass
class V5BoxesSpecialistPPORunnerCfg(V5FoundationPPORunnerCfg):
    """Transfer-learning PPO for the discrete-box obstacle specialist."""

    max_iterations = 4000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_v5_boxes_specialist"

    def __post_init__(self):
        super().__post_init__()
        # Transfer policy/GRU/normalizers from foundation3200, but train the
        # changed terrain objective with a fresh conservative optimizer.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"


@configclass
class V5BoxesMixedCommandRemediationPPORunnerCfg(V5FoundationPPORunnerCfg):
    """Conservative PPO for expanding boxes command coverage from model3999."""

    max_iterations = 2000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_v5_boxes_mixed_command_remediation"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "adaptive"


@configclass
class V5BoxesSafetyRemediationPPORunnerCfg(V5FoundationPPORunnerCfg):
    """Low-rate transfer stage for hard-box mixed-command safety."""

    max_iterations = 1000
    save_interval = 100
    experiment_name = "unitree_go2_velocity_v5_boxes_safety_remediation"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 2.5e-5
        self.algorithm.schedule = "adaptive"


@configclass
class V5StairsAscendingRemediationPPORunnerCfg(V5FoundationPPORunnerCfg):
    """Conservative transfer stage focused on ascending stair behavior."""

    max_iterations = 1200
    save_interval = 100
    experiment_name = "unitree_go2_velocity_v5_stairs_ascending_remediation"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "adaptive"
