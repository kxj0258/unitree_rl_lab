# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Go2 Velocity Baseline Reproduction

This workspace includes a completed and reproducible Go2 flat-terrain velocity-locomotion baseline tested with the following fixed stack:

- NVIDIA Driver 580.173.02 Open Kernel Module
- Python 3.11.16
- PyTorch 2.7.0+cu128
- Isaac Sim 5.1.0.0
- Isaac Lab v2.3.2
- Unitree RL Lab commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`

Baseline experiment:

- Task: `Unitree-Go2-Velocity`
- 2048 parallel environments
- Seed 42
- 3000 PPO iterations
- 147,456,000 simulation steps
- 2:02:05 training time
- Approximately 20.3k steps/s during the final 500 iterations

Official checkpoint: `model_2999.pt` (`SHA-256: a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928`). It was selected by evaluating three candidate checkpoints under scenario seeds 7, 42, and 123: 64 environments × 2 episodes per seed, or 384 episodes per checkpoint.

Official-checkpoint evaluation results (mean ± standard deviation across the three scenario seeds):

| Metric | Result |
|---|---:|
| Mean episode reward | 34.6861 ± 0.0358 |
| Mean XY velocity error | 0.0756 ± 0.0021 m/s |
| XY velocity RMSE | 0.1210 ± 0.0016 m/s |
| Mean yaw velocity error | 0.0575 ± 0.0007 rad/s |
| Episodes reaching the 20 s time limit | 382 / 384 (99.48%) |
| Base-contact terminations | 0 / 384 |

These are multiple evaluation-scenario seeds for one policy trained with seed 42, not multiple independently trained seeds. The current baseline enables only the flat sub-terrain; complex terrain and training-seed ablations remain future work.

Project documentation:

- [Current project status](doc/PROJECT_STATUS.md)
- [Reproduction guide](doc/REPRODUCTION.md)
- [Baseline results](doc/RESULTS_BASELINE.md)
- [Resume and interview notes](doc/RESUME_PROJECT.md)
- [Evaluation artifact manifest](evaluation/README.md)

Reusable commands:

```bash
# Train with the recorded baseline defaults
scripts/project/train_go2_baseline.sh

# Analyze a TensorBoard run
scripts/project/analyze_tensorboard.py logs/rsl_rl/unitree_go2_velocity/<RUN_DIRECTORY>

# Evaluate a checkpoint without rendering
scripts/project/evaluate_go2.py \
  --task Unitree-Go2-Velocity \
  --num_envs 64 --seed 42 --episodes_per_env 2 \
  --checkpoint /absolute/path/to/model_2999.pt \
  --output_dir evaluation/eval_model2999_env64_seed42_ep2

# Record a single-environment follow video
scripts/project/record_go2_video.sh \
  /absolute/path/to/model_2999.pt \
  evaluation/model2999_follow \
  follow 1 42 1000
```

Robot asset paths are read from the `UNITREE_MODEL_DIR` and `UNITREE_ROS_DIR` environment variables.

## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif) |

</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Set `UNITREE_MODEL_DIR` to the downloaded model directory.

    ```bash
    export UNITREE_MODEL_DIR=/home/user/projects/unitree_model
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Set `UNITREE_ROS_DIR` to the downloaded ROS package directory.
    ```bash
    export UNITREE_ROS_DIR=/home/user/projects/unitree_ros/unitree_ros
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files



- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab
    ```
  - Running a task:

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
