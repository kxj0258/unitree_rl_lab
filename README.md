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
- Unitree RL Lab upstream baseline commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`
- Go2 rough task commit `dea17cc`; stratified evaluator commit `61843d8`

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

These are multiple evaluation-scenario seeds for one policy trained with seed 42, not multiple independently trained seeds. This canonical baseline enables only the flat sub-terrain; its metrics remain separate from the rough-stratified experiment below.

## Go2 Rough-Terrain Extension

The completed rough stage adds `Unitree-Go2-Velocity-Rough` with a 50/50 mix of flat and random-uniform rough terrain. It keeps the 45-dimensional policy observation unchanged and trains terrain curriculum from scratch with the same 2,048-environment, seed-42, 3,000-iteration budget.

Formal rough experiment:

- 10 × 20 terrain grid with levels 0–9
- 50% flat and 50% random-uniform rough terrain (0.01–0.06 m noise)
- 147,456,000 simulation transitions
- 2:11:41 training time
- 12-run checkpoint matrix: flat control plus three rough checkpoints, scenario seeds 7/42/123
- 1,536 audited evaluation episodes with zero non-finite values and zero base-contact terminations

Official rough checkpoint: `model_2900.pt` (`SHA-256: 9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889`). Across its 384 formal episodes it achieved 0.0950 ± 0.0008 m/s overall XY error, 0.1123 ± 0.0023 m/s random-rough XY error, and a 99.22% timeout rate. Against the flat-trained control on the same random-rough subset, it reduced XY error by 55.7% and increased timeout rate by 43.23 percentage points.

The follow-up multi-training-seed stage completed the same formal budget for training seeds 42, 7, and 123. The fixed `model_2999` comparison across those three seeds achieved 0.0956 ± 0.0009 m/s overall XY error, 0.1131 ± 0.0013 m/s random-rough XY error, and 1144/1152 timeouts. The uncertainty is sample standard deviation across independent training seeds. This still covers only flat/random-uniform rough terrain; it does not establish stairs, boxes, slopes, general complex-terrain, Sim2Sim, or real-robot performance.

## Go2 Seven-Type MultiTerrain Stage

The current stage adds `Unitree-Go2-Velocity-MultiTerrain`, covering flat, random rough, positive/inverted slopes, positive/inverted stairs, and boxes with terrain levels 0--9. A fixed `multiterrain_stratified` protocol evaluates all seven types and ten levels with 140 episodes per checkpoint.

Starting from training-seed-42 `model_2995`, two controlled 500-iteration continuations were compared: the original survival curriculum and a tracking-aware curriculum variant. The selected survival `model_3300.pt` achieved 0.270 m/s XY error and an 83.6% timeout rate at evaluation scenario seed 42. Across scenario seeds 7/42/123 it achieved 0.272 ± 0.010 m/s XY error and a 79.3% ± 4.3 percentage-point timeout rate. The tracking-aware branch did not improve the combined tracking/survival trade-off and is retained as a documented negative result.

Two audited dynamic portfolio videos are available locally under Git-ignored `evaluation/`:

- seven-terrain MultiTerrain video: 141.86 s, SHA-256 `5b64a6722651b33977048453c57f20017c39754985c45ad13a41ccbbe9f93939`;
- three-training-seed Rough video: 61.94 s, SHA-256 `3114114451694c63535542d79340dfb7e4328e89189622e018ab45ed51972b42`.

The MultiTerrain policy tracks well on flat terrain but slows substantially on stairs, boxes, and inverted terrain. It demonstrates broad coverage and a reproducible evaluation workflow, not solved high-speed traversal on every terrain.

Project documentation:

- [Current project status](doc/PROJECT_STATUS.md)
- [Reproduction guide](doc/REPRODUCTION.md)
- [Baseline results](doc/RESULTS_BASELINE.md)
- [Rough-terrain results](doc/RESULTS_ROUGH.md)
- [Rough multi-training-seed results](doc/RESULTS_ROUGH_MULTITRAINING.md)
- [MultiTerrain continuation and dynamic demo results](doc/RESULTS_MULTITERRAIN_CONTINUATION_2026-08-26.md)
- [Command-to-script reproducibility audit](doc/COMMAND_SCRIPT_AUDIT_2026-08-26.md)
- [Resume and interview notes](doc/RESUME_PROJECT.md)
- [Latest MultiTerrain session handoff](doc/SESSION_HANDOFF_2026-08-25_MULTITERRAIN_PROTOTYPE.md)
- [Evaluation artifact manifest](evaluation/README.md)

Reusable commands:

```bash
# Train with the recorded baseline defaults
scripts/project/train_go2_baseline.sh

# Train the recorded 50/50 rough task defaults
scripts/project/train_go2_rough.sh

# Preview the exact archived MultiTerrain Stage-2 workflow (safe dry-run)
OUTPUT_ROOT=evaluation/new_stage2_eval \
scripts/project/reproduce_go2_multiterrain_stage2.sh baseline-eval

# Verify selected Stage-2 checkpoints, summaries, and videos
EXECUTE=1 scripts/project/reproduce_go2_multiterrain_stage2.sh verify-archive

# Analyze a TensorBoard run
scripts/project/analyze_tensorboard.py logs/rsl_rl/unitree_go2_velocity/<RUN_DIRECTORY>

# Evaluate a checkpoint without rendering
scripts/project/evaluate_go2.py \
  --task Unitree-Go2-Velocity \
  --num_envs 64 --seed 42 --episodes_per_env 2 \
  --checkpoint /absolute/path/to/model_2999.pt \
  --output_dir evaluation/eval_model2999_env64_seed42_ep2

# Preview or execute the formal rough checkpoint matrix
DRY_RUN=1 scripts/project/evaluate_go2_rough_matrix.sh

# Audit and verify the completed rough evaluation archive
scripts/project/audit_go2_rough_evaluation.py \
  --verify-csv doc/repro/rough_eval_multiseed_2026-08-25.csv
scripts/project/archive_go2_rough.py --verify

# Audit and verify the multi-training-seed rough archive
scripts/project/audit_go2_rough_multitraining.py \
  --verify-selected-csv doc/repro/rough_multitraining_selected_official_2026-08-25.csv \
  --verify-fixed-csv doc/repro/rough_multitraining_fixed_checkpoints_2026-08-25.csv
scripts/project/archive_go2_rough_multitraining.py --verify

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
