# Go2 Flat Baseline 与 Rough 扩展复现说明

## 环境激活

```bash
conda activate unitree_rl
cd /home/kxj/rl_ws/unitree_rl_lab
```

所需环境变量：

```text
ISAACLAB_PATH=/home/kxj/rl_ws/IsaacLab
UNITREE_RL_LAB_PATH=/home/kxj/rl_ws/unitree_rl_lab
UNITREE_MODEL_DIR=/home/kxj/rl_ws/unitree_model
LD_PRELOAD=/home/kxj/miniforge3/envs/unitree_rl/lib/libstdc++.so.6
```

检查：

```bash
echo "$ISAACLAB_PATH"
echo "$UNITREE_RL_LAB_PATH"
echo "$UNITREE_MODEL_DIR"
echo "$LD_PRELOAD"
python -m pip check
```

固定版本、pip freeze、Conda explicit spec、系统信息和兼容补丁位于：

```text
doc/repro/
```

## 正式训练

默认参数：

- Task：`Unitree-Go2-Velocity`
- Environments：2048
- Seed：42
- PPO iterations：3000
- Logger：TensorBoard

运行：

```bash
powerprofilesctl set performance
scripts/project/train_go2_baseline.sh
```

覆盖默认参数的示例：

```bash
NUM_ENVS=1024 \
SEED=7 \
MAX_ITERATIONS=100 \
RUN_NAME=smoke_seed7_env1024_i100 \
scripts/project/train_go2_baseline.sh
```

脚本会为控制台日志添加时间戳，并拒绝覆盖已有日志。

## 分析 TensorBoard

```bash
scripts/project/analyze_tensorboard.py \
  logs/rsl_rl/unitree_go2_velocity/<RUN_DIRECTORY>
```

保存报告：

```bash
scripts/project/analyze_tensorboard.py \
  logs/rsl_rl/unitree_go2_velocity/<RUN_DIRECTORY> \
  | tee evaluation/<REPORT_NAME>.txt
```

正式 baseline 已生成：

```text
evaluation/baseline_training_analysis.txt
```

## Rough 训练

Rough 任务固定为 50% flat / 50% random-uniform rough。正式训练默认值为 2048 environments、seed 42、3000 iterations：

```bash
scripts/project/train_go2_rough.sh
```

脚本同时记录 console log 和 5 秒间隔的 GPU CSV，拒绝覆盖已有文件。运行名称默认包含 `roughmix50_curriculum`、stage、seed、environment 数和 iteration 数。

在正式训练前复现 pilot 或 scale check：

```bash
RUN_STAGE=scalecheck \
MAX_ITERATIONS=100 \
scripts/project/train_go2_rough.sh

RUN_STAGE=pilot \
MAX_ITERATIONS=500 \
scripts/project/train_go2_rough.sh
```

`RUN_NAME`、`CONSOLE_LOG`、`GPU_LOG`、`NUM_ENVS`、`SEED`、`MAX_ITERATIONS` 和 `DEVICE` 均可通过环境变量覆盖。输出名称必须是新的；不要指向已存在的正式日志。

## GUI 推理

```bash
./unitree_rl_lab.sh --play \
  --task Unitree-Go2-Velocity \
  --num_envs 1 \
  --seed 42 \
  --checkpoint logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt \
  --real-time
```

关闭 Isaac Sim 窗口即可结束。

## Headless 确定性评估

输出目录必须不存在。以下命令收集 64 × 2 = 128 个 episode，并生成 `metadata.json`、`episodes.csv` 和 `summary.json`：

```bash
scripts/project/evaluate_go2.py \
  --task Unitree-Go2-Velocity \
  --num_envs 64 \
  --seed 42 \
  --episodes_per_env 2 \
  --checkpoint logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt \
  --output_dir evaluation/eval_model2999_env64_seed42_ep2
```

正式 checkpoint 选择比较了 `model_2400.pt`、`model_2900.pt`、`model_2999.pt` 和评估场景 seed 7、42、123，共 9 次运行、1152 个 episode。评估 seed 只改变随机命令、扰动和初始状态，不代表重新训练策略。

## Rough 分层评估

先用 dry-run 核对固定的 12-run 矩阵和 checkpoint 路径：

```bash
DRY_RUN=1 scripts/project/evaluate_go2_rough_matrix.sh
```

正式运行必须指定一个不存在的输出根目录：

```bash
OUTPUT_ROOT=evaluation/rough_stratified_v1_rerun_2026-08-26 \
scripts/project/evaluate_go2_rough_matrix.sh
```

矩阵比较 flat `model_2999` control 与 rough `model_2400/2900/2999`，默认场景 seeds 7/42/123、64 environments × 2 episodes。完整 evaluator 输出进入各 run 目录，console 输出单独保存为 `.console.log`。

单独评估一个 checkpoint：

```bash
scripts/project/evaluate_go2.py \
  --task Unitree-Go2-Velocity-Rough \
  --terrain_protocol rough_stratified \
  --num_envs 64 \
  --episodes_per_env 2 \
  --seed 42 \
  --checkpoint /absolute/path/to/model_2900.pt \
  --output_dir evaluation/new_rough_model2900_seed42_env64_ep2 \
  --device cuda:0
```

`rough_stratified` 强制 10 × 20 grid、精确 50/50 terrain type 分配和 level 0–9 覆盖，并关闭运行期 terrain/command curriculum 更新。

## Rough 评估审计与归档

审计现有 12-run、1536 episodes，并验证已归档聚合 CSV：

```bash
scripts/project/audit_go2_rough_evaluation.py \
  --verify-csv doc/repro/rough_eval_multiseed_2026-08-25.csv
```

审计检查 checkpoint hash、run/episode 数、terrain type/level、assignment 不变性、非有限值、console fatal marker 和 termination，并输出 overall/type/level 聚合。

验证 official marker、候选 checkpoint inventory、聚合 CSV、48-file evaluation manifest 和 13-file formal manifest：

```bash
scripts/project/archive_go2_rough.py --verify
```

在全新归档位置创建文件时使用 `--create --output-dir <NEW_DIRECTORY>`；脚本拒绝覆盖任意已有目标。

## Rough 多训练 Seed 评估与归档

对一个新的 formal training seed 预览 9-run checkpoint 矩阵：

```bash
TRAINING_SEED=<TRAINING_SEED> \
ROUGH_RUN=/absolute/path/to/formal_training_run \
DRY_RUN=1 \
scripts/project/evaluate_go2_rough_training_seed_matrix.sh
```

正式执行时必须提供一个不存在的 `OUTPUT_ROOT`。输出名称同时编码 training seed 与 evaluation scenario seed，避免将二者混淆。

验证已完成的三个 training seeds、27 rough runs 和 3456 episodes：

```bash
scripts/project/audit_go2_rough_multitraining.py \
  --verify-selected-csv doc/repro/rough_multitraining_selected_official_2026-08-25.csv \
  --verify-fixed-csv doc/repro/rough_multitraining_fixed_checkpoints_2026-08-25.csv

scripts/project/archive_go2_rough_multitraining.py --verify
sha256sum --check doc/repro/rough_multitraining_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/rough_multitraining_final_artifacts_2026-08-25.sha256
```

旧 seed-42 rough 归档与新的多 training-seed 归档是追加关系；不要覆盖或删除任意旧 manifest。

## 录制 follow 视频

输出目录必须不存在：

```bash
scripts/project/record_go2_video.sh \
  logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt \
  evaluation/video_model2999_seed42_follow_1000 \
  follow \
  1 \
  42 \
  1000
```

## 录制多环境 grid 视频

```bash
scripts/project/record_go2_video.sh \
  logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt \
  evaluation/video_model2999_seed42_grid9_1000 \
  grid \
  9 \
  42 \
  1000
```

注意：grid 中每个机器人属于独立的克隆环境，跨环境碰撞被隔离。视觉轨迹交叠不是多机器人碰撞仿真。

## 训练产物

正式训练目录：

```text
logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000
```

正式 checkpoint：

```text
logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt
SHA-256: a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928
```

正式控制台日志：

```text
/home/kxj/rl_ws/unitree_go2_baseline_seed42.log
```

GPU 监控：

```text
/home/kxj/rl_ws/gpu_baseline_seed42.csv
```

不要覆盖或删除上述目录和文件。

Rough 正式训练目录：

```text
logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000
```

Rough 正式 checkpoint：

```text
logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000/model_2900.pt
SHA-256: 9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889
```

Rough 正式 console 与 GPU CSV：

```text
/home/kxj/rl_ws/unitree_go2_roughmix50_curriculum_formal_seed42_env2048_i3000_2026-08-25_04-40-14.log
/home/kxj/rl_ws/gpu_roughmix50_curriculum_formal_seed42_env2048_i3000_2026-08-25_04-40-14.csv
```

## 最终产物验证

在项目根目录执行：

```bash
sha256sum --check doc/repro/final_artifacts_2026-08-25.sha256
sha256sum --check doc/repro/rough_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/rough_final_artifacts_2026-08-25.sha256
sha256sum --check doc/repro/rough_multitraining_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/rough_multitraining_final_artifacts_2026-08-25.sha256
```

大型模型、视频和评估输出位于本地 `logs/`、`evaluation/`，默认不提交 Git；校验清单和聚合结果保存在 `doc/repro/`。

## 兼容补丁

当前补丁快照：

```text
doc/repro/isaaclab_compat_2026-08-25.patch
doc/repro/unitree_rl_lab_compat_2026-08-25.patch
```

这些补丁与固定版本配套。升级依赖前必须重新验证是否仍需要。
