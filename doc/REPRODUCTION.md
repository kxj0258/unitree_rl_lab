# Go2 Baseline 复现说明

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

## 最终产物验证

在项目根目录执行：

```bash
sha256sum --check doc/repro/final_artifacts_2026-08-25.sha256
```

大型模型、视频和评估输出位于本地 `logs/`、`evaluation/`，默认不提交 Git；校验清单和聚合结果保存在 `doc/repro/`。

## 兼容补丁

当前补丁快照：

```text
doc/repro/isaaclab_compat_2026-08-25.patch
doc/repro/unitree_rl_lab_compat_2026-08-25.patch
```

这些补丁与固定版本配套。升级依赖前必须重新验证是否仍需要。
