# Unitree Go2 RL Baseline：会话总结与新会话接管说明

更新时间：2026-08-25（Asia/Shanghai）

本文件是 2026-08-24 至 2026-08-25 会话的最终总结。新会话应优先读取本文件，再按需读取 `PROJECT_STATUS.md`、`RESULTS_BASELINE.md` 和两个仓库的 Git diff/log。旧文件 `SESSION_HANDOFF_2026-08-24.md` 保留完整过程记录，但其中“训练正在运行”的部分已经过时。

## 1. 协作方式与安全边界

默认协作方式：助手每次提供一批明确的终端命令，用户运行后发送完整输出；助手依据实际结果分析、总结并给下一步。只有用户明确说“你直接运行/实现”时，助手才直接修改或执行。

禁止事项：

- 不升级或重新安装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；
- 不覆盖、移动或删除现有 checkpoint、训练日志、GPU CSV、评估 CSV 和视频；
- 不把评估场景 seed 表述为多训练 seed；
- 不把平地 baseline 表述为复杂地形泛化；
- 不把 grid 视频表述为多机器人避碰；
- 不把机械功率代理表述为真实电池功耗。

## 2. 固定软件与仓库版本

- Ubuntu 22.04.5 LTS，kernel 6.8.0-138-generic
- NVIDIA Driver 580.173.02 Open Kernel Module
- Conda env：`/home/kxj/miniforge3/envs/unitree_rl`
- Python 3.11.16
- PyTorch 2.7.0+cu128
- Torchvision 0.22.0+cu128
- Isaac Sim 5.1.0.0
- Isaac Lab v2.3.2，基线 commit `37ddf626871758333d6ed89cf64ad702aef127d0`
- Unitree RL Lab 基线 commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`
- Unitree model commit `323e350252b9c3aee9c40acbfdad84f6ce46a5ac`
- RSL-RL 3.1.2

重要环境变量：

```text
ISAACLAB_PATH=/home/kxj/rl_ws/IsaacLab
UNITREE_RL_LAB_PATH=/home/kxj/rl_ws/unitree_rl_lab
UNITREE_MODEL_DIR=/home/kxj/rl_ws/unitree_model
LD_PRELOAD=/home/kxj/miniforge3/envs/unitree_rl/lib/libstdc++.so.6
```

## 3. 已解决的关键兼容问题

1. NVIDIA 595.x 使 Isaac Sim 5.1 在 RTX/Vulkan 初始化后退出码 139；固定到 580.173.02 Open 驱动后解决。
2. Conda/system C++ ABI 冲突通过在 `unitree_rl` 环境设置 Conda `libstdc++.so.6` 的 `LD_PRELOAD` 解决。
3. Isaac Sim 5.1 固定 FastAPI 0.115.7，与 Isaac Lab 2.3.2 的 Starlette 0.49.1 冲突；Isaac Lab 本地兼容补丁固定 Starlette 0.45.3。
4. Unitree 模型路径由源码占位符改成 `UNITREE_MODEL_DIR` / `UNITREE_ROS_DIR` 环境变量。
5. `play.py` 的 pretrained checkpoint import 更新为 Isaac Lab 2.3.2 所在的 `isaaclab_rl.utils` 路径。
6. `play.py` 增加显式 `--seed` 和 world/follow/grid 摄像模式；follow 保持机器人居中，grid 分离环境原点用于展示。

## 4. 正式训练已完成

- Task：`Unitree-Go2-Velocity`
- Environments：2048
- 训练 seed：42
- PPO iterations：3000
- Simulation transitions：147,456,000
- Training time：2:02:05
- 最后 500 iterations 平均吞吐：约 20,337 steps/s
- Exit code：0
- TensorBoard 非有限值：0

训练目录：

```text
/home/kxj/rl_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000
```

控制台日志与 GPU CSV：

```text
/home/kxj/rl_ws/unitree_go2_baseline_seed42.log
/home/kxj/rl_ws/gpu_baseline_seed42.csv
```

当前 Go2 任务只启用 flat sub-terrain；rough、stairs、boxes 等仍未启用。

## 5. 自动评估与正式 checkpoint

已实现 `scripts/project/evaluate_go2.py`：

- headless、确定性 inference；
- 显式评估 seed；
- 关闭评估期 terrain/command curriculum 更新；
- 输出 `metadata.json`、逐 episode `episodes.csv`、`summary.json`；
- 统计 reward、XY/yaw 误差、动作变化、机械功率代理与 termination 原因；
- 拒绝覆盖已有输出目录。

候选 `model_2400.pt`、`model_2900.pt`、`model_2999.pt` 分别使用评估场景 seed 7、42、123；每次 64 environments × 2 episodes，共 9 次运行、1152 个 episode。

正式 checkpoint：

```text
/home/kxj/rl_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt
SHA-256: a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928
```

`model_2999` 的三个评估场景 seed 聚合结果（384 episodes）：

- Mean reward：34.6861 ± 0.0358
- Mean XY velocity error：0.0756 ± 0.0021 m/s
- XY velocity RMSE：0.1210 ± 0.0016 m/s
- Mean yaw velocity error：0.0575 ± 0.0007 rad/s
- Timeout：382/384，99.48%
- Base contact：0/384
- Bad orientation：2/384，0.52%

选择原因：聚合 reward 最高、XY 平均误差/RMSE 最低、reward 跨评估场景波动最小。`model_2900` 保留为低机械功率/yaw 跟踪备选。

## 6. 视频与导出已完成

正式 follow 视频：

```text
evaluation/video_model2999_seed42_follow_1000/videos/play/rl-video-step-0.mp4
```

正式 9 环境 grid 视频：

```text
evaluation/video_model2999_seed42_grid9_1000/videos/play/rl-video-step-0.mp4
```

正式归档导出：

```text
evaluation/video_model2999_seed42_follow_1000/exported/policy.pt
evaluation/video_model2999_seed42_follow_1000/exported/policy.onnx
```

两种视角导出的 ONNX byte-for-byte 相同。两份 TorchScript 的模型张量一致，但自动生成类名与 serialization ID 不同，因此 hash 不同；归档时只指定 follow 版本。

Grid 中的机器人属于不同克隆环境，跨环境碰撞被隔离。视觉轨迹交叠是展示层现象，不是策略穿过同一物理场景中的其他机器人。

## 7. 已固化的工程资产

脚本：

- `scripts/project/train_go2_baseline.sh`
- `scripts/project/analyze_tensorboard.py`
- `scripts/project/evaluate_go2.py`
- `scripts/project/record_go2_video.sh`

文档：

- `README.md`
- `doc/PROJECT_STATUS.md`
- `doc/REPRODUCTION.md`
- `doc/RESULTS_BASELINE.md`
- `doc/RESUME_PROJECT.md`
- `evaluation/README.md`

复现快照：

- `doc/repro/revisions_2026-08-25.txt`
- `doc/repro/conda_explicit_2026-08-25.txt`
- `doc/repro/pip_freeze_2026-08-25.txt`
- `doc/repro/system_2026-08-25.txt`
- `doc/repro/OFFICIAL_BASELINE_CHECKPOINT.txt`
- `doc/repro/baseline_eval_multiseed_2026-08-25.csv`
- `doc/repro/final_artifacts_2026-08-25.sha256`

大型 `logs/` 与 `evaluation/` 产物默认不提交 Git，README 和小型追踪清单除外。最终清单中 14/14 文件已通过 SHA-256 检查。

## 8. 当前 Git 提交策略

本轮结束时应存在：

- Isaac Lab 本地分支 `compat/isaac-sim-5.1`：Starlette 兼容提交；
- Unitree RL Lab `main`：运行兼容与项目自动化提交；
- Unitree RL Lab `main`：baseline 文档、复现快照和本会话总结提交。

新助手应运行 `git log --oneline -5` 获取实际 commit hash，不要依赖本文件预写 hash。

## 9. 下一阶段建议

Baseline 已结束；截至本文件生成时没有训练任务需要继续等待。下一阶段建议：

1. 只读审计 Go2 terrain 配置，确认 flat 之外的 rough/stairs/boxes 配置和 curriculum 依赖；
2. 设计一个最小复杂地形 baseline 与一个明确的 curriculum 改进；
3. 先进行短 smoke test，再决定正式训练规模；
4. 正式实验至少补充多训练 seed，并使用现有 evaluator 做同协议对比；
5. 输出 baseline/改进版消融表、训练曲线和视频；
6. 时间允许时再做 ONNX/MuJoCo Sim2Sim。

如果用户在新会话中发送“训练结束后的完整输出”，应先判断它属于哪个新 run，再检查 exit code、checkpoint、episode termination、tracking error、吞吐和非有限值，不能把它与本 baseline run 混淆。

## 10. 新会话首次检查命令

```bash
conda activate unitree_rl
cd /home/kxj/rl_ws/unitree_rl_lab

sed -n '1,360p' doc/SESSION_SUMMARY_2026-08-25_BASELINE_COMPLETE.md
git log --oneline -5
git status --short
git -C /home/kxj/rl_ws/IsaacLab log --oneline -3
git -C /home/kxj/rl_ws/IsaacLab status --short
sha256sum --check doc/repro/final_artifacts_2026-08-25.sha256
```

如果用户同时提供新训练输出，新助手应在完成上述只读检查后直接分析输出，并继续采用“给一批命令 → 用户运行 → 返回完整结果 → 分析并指导下一步”的方式。
