# Go2 Rough Velocity 多训练 Seed 结果

## 实验定位

本阶段在已归档的 training seed 42 rough 实验之上，使用完全相同的任务、训练预算和评估协议补充 training seeds 7 与 123，从而区分训练随机性与评估场景随机性。

软件栈保持固定：NVIDIA Driver 580.173.02、Python 3.11.16、PyTorch 2.7.0+cu128、Isaac Sim 5.1.0.0、Isaac Lab v2.3.2 和 RSL-RL 3.1.2。本阶段没有升级或重新安装这些组件。

## 训练矩阵

三个 formal run 均使用：

- Task：`Unitree-Go2-Velocity-Rough`；
- 2048 environments；
- 3000 PPO iterations；
- 每个 training seed 147,456,000 simulation transitions；
- 50% flat / 50% random-uniform rough；
- terrain curriculum 启用；
- 45 维 policy observation，不加入 height scan。

| Training seed | Wall time | Exit | Checkpoints | 后期吞吐 | Non-finite |
|---:|---:|---:|---:|---:|---:|
| 42 | 2:11:41 | 0 | 31 | 约 18.8k steps/s | 0 |
| 7 | 2:14:55 | 0 | 31 | 约 18.5k steps/s | 0 |
| 123 | 1:53:56 | 0 | 31 | 约 22.0k steps/s | 0 |

三个 run 共 442,368,000 simulation transitions，formal wall time 合计 6:20:32。

## 统一评估协议

所有 checkpoint 均使用 `rough_stratified v1`：

- evaluation scenario seeds：7、42、123；
- 64 environments × 2 episodes = 128 episodes/run；
- 精确 50% flat / 50% random_rough；
- 两类均覆盖 terrain level 0–9；
- terrain 与 command curriculum 的运行期更新关闭；
- 确定性 inference policy。

每个 training seed 评估 `model_2400/2900/2999`，因此 rough 固定 checkpoint 矩阵共有 27 runs、3456 episodes。原 seed-42 矩阵另含 3 个 flat control runs；三个评估目录合计 120 个文件。

所有 27 个 rough runs exit 0，episode 数、checkpoint hash、terrain type/level、环境 assignment、termination 和 console marker 审计通过；非有限值和 base-contact termination 均为 0。

## Per-seed selected official

| Training seed | Selected checkpoint | SHA-256 |
|---:|---|---|
| 42 | `model_2900.pt` | `9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889` |
| 7 | `model_2999.pt` | `353ea7a9d1cfa3e35fdc8678901ede952c54e7f4fb54782ad86fd2f02352f4af` |
| 123 | `model_2999.pt` | `d83a237bbf9918b3f81d269b1ce4684659756ee30c902b384f92d9dfced0945c` |

跨 training seed 的 selected-official 结果（mean ± sample SD）：

| Metric | Overall | Random rough |
|---|---:|---:|
| Reward ↑ | 32.0299 ± 0.1508 | 30.5034 ± 0.1521 |
| XY error ↓ | 0.0953 ± 0.0010 m/s | 0.1124 ± 0.0008 m/s |
| Yaw error ↓ | 0.0706 ± 0.0005 rad/s | 0.0802 ± 0.0005 rad/s |
| Action Δ L2 ↓ | 0.8207 ± 0.0079 | 0.8348 ± 0.0073 |
| Power proxy ↓ | 53.57 ± 1.01 W | 56.42 ± 0.89 W |
| Timeout ↑ | 1144/1152（99.31%） | 570/576（98.96%） |
| Base contact ↓ | 0/1152 | 0/576 |
| Bad orientation ↓ | 8/1152（0.69%） | 6/576（1.04%） |

## Fixed-iteration 比较

为避免仅报告按同一评估矩阵选择后的 checkpoint，另外按固定 iteration 跨三个 training seeds 聚合。

| Checkpoint | Reward ↑ | Overall XY ↓ | Rough XY ↓ | Overall yaw ↓ | Action Δ ↓ | Power ↓ | Timeout ↑ | Bad orientation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `model_2400` | 31.7338 | 0.0973 | 0.1138 | 0.0711 | 0.8371 | 55.04 W | 98.35% | 1.65% |
| `model_2900` | 31.8066 | **0.0946** | **0.1125** | 0.0713 | 0.8296 | 54.41 W | 98.26% | 1.74% |
| **`model_2999`** | **32.0408** | 0.0956 | 0.1131 | **0.0704** | **0.8264** | **53.92 W** | **99.31%** | **0.69%** |

`model_2900` 的 overall XY error 比 `model_2999` 低约 1.0%，rough XY error 低约 0.5%；但固定 `model_2999` 在 reward、yaw、action delta、power、timeout、bad orientation 及跨 seed 稳定性上总体占优。因此：

- 历史 seed-42 official `model_2900` 保持不变；
- seed-7 和 seed-123 official 为各自的 `model_2999`；
- 跨 training-seed 的固定 iteration 主结果推荐 `model_2999`。

固定 `model_2999` 的主要跨 training-seed 指标（mean ± sample SD）：

- Overall reward：`32.0408 ± 0.1351`；
- Overall XY error：`0.0956 ± 0.0009 m/s`；
- Random-rough XY error：`0.1131 ± 0.0013 m/s`；
- Overall yaw error：`0.0704 ± 0.0008 rad/s`；
- Timeout：`1144/1152`（99.31%）；
- Base contact：`0/1152`；
- Bad orientation：`8/1152`（0.69%）。

这里的 training-seed SD 只基于 3 个独立训练 seed；evaluation scenario seed 不计作独立训练样本。

## 复现与验证

```bash
# 对一个新 training seed 预览 9-run 矩阵
TRAINING_SEED=<seed> \
ROUGH_RUN=/absolute/path/to/formal_run \
DRY_RUN=1 \
scripts/project/evaluate_go2_rough_training_seed_matrix.sh

# 审计现有 27-run、3456-episode 矩阵并验证聚合 CSV
scripts/project/audit_go2_rough_multitraining.py \
  --verify-selected-csv doc/repro/rough_multitraining_selected_official_2026-08-25.csv \
  --verify-fixed-csv doc/repro/rough_multitraining_fixed_checkpoints_2026-08-25.csv

# 验证新的六文件归档
scripts/project/archive_go2_rough_multitraining.py --verify
sha256sum --check doc/repro/rough_multitraining_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/rough_multitraining_final_artifacts_2026-08-25.sha256
```

旧 seed-42 归档仍由 `scripts/project/archive_go2_rough.py --verify` 独立验证。

## 项目边界

本阶段完成的是三个 training seeds 下的 50/50 flat/random-uniform rough 复现与方差分析。尚未完成 stairs、boxes、slopes、height-scan policy、terrain/reward/扰动/观测噪声消融、rough 正式视频与导出、MuJoCo Sim2Sim、Sim2Real 或实机部署。
