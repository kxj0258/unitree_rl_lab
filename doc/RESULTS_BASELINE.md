# Go2 Velocity Baseline 结果

## 实验设置

- Task：`Unitree-Go2-Velocity`
- Environments：2048
- 训练 seed：42
- PPO iterations：3000
- Total timesteps：147,456,000
- Training time：2:02:05
- Exit code：0
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop
- Driver：580.173.02
- Isaac Sim：5.1.0.0
- Isaac Lab：v2.3.2

当前配置仅启用 flat sub-terrain，因此本实验定位为带动力学随机化、外力扰动和速度指令课程的平地 locomotion baseline，不能据此宣称复杂地形泛化能力。

## 训练结果

| Iterations | Mean reward | Episode length | XY error | Yaw error | Timeout | Bad orientation |
|---|---:|---:|---:|---:|---:|---:|
| 0–499 | 16.079 | 800.39 | 0.3150 | 0.4311 | 0.7320 | 0.2432 |
| 500–999 | 23.839 | 974.20 | 0.2622 | 0.2467 | 0.9448 | 0.0552 |
| 1000–1499 | 26.651 | 987.35 | 0.2199 | 0.2242 | 0.9714 | 0.0286 |
| 1500–1999 | 28.086 | 992.17 | 0.2071 | 0.2113 | 0.9826 | 0.0174 |
| 2000–2499 | 29.040 | 994.53 | 0.1993 | 0.2037 | 0.9880 | 0.0120 |
| 2500–2999 | 28.925 | 993.30 | 0.1985 | 0.2042 | 0.9844 | 0.0156 |

最后 500 iterations 的 mean reward 为 28.9248，mean episode length 为 993.30/1000，mean throughput 为 20,337 steps/s，TensorBoard 中无非有限值。训练后期已进入平台区，没有证据支持直接续训至 5000 iterations。

## 统一确定性评估

评估协议：

- 候选 checkpoint：`model_2400.pt`、`model_2900.pt`、`model_2999.pt`；
- 评估场景 seed：7、42、123；
- 每组 64 个并行环境，每环境收集 2 个完整 episode；
- 每个 checkpoint 共 384 个 episode，最长 20 秒/episode；
- 使用确定性 inference policy；
- 禁用评估期间的 terrain/command curriculum 更新；
- 保留训练配置中的随机初始状态、动力学随机化、周期外力扰动和随机速度指令。

这里的三个 seed 是同一个训练策略的评估场景 seed，不是三次独立训练 seed。

### 三 seed 聚合结果

以下为三个评估 seed 的 run-level mean ± sample standard deviation。

| Checkpoint | Reward ↑ | XY error (m/s) ↓ | XY RMSE (m/s) ↓ | Yaw error (rad/s) ↓ | Action Δ L2 ↓ | Mechanical power proxy (W) ↓ | Timeout ↑ | Bad orientation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| model_2400 | 34.4646 ± 0.1978 | 0.0802 ± 0.0028 | 0.1262 ± 0.0043 | 0.0587 ± 0.0001 | **0.7347 ± 0.0089** | 47.49 ± 1.87 | **99.74%** | **0.26%** |
| model_2900 | 34.6671 ± 0.2651 | 0.0813 ± 0.0045 | 0.1257 ± 0.0059 | **0.0561 ± 0.0006** | 0.7381 ± 0.0024 | **45.55 ± 0.65** | 99.22% | 0.78% |
| model_2999 | **34.6861 ± 0.0358** | **0.0756 ± 0.0021** | **0.1210 ± 0.0016** | 0.0575 ± 0.0007 | 0.7525 ± 0.0036 | 46.80 ± 0.97 | 99.48% | 0.52% |

三个 checkpoint 的 base-contact termination 均为 0%。机械功率指标是仿真中的 `sum(abs(torque * joint_velocity))` 代理值，不等同于真实电机或电池功耗。

## 正式 checkpoint

```text
logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt
SHA-256: a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928
```

选择理由：`model_2999.pt` 取得最高聚合 reward、最低 XY 平均误差与 RMSE，并且 reward 跨评估 seed 波动最小。相比 `model_2900.pt`，其机械功率代理和动作变化略高，但综合跟踪精度与跨场景一致性更适合作为正式展示模型。`model_2900.pt` 保留为低功率/yaw 跟踪备选。

在 384 个 episode 中，`model_2999.pt` 有 382 个达到 20 秒时限，无 base-contact termination，2 个因 bad orientation 提前终止。

## 正式展示与导出产物

- 单机器人 follow：`evaluation/video_model2999_seed42_follow_1000/videos/play/rl-video-step-0.mp4`
- 9 环境 grid：`evaluation/video_model2999_seed42_grid9_1000/videos/play/rl-video-step-0.mp4`
- TorchScript：`evaluation/video_model2999_seed42_follow_1000/exported/policy.pt`
- ONNX：`evaluation/video_model2999_seed42_follow_1000/exported/policy.onnx`

Grid 视频展示相互独立的向量化克隆环境。跨环境碰撞被隔离，视觉轨迹交叠不是多机器人避碰实验。

最终模型、导出文件、视频和九组评估 summary 的校验值保存在 `doc/repro/final_artifacts_2026-08-25.sha256`。

## 结论与边界

平地 Go2 velocity locomotion baseline 已完成训练、统一评估、模型选择、JIT/ONNX 导出和双视角视频验收。定量结果支持策略具备稳定存活、速度跟踪和外力扰动恢复能力。

尚未完成的部分包括多训练 seed、复杂地形、消融实验、MuJoCo Sim2Sim 和实机部署；这些内容不得写成已经实现。
