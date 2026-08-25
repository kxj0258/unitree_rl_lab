# Go2 Rough Velocity 结果

## 实验定位

本阶段在已完成的 Go2 平地 baseline 之上增加 `Unitree-Go2-Velocity-Rough` 任务，验证 50% flat 与 50% random-uniform rough 混合地形上的速度跟踪和生存能力。

训练代码对应提交 `dea17cc`，分层评估器对应提交 `61843d8`。软件栈继续固定为 NVIDIA Driver 580.173.02、Python 3.11.16、PyTorch 2.7.0+cu128、Isaac Sim 5.1.0.0、Isaac Lab v2.3.2 和 RSL-RL 3.1.2；本阶段没有升级或重新安装这些组件。

## Rough 地形配置

- Terrain grid：10 rows × 20 columns；
- Sub-terrain：50% `MeshPlaneTerrainCfg`、50% `HfRandomUniformTerrainCfg`；
- Rough noise range：0.01–0.06 m；
- Noise step：0.01 m；
- Rough border width：0.25 m；
- Difficulty range：0.0–1.0；
- 训练期 terrain curriculum：启用；
- 初始最大 terrain level：1；
- Policy observation：保持 45 维，不加入 height scan。

该任务只包含 flat 和 random-uniform rough，不包含 stairs、boxes、slope 或离散障碍，因此不能概括为任意复杂地形泛化。

## 训练过程

训练前依次完成：

1. 256 environments × 100 iterations smoke；
2. 2048 environments × 100 iterations scale check；
3. 2048 environments × 500 iterations pilot；
4. 2048 environments × 3000 iterations formal run。

正式训练参数：

- Task：`Unitree-Go2-Velocity-Rough`；
- Environments：2048；
- Training seed：42；
- PPO iterations：3000；
- Simulation transitions：147,456,000；
- Training wall time：2:11:41；
- Exit code：0；
- Checkpoints：31；
- Console/TensorBoard 非有限值：0。

最后 500 iterations 的统计：

| Metric | Mean |
|---|---:|
| Reward | 25.8997 |
| Episode length | 992.86 / 1000 |
| XY tracking error | 0.2197 m/s |
| Yaw tracking error | 0.2255 rad/s |
| Timeout rate | 98.43% |
| Bad-orientation rate | 1.57% |
| Terrain level | 1.5206 |
| Command curriculum level | 1.0000 |
| Observation-noise curriculum level | 0.2387 |
| Throughput | 18,847.5 steps/s |

训练后期已进入平台区。正式 checkpoint 选择由独立评估决定，没有仅按最终 iteration 或训练 reward 选择，也没有证据支持直接续训到 5000 或 10000 iterations。

## Rough 分层评估协议

`scripts/project/evaluate_go2.py --terrain_protocol rough_stratified` 使用以下协议：

- 10 × 20 terrain grid，生成 seed 与评估场景 seed 一致；
- 精确 50% flat / 50% random_rough 环境分配；
- 两类地形均覆盖 level 0–9；
- flat 与 random_rough 使用配对的确定性 level 分配；
- 运行期 terrain/command curriculum 更新关闭；
- 确定性 inference policy；
- 每个 run：64 environments × 2 episodes = 128 episodes；
- 评估场景 seed：7、42、123。

比较对象为 flat baseline `model_2999` control，以及 rough `model_2400`、`model_2900`、`model_2999`。共 12 runs、1536 episodes。三个 seed 是评估场景 seed，不是三次独立训练。

## 三 seed 聚合结果

下表的连续指标为三个 run 的 mean ± sample standard deviation；termination rate 为 384 episodes 的聚合比例。

| Checkpoint | Reward ↑ | Overall XY error ↓ | Rough XY error ↓ | Overall yaw error ↓ | Action Δ L2 ↓ | Power proxy ↓ | Timeout ↑ | Bad orientation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat model_2999 control | 25.6747 ± 0.4542 | 0.1670 ± 0.0057 | 0.2532 ± 0.0076 | 0.1231 ± 0.0017 | 0.8367 ± 0.0145 | 72.00 ± 6.66 W | 77.34% | 22.66% |
| rough model_2400 | 31.6247 ± 0.3018 | 0.0993 ± 0.0019 | 0.1153 ± 0.0036 | 0.0721 ± 0.0002 | 0.8362 ± 0.0075 | 54.37 ± 1.12 W | 97.92% | 2.08% |
| **rough model_2900** | 31.8819 ± 0.1171 | **0.0950 ± 0.0008** | **0.1123 ± 0.0023** | 0.0702 ± 0.0005 | **0.8188 ± 0.0046** | **52.89 ± 0.54 W** | **99.22%** | **0.78%** |
| rough model_2999 | **31.9147 ± 0.1162** | 0.0957 ± 0.0022 | 0.1143 ± 0.0047 | **0.0696 ± 0.0015** | 0.8361 ± 0.0082 | 53.93 ± 1.15 W | **99.22%** | **0.78%** |

所有 1536 个 episode 的 base-contact termination 均为 0，所有数值字段均为有限值。

### 正式模型相对 flat control

在 random_rough subset 上，rough `model_2900` 相比 flat control：

- Mean reward 提升 78.0%；
- XY tracking error 降低 55.7%；
- Yaw tracking error 降低 57.5%；
- Action delta 降低 6.8%；
- Mechanical-power proxy 降低 40.7%；
- Timeout rate 从 55.73% 提升到 98.96%，增加 43.23 个百分点。

在 flat subset 上，rough `model_2900` 的 XY error 为 0.0777 m/s，flat control 为 0.0808 m/s；没有观察到 flat retention 退化。该比较只适用于相同 rough-stratified 评估协议，不能与原始 flat baseline 的 canonical 0.0756 m/s 指标直接混合。

## 正式 checkpoint

```text
logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000/model_2900.pt
SHA-256: 9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889
```

选择 `model_2900.pt` 的原因：

- Overall 与 random_rough XY tracking error 最低；
- 在 10 个 random_rough level 中有 7 个的 XY error 优于 `model_2999`；
- 与 `model_2999` 的 timeout/bad-orientation 相同；
- Action delta 和 mechanical-power proxy 更低；
- Overall 和 rough XY error 的跨 seed 波动更小。

`model_2999.pt` 保留为 reward/yaw 略优的 late-training alternative，但不作为正式 rough checkpoint。

正式模型覆盖 384 episodes：381 个达到 20 秒时限，0 个 base-contact termination，3 个 bad-orientation termination。

## 产物与验证

正式训练目录：

```text
logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000
```

正式评估目录：

```text
evaluation/rough_stratified_v1_2026-08-25
```

验证命令：

```bash
scripts/project/audit_go2_rough_evaluation.py \
  --verify-csv doc/repro/rough_eval_multiseed_2026-08-25.csv

scripts/project/archive_go2_rough.py --verify

sha256sum --check doc/repro/rough_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/rough_final_artifacts_2026-08-25.sha256
```

## 结论与边界

本阶段证明：针对 50/50 flat/random-uniform rough 混合地形训练的单一 seed-42 策略，相比 flat-only control 显著改善了相同 rough-stratified 协议下的跟踪、生存和功率代理指标，并保持 flat subset 表现。

尚未完成：多训练 seed、stairs/boxes/slopes、height-scan policy、reward/terrain 消融、rough 视频与策略导出、MuJoCo Sim2Sim 和实机部署。上述内容不得描述为已经实现。
