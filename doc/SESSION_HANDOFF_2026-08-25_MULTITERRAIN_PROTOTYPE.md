# Unitree Go2 MultiTerrain 原型会话接管总结

更新时间：2026-08-25（Asia/Shanghai）

本文件承接 `SESSION_HANDOFF_2026-08-25_ROUGH_MULTITRAINING_FINAL.md`。本次会话在不修改既有 flat/rough 正式归档的前提下，完成了 Go2 challenging MultiTerrain 原型训练、固定分层评估和策略导出。下一会话必须先完整读取两个接管文件，并先做只读检查；不要重新运行已经完成的训练或评估。

## 1. 固定基础与安全边界

- Unitree RL Lab 接管起点：`5dd4dfd85dc8e4b22d84a7e9b8be8ccad5f1dd6b`；
- MultiTerrain training task：`663c240`（`feat: add Go2 challenging multiterrain training task`）；
- MultiTerrain stratified evaluation：`b005852`（`feat: add stratified Go2 multiterrain evaluation`）；
- MultiTerrain prototype artifact archive：`e2229c0`（`docs: archive Go2 multiterrain prototype artifacts`）；
- rough 多训练 seed 正式归档：`8079665fa05a77986c63cc5d6599dd3464c16377`；
- IsaacLab HEAD：`39b494e`，v2.3.2 基线 `37ddf62`；
- Python 3.11.16，PyTorch 2.7.0+cu128，Isaac Sim 5.1.0.0，RSL-RL 3.1.2；
- 禁止升级或重新安装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；
- 禁止覆盖、移动或删除已有 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果和视频。

既有 flat baseline、rough training seeds 42/7/123、official checkpoint、`rough_stratified v1` 和正式归档仍以旧最终接管文件为准，本次没有改动。

## 2. 本次目标与最终路线

用户要求按加速后的 7 天路线先跑通全流程：

1. slopes、stairs、boxes 和多地形组合；
2. 一个 unified policy 的训练；
3. 固定分层验证和初版 Demo checkpoint；
4. TorchScript/ONNX 导出；
5. MuJoCo Sim2Sim；
6. 最后再集中完善 Demo、多 training-seed 正式实验和受控消融。

当前已完成第 1–4 项的原型闭环。尚未安装或克隆 MuJoCo/SDK2，也尚未制作正式视频。

## 3. 新增 MultiTerrain 任务

注册 task：`Unitree-Go2-Velocity-MultiTerrain`。

配置标识：`multiterrain_v3_challenging_noheightscan`。保持 45 维 policy observation，无 height scan；沿用原 Go2 reward、observation noise、termination、domain randomization 和 PPO 配置。

地形网格为 10 rows × 20 columns，比例与最高难度：

| Terrain type | Proportion | Columns | Range |
|---|---:|---:|---|
| flat | 10% | 2 | plane |
| random_rough | 20% | 4 | height noise 0.01–0.08 m |
| pyramid_slope | 10% | 2 | 0.05–0.45 rad |
| inverted_pyramid_slope | 10% | 2 | 0.05–0.45 rad |
| pyramid_stairs | 15% | 3 | step height 0.05–0.20 m |
| inverted_pyramid_stairs | 15% | 3 | step height 0.05–0.20 m |
| boxes | 20% | 4 | grid height 0.04–0.16 m |

初始 linear command range 已由继承的 ±0.1 m/s 改为：

- `lin_vel_x=(-0.5, 0.5)`；
- `lin_vel_y=(-0.2, 0.2)`。

## 4. Curriculum 诊断与 v3 修复

原 IsaacLab `terrain_levels_vel` 要求机器人离 terrain origin 超过 4 m 才升级。早期 ±0.1 m/s 命令和会重采样/反向的命令与净位移判据不匹配，导致 v1/v2 训练在低难度学会存活后仍锁死 level 0。

本次新增只作用于 MultiTerrain 的 `multiterrain_levels_survival`：

- episode 到 timeout：level +1；
- episode 前 50% 失败：level -1；
- 50%–100% 之间失败：保持 level；
- 超过最高 level 后沿用 IsaacLab 行为，随机回到 0–8。

原 flat/rough task 继续使用旧 curriculum，没有被更改。

## 5. Training seed 42 原型训练链

所有以下 run 都是同一个 **training seed 42** policy 的连续训练，不是独立 training seeds；每次 resume 会重新初始化环境 terrain 分布，但恢复 policy/optimizer iteration。

### Smoke：64 env × 5 iterations

Run：

```text
logs/rsl_rl/unitree_go2_velocity_multiterrain/
2026-08-25_20-46-07_multiterrain_v1_challenging_noheightscan_smoke_seed42_env64_i5
```

- exit 0；`model_0.pt`、`model_4.pt`；
- TensorBoard 175 values，non-finite 0；
- 只验证 task、七类地形和训练链路。

### Pilot：1024 env × 500 iterations

Run：

```text
2026-08-25_20-52-08_multiterrain_v1_challenging_noheightscan_pilot_seed42_env1024_i500
```

- 12,288,000 transitions，28:19，exit 0；
- final `model_499.pt`；
- reward 27.22，episode length 994.74，timeout 97.29%；
- terrain level 0，command level 0.1，暴露 curriculum 锁死。

### v1 resume：追加 1500，到约 iteration 1999

Run：

```text
2026-08-25_21-27-52_multiterrain_v1_challenging_noheightscan_prototype_resume_from499_seed42_env1024_addi1500_total2000
```

- 从 `model_499.pt` 恢复；新增 36,864,000 transitions，53:04，exit 0；
- final `model_1998.pt`；
- reward 32.27，timeout 98.69%，yaw error 0.2313；
- terrain level仍为 0，command 0.1。

### v2 command fix：追加 200，到约 iteration 2198

Run：

```text
2026-08-25_22-30-26_multiterrain_v2_challenging_noheightscan_curriculumfix_resume_from1998_seed42_env1024_addi200
```

- 从 `model_1998.pt` 恢复；新增 4,915,200 transitions，7:06，exit 0；
- final `model_2197.pt`；
- command level 0.5 生效，但 terrain level仍降到 0.002；
- 证明仅扩大命令不足以解决问题。

### v3 survival curriculum：追加 200，到约 iteration 2397

Run：

```text
2026-08-25_22-41-35_multiterrain_v3_challenging_noheightscan_survivalcurriculum_resume_from2197_seed42_env1024_addi200
```

- 从 `model_2197.pt` 恢复；新增 4,915,200 transitions，7:01，exit 0；
- final `model_2396.pt`；
- terrain level 0.01 → 4.25，command 0.5 → 0.6；
- v3 curriculum 被确认有效。

### v3 prototype：追加 600，到约 iteration 2996

Run：

```text
logs/rsl_rl/unitree_go2_velocity_multiterrain/
2026-08-25_22-53-12_multiterrain_v3_challenging_noheightscan_prototype_resume_from2396_seed42_env1024_addi600_totalapprox3000
```

- 从 `model_2396.pt` 恢复；新增 14,745,600 transitions，21:28，exit 0；
- checkpoints：`model_2400/2500/2600/2700/2800/2900/2995.pt`；
- final terrain level 5.77，command level 0.6；
- final reward 12.40，XY 0.2816，yaw 0.3188；
- final timeout 73.80%，base contact 0.39%，bad orientation 25.91%；
- TensorBoard 21,000 values，non-finite 0；GPU telemetry正常。

训练末尾统计受动态 terrain level 分布影响，不能直接选择 checkpoint。

## 6. `multiterrain_stratified` smoke v0

新协议：

- evaluation scenario seed：42（不是 training seed）；
- 70 env × 2 episodes = 140 episodes/run；
- 七类 terrain 各 10 env，每类精确覆盖 level 0–9；
- 每类 20 episodes，每 level 14 episodes，共 70 个 type×level cells；
- deterministic policy；
- runtime terrain/command curriculum关闭；
- 固定 terrain type/column/level，并审计 episode 过程中不变化。

目录：

```text
evaluation/terrain_generalization_smoke_v0_2026-08-25
```

已评估 training seed 42 的 `model_2600/2900/2995`，全部 exit 0、140 episodes、non-finite 0、base contact 0。

总体结果：

| Checkpoint | Reward | XY error | Yaw error | Timeout | Bad orientation |
|---|---:|---:|---:|---:|---:|
| model_2600 | 13.7961 | 0.3200 | 0.2014 | 67.14% | 32.86% |
| model_2900 | 15.7300 | 0.3079 | 0.1812 | 75.00% | 25.00% |
| model_2995 | **16.1924** | **0.2918** | **0.1646** | **77.14%** | **22.86% |

`model_2995` per-terrain timeout：

- flat 100%；
- inverted slope 85%；
- slope 80%；
- inverted stairs 100%；
- random rough 75%；
- boxes 65%；
- pyramid stairs 35%。

结论：`model_2995.pt` 是当前 provisional Demo/export checkpoint。正向上楼梯是无 height scan 策略的主要短板，不应把当前结果描述为正式多-seed结论。

注意：首次 `model_2600` 评估时 `tee` 因父目录尚未创建而没有保存 `.console.log`；评估器自身 exit 0，`episodes.csv/metadata.json/summary.json` 完整且已审计。`model_2900/2995` console均已保存。不要为了补这个非关键console而重复评估。

## 7. 策略导出

Checkpoint：

```text
logs/rsl_rl/unitree_go2_velocity_multiterrain/
2026-08-25_22-53-12_multiterrain_v3_challenging_noheightscan_prototype_resume_from2396_seed42_env1024_addi600_totalapprox3000/model_2995.pt
```

SHA-256：

```text
model_2995.pt  67da571685f42e56df412730cd6c13c335a63aa85a07beaf7f97b35cfb3dd2da
policy.pt      2acbed11b1ff6648d8755d576bd9bc42bffc35d283d2cb59ae3dcee852dfb304
policy.onnx    0e13dd581d4606d52bb8642eeaa68ce5d2e81504bf287e2609aa8485682a4f4a
```

导出目录：

```text
.../2026-08-25_22-53-12_.../exported/policy.pt
.../2026-08-25_22-53-12_.../exported/policy.onnx
```

- TorchScript 769,426 bytes；ONNX 758,549 bytes；
- ONNX checker PASS；输入 `obs [batch,45]`，输出 `actions [batch,12]`；
- 8 组随机 observation：最大绝对差 `2.3841858e-6`，平均差 `5.6271e-7`；
- TorchScript/ONNX `allclose(rtol=1e-5, atol=1e-5)` PASS；
- 导出 exit 0，无 fatal/non-finite；
- 另生成 2-step 占位视频，不是正式 Demo。

## 8. 已提交代码与归档

本次功能与结果已分三批提交：

```text
663c240 feat: add Go2 challenging multiterrain training task
b005852 feat: add stratified Go2 multiterrain evaluation
e2229c0 docs: archive Go2 multiterrain prototype artifacts
```

归档新增：

```text
doc/RESULTS_MULTITERRAIN_PROTOTYPE.md
doc/repro/OFFICIAL_MULTITERRAIN_PROVISIONAL_CHECKPOINT.txt
doc/repro/multiterrain_checkpoint_inventory_2026-08-25.txt
doc/repro/multiterrain_checkpoint_sha256_2026-08-25.txt
doc/repro/multiterrain_evaluation_inventory_2026-08-25.txt
doc/repro/multiterrain_evaluation_sha256_2026-08-25.txt
doc/repro/multiterrain_export_inventory_2026-08-25.txt
doc/repro/multiterrain_final_artifacts_2026-08-25.sha256
```

清单覆盖 37 个 checkpoint、11 个 evaluation 文件、2 个导出文件，并以最终清单固定 provisional checkpoint、3 个 summary、导出、占位视频、导出日志和归档文本。大文件仍保留在 Git 忽略的原始位置，没有复制、覆盖、移动或删除。

提交前后均通过 `git diff --check`、Python AST/内存 compile、Shell `bash -n`、旧 `rough_stratified` assignment 静态等价检查、MultiTerrain CSV/JSON 逐字段审计，以及 checkpoint/evaluation/final-artifact SHA-256 全量验证。

## 9. Git 版本管理结果

已按以下边界提交；logs、evaluation、GPU CSV、console、checkpoint、导出大文件和视频均未进入 Git：

1. `feat: add Go2 challenging multiterrain training task`
   - `multiterrain_velocity_env_cfg.py`
   - Go2 task registration
   - `multiterrain_levels_survival`
   - `train_go2_multiterrain.sh`
2. `feat: add stratified Go2 multiterrain evaluation`
   - `evaluate_go2.py`
3. `docs: add Go2 multiterrain prototype handoff`
   - 原型结果文档、checkpoint/evaluation/export inventory与SHA-256清单；
4. `docs: finalize Go2 multiterrain prototype handoff`
   - 本文件，记录实际提交和最终验证状态。

## 10. MuJoCo 当前状态

尚未运行任何安装/克隆命令：

- `/home/kxj/rl_ws/unitree_sdk2` 不存在；
- `/home/kxj/rl_ws/unitree_mujoco` 不存在；
- `libyaml-cpp-dev/libboost-all-dev/libeigen3-dev/libspdlog-dev/libfmt-dev/libglfw3-dev` 尚未安装；
- 项目已有 `deploy/robots/go2` C++ ONNX Runtime controller和完整 `deploy.yaml`；
- bundled ONNX Runtime C++ 1.22.0存在；Python环境没有 `onnxruntime`，但使用 ONNX ReferenceEvaluator 已完成数值等价验证；
- 用户明确要求暂不执行MuJoCo依赖安装命令，先整理代码并做Git版本管理。

## 11. 下一会话优先级

1. 完整读取旧 rough final handoff和本文件；
2. 只读确认两个仓库HEAD和工作区；
3. 复核本次5个功能文件及本文件；
4. 验证本文件所列四批提交与干净工作区；
5. 用户另行确认后再安装/克隆官方 `unitree_sdk2`、MuJoCo 3.3.6、`unitree_mujoco`，构建Go2 controller；
6. 完成 MuJoCo flat站立、低速前进、转向和并排 Demo；
7. 最后补正式视频、多 evaluation scenario seeds、多 training seeds和受控消融。

## 12. 下一会话首次只读命令

```bash
cd /home/kxj/rl_ws/unitree_rl_lab

sed -n '1,520p' doc/SESSION_HANDOFF_2026-08-25_ROUGH_MULTITRAINING_FINAL.md
sed -n '1,620p' doc/SESSION_HANDOFF_2026-08-25_MULTITERRAIN_PROTOTYPE.md

git log --oneline -8
git status --short
git diff --check
git diff --stat
sha256sum --check doc/repro/multiterrain_checkpoint_sha256_2026-08-25.txt
sha256sum --check doc/repro/multiterrain_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/multiterrain_final_artifacts_2026-08-25.sha256
git -C /home/kxj/rl_ws/IsaacLab log --oneline -3
git -C /home/kxj/rl_ws/IsaacLab status --short
```

默认协作方式：类似读取/只读检查由助手自行运行和分析；需要修改、测试、训练、评估、安装或Git提交时，先向用户说明并取得直接授权，或给出明确终端命令由用户运行并返回完整结果。始终先依据 task、run name、training seed、checkpoint、目录和时间识别run；严格区分 training seed与evaluation scenario seed。
