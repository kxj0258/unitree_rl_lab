# Unitree Go2 RL 项目状态

更新时间：2026-08-25

## 项目目标

基于 Unitree RL Lab 和 Isaac Lab 复现 Go2 velocity locomotion，并形成面向机器人强化学习/具身智能岗位的完整项目。目前不做实机部署。

## 固定软件版本

- Ubuntu 22.04.5 LTS
- NVIDIA Driver 580.173.02 Open Kernel Module
- Python 3.11.16
- PyTorch 2.7.0+cu128
- Torchvision 0.22.0+cu128
- Isaac Sim 5.1.0.0
- Isaac Lab v2.3.2，commit `37ddf626871758333d6ed89cf64ad702aef127d0`
- Unitree RL Lab upstream baseline commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`
- Go2 rough task commit `dea17cc`
- Rough stratified evaluator commit `61843d8`
- Unitree model commit `323e350252b9c3aee9c40acbfdad84f6ce46a5ac`

禁止在没有完整回归验证的情况下单独升级驱动、Isaac Sim、Isaac Lab、PyTorch，或覆盖 checkpoint 与训练日志。

## Baseline 阶段：已完成

- 完成 Isaac Sim 5.1 headless、Isaac Lab Cartpole 和 Go2 多环境吞吐验收。
- 使用 2048 environments、训练 seed 42、3000 PPO iterations 完成 147,456,000 timesteps 训练。
- 分析 TensorBoard 全部训练曲线，无非有限值。
- 实现可复用训练、TensorBoard 分析、确定性评估和视频录制脚本。
- 对 2400/2900/2999 三个 checkpoint 使用评估 seed 7/42/123 完成统一评估，共 1152 个 episode。
- 正式选择 `model_2999.pt`；该模型单独覆盖 384 个 episode。
- 完成单环境 GUI 推理、单环境 follow 视频和 9 环境 grid 视频。
- 完成 TorchScript 与 ONNX 导出。
- 保存环境、版本、兼容 diff、checkpoint、评估 summary 和最终产物校验值。

## 正式 baseline

```text
Checkpoint: logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt
SHA-256:   a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928
```

三个评估 seed、384 个 episode 的聚合指标：

- Mean reward：34.6861 ± 0.0358
- Mean XY velocity error：0.0756 ± 0.0021 m/s
- XY velocity RMSE：0.1210 ± 0.0016 m/s
- Mean yaw velocity error：0.0575 ± 0.0007 rad/s
- Episode timeout rate：99.48%（382/384）
- Base-contact termination：0%（0/384）
- Bad-orientation termination：0.52%（2/384）

这是单次训练 seed 42 的策略在三个评估场景 seed 下的结果，不是多训练 seed 结果。

## Baseline 定位

当前 Go2 配置虽然名为 `COBBLESTONE_ROAD_CFG`，但只有 `flat` sub-terrain 被启用，rough、stairs、boxes 等复杂地形仍被注释。

准确表述为：

> 带动力学随机化、周期外力扰动和速度指令课程的平地 Go2 locomotion baseline。

不能将当前 `terrain_levels` 曲线解释为复杂地形训练结果。

## Rough 50/50 阶段：已完成

- 新增注册任务 `Unitree-Go2-Velocity-Rough`；
- 使用 10 × 20 terrain grid，50% flat、50% random-uniform rough；
- rough noise range 为 0.01–0.06 m，difficulty range 为 0.0–1.0；
- 保持 45 维 policy observation，不加入 height scan；
- 完成 256×100 smoke、2048×100 scale check、2048×500 pilot；
- 完成 2048 environments、seed 42、3000 iterations 正式训练；
- 正式训练共 147,456,000 transitions，耗时 2:11:41，exit 0，非有限值 0；
- 实现精确 50/50、level 0–9 覆盖的 `rough_stratified v1` 评估协议；
- 完成 flat control + rough 2400/2900/2999 × seeds 7/42/123 的 12-run 矩阵；
- 审计 1536 episodes，base-contact termination 和非有限值均为 0；
- 正式选择 rough `model_2900.pt`。

## 正式 rough checkpoint

```text
Checkpoint: logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000/model_2900.pt
SHA-256:   9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889
```

三个评估场景 seed、384 episodes 的聚合指标：

- Mean reward：31.8819 ± 0.1171；
- Overall XY error：0.0950 ± 0.0008 m/s；
- Random-rough XY error：0.1123 ± 0.0023 m/s；
- Overall yaw error：0.0702 ± 0.0005 rad/s；
- Timeout：99.22%（381/384）；
- Base contact：0%（0/384）；
- Bad orientation：0.78%（3/384）。

相较 flat-trained control 的相同 random-rough subset，rough 正式模型将 XY error 降低 55.7%，将 timeout rate 提升 43.23 个百分点。该结论仍来自单一训练 seed 42。

## Rough 多训练 Seed：已完成

- 使用相同 2048 environments × 3000 iterations 预算完成 training seeds 42、7、123；
- 三个 formal run 共 442,368,000 transitions，exit 与非有限值检查全部通过；
- 每个 training seed 评估 `model_2400/2900/2999` × scenario seeds 7/42/123；
- 审计 27 个 rough runs、3456 episodes，base contact 与非有限值均为 0；
- Per-seed official：seed 42 `model_2900`、seed 7 `model_2999`、seed 123 `model_2999`；
- 固定 iteration 跨 training-seed 主结果推荐 `model_2999`；
- 固定 `model_2999`：overall XY `0.0956 ± 0.0009 m/s`，rough XY `0.1131 ± 0.0013 m/s`，timeout `1144/1152`；
- 完整结果见 `RESULTS_ROUGH_MULTITRAINING.md`。

## 兼容修改

### Isaac Lab

`source/isaaclab/setup.py`：Starlette 从 0.49.1 固定为 0.45.3，以兼容 Isaac Sim 5.1 固定的 FastAPI 0.115.7。

### Unitree RL Lab

- `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`：模型路径读取 `UNITREE_MODEL_DIR` 和 `UNITREE_ROS_DIR`。
- `scripts/rsl_rl/play.py`：修正 Isaac Lab v2.3.2 导入路径，增加显式 seed 与 world/follow/grid 摄像模式。
- `scripts/project/evaluate_go2.py`：固定场景 seed、关闭评估期 curriculum 更新，输出 episode CSV、metadata 和 summary JSON。
- `scripts/project/train_go2_rough.sh`：参数化 rough smoke/pilot/formal 训练，记录 console 与 GPU CSV。
- `scripts/project/evaluate_go2_rough_matrix.sh`：正式 12-run rough checkpoint 矩阵与 dry-run。
- `scripts/project/audit_go2_rough_evaluation.py`：评估完整性、非有限值、termination 和分层指标审计。
- `scripts/project/archive_go2_rough.py`：rough official marker、聚合 CSV、inventory 和 SHA manifest 的 create/verify。
- `scripts/project/evaluate_go2_rough_training_seed_matrix.sh`：明确区分 training/evaluation seed 的单 training-seed 9-run 矩阵。
- `scripts/project/audit_go2_rough_multitraining.py`：27-run 多 training-seed 审计及 selected/fixed 聚合。
- `scripts/project/archive_go2_rough_multitraining.py`：追加式六文件多 training-seed 归档。

## 视频说明

- 9 环境 grid 视频用于展示并行仿真和策略表现力。
- 单环境 follow 视频用于展示步态、姿态和转向细节。
- 多环境视频中的机器人属于独立克隆环境，跨环境轨迹交叠不代表真实碰撞或多机器人避碰能力。

## 下一阶段计划

1. 设计 terrain curriculum、扰动、观测噪声和 reward 消融，每次只改变一个因素。
2. 逐项扩展 stairs、boxes、slopes，并为每类新增地形定义独立评估协议。
3. 完成 rough 正式视频、ONNX 导出与 MuJoCo Sim2Sim 回归。
4. 在完成 Sim2Sim 与安全审查后再讨论 Sim2Real；当前不宣称实机部署。
