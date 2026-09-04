# Unitree Go2 强化学习项目：简历与面试表述

> 状态更新（2026-09-04）：项目已从早期 flat/rough/MultiTerrain 扩展到七地形 Teacher–Student 路线。Teacher v1 是冻结 baseline；proxy student 的 imitation loss 下降但闭环失败；Teacher v2 完成最终 heading-command 接口对齐；Teacher v3 针对楼梯与 boxes 的异常步态加入 HeightScan、最弱腿参与/对称、负载足滑移和防绕圈约束。三个 V3 专项策略已完成 2,048 env × 5,000 updates 正式训练和候选筛选。当前两个楼梯候选通过数值 gait gate，Boxes 仍待修复。最终统一 recurrent perceptive student 尚未开始，简历和面试中不得宣称七地形统一策略已经完成。

## 推荐项目标题

**基于 Isaac Lab 的 Unitree Go2 强化学习运动控制与可复现评估**

## 一句话简介

在 Isaac Sim 5.1 / Isaac Lab 2.3.2 中复现 Unitree Go2 速度跟踪策略，完成平地、Rough 多训练 seed、七类 MultiTerrain curriculum、统一分层评估和可复现归档。

## 中文简历版本

- 基于 Isaac Lab 与 RSL-RL 搭建 Unitree Go2 平地 velocity locomotion 训练链路，使用 2048 个并行环境完成 3000 次 PPO iteration、1.47 亿 simulation transitions，训练耗时约 2.0 小时，后期吞吐约 20.3k steps/s。
- 设计可复用的 headless evaluator，固定评估协议并按 episode 输出 CSV/JSON，比较 3 个 checkpoint × 3 个评估场景 seed，共评估 1152 个 episode，避免仅凭训练 reward 选择模型。
- 最终策略在 384 个 episode 上取得 0.0756 ± 0.0021 m/s 平面速度跟踪误差，99.48% episode 达到 20 秒时限，base-contact termination 为 0%。
- 解决 Isaac Sim 5.1、Isaac Lab 2.3.2 与项目代码的依赖及 API 兼容问题，固化训练、评估、视频录制、环境快照和 checksum 流程，并导出 TorchScript/ONNX 策略。
- 扩展 50% flat / 50% random-uniform rough terrain curriculum，在相同 rough-stratified 协议下将 random-rough XY error 相对 flat control 降低 55.7%，timeout rate 提升 43.23 个百分点。
- 将 rough 正式训练、12-run 矩阵、1536-episode 审计与 SHA-256 归档固化为可复用脚本，正式选择跨 seed 更稳定的 `model_2900.pt`。
- 完成 rough training seeds 42/7/123 的独立正式训练与 3456-episode 审计；固定 `model_2999` 的 random-rough XY error 为 0.1131 ± 0.0013 m/s。
- 构建七类 MultiTerrain 与 140-episode 分层协议，比较 survival/tracking-aware 两条受控续训分支；选择 baseline `model_3300`，跨 evaluation scenario seeds 的 XY error 为 0.272 ± 0.010 m/s，并将 tracking-aware 作为负结果归档。
- 制作带动态 command/actual vx 曲线、误差、累计位移和 termination 的七地形及三 training-seed 可审计 Demo。

如果简历空间有限，优先保留训练规模、统一评估、rough 改进和可复现自动化四类信息。

## English resume version

- Reproduced a Unitree Go2 flat-terrain velocity-locomotion policy with Isaac Lab and RSL-RL; trained PPO for 3,000 iterations across 2,048 parallel environments (147M simulation transitions) in 2.0 hours at ~20.3k steps/s.
- Built a deterministic headless evaluation pipeline with per-episode CSV/JSON metrics and compared 3 checkpoints across 3 evaluation-scenario seeds (1,152 episodes total) instead of selecting solely by training reward.
- Selected a final policy achieving 0.0756 ± 0.0021 m/s mean planar-velocity tracking error, a 99.48% 20-second episode completion rate, and zero base-contact terminations over 384 episodes.
- Resolved Isaac Sim 5.1 / Isaac Lab 2.3.2 compatibility issues, automated reproducibility snapshots and artifact checksums, and exported the policy to TorchScript and ONNX.
- Extended training to a 50/50 flat/random-uniform rough-terrain curriculum; reduced random-rough XY tracking error by 55.7% and improved 20-second completion by 43.23 percentage points versus the flat-trained control under the same protocol.
- Automated the rough training wrapper, 12-run stratified matrix, 1,536-episode audit, and immutable checksum manifests; selected `model_2900.pt` for lower tracking error, action variation, power proxy, and cross-scenario variance.
- Completed independent rough training seeds 42/7/123 and a 3,456-episode audit; achieved 0.1131 ± 0.0013 m/s random-rough XY error in the fixed-iteration comparison.
- Built a seven-type MultiTerrain task and 140-episode stratified protocol; compared survival and tracking-aware continuations, selected baseline `model_3300`, and documented the tracking-aware branch as a negative result.

## 面试展开结构

### 1. 为什么不能只看训练 reward？

训练 reward 同时受到采样噪声、课程阶段和价值估计影响。项目先按训练窗口筛出候选，再使用相同评估 seed 和命令分布做配对比较。Flat baseline 最终选择 `model_2999.pt`；rough 扩展则在 `model_2400.pt`、`model_2900.pt` 和 `model_2999.pt` 中选择跨 seed 更稳定、综合指标更好的 `model_2900.pt`。

### 2. 评估协议如何保证公平？

- 所有 checkpoint 使用相同任务、环境数、seed 集合和 episode 数；
- inference policy 不采样训练噪声；
- 关闭评估期间的 terrain/command curriculum 更新；
- 保留随机初始状态、动力学随机化、外力扰动和随机速度指令；
- 同时记录 reward、XY/yaw 误差、动作变化、机械功率代理和终止原因。

### 3. 为什么训练误差比离线评估误差高？

训练统计来自带探索噪声和持续更新的 rollout；统一评估使用固定 checkpoint 的确定性动作，并关闭 curriculum 更新。两者协议不同，不能直接当作同一指标比较。

### 4. 多机器人视频为什么会“穿模”？

视频将多个独立向量化克隆环境放在同一视图展示。Isaac Lab 隔离跨环境碰撞，因此视觉轨迹可以交叠；这不是多机器人交互或避碰实验。单机器人 follow 视频用于验证真实步态细节，grid 视频用于展示并行策略表现。

### 5. 当前项目边界是什么？

当前已完成 flat baseline、50/50 flat/random-uniform rough 的 training seeds 42/7/123，以及单 training-seed 的七类 MultiTerrain 原型。MultiTerrain 在 stairs/boxes/inverted terrain 上仍明显降速，不能表述为任意复杂地形高速泛化。MuJoCo Sim2Sim、Sim2Real 和实机部署尚未完成。

## 下一阶段可讲的改进方向

1. 为 MultiTerrain 增加独立 training seeds，量化训练方差；
2. 解耦 command/terrain curriculum，并针对 stairs/boxes 做受控消融；
3. 做 reward、扰动、观测噪声和 domain randomization 消融；
4. 使用 ONNX 在 MuJoCo 中完成 Sim2Sim 回归；
5. 评估控制频率、延迟、摩擦和质量扰动敏感性，为后续 Sim2Real 做准备。

## 不应使用的夸大表述

- 不写“完成任意复杂地形泛化”；当前 rough 只覆盖 random-uniform height noise。
- 不把 Rough 的多 training-seed 结果外推为 MultiTerrain 多 training-seed 泛化。
- 不写“实现多机器人避碰”；grid 视频只是克隆环境展示。
- 不把机械功率代理写成真实电池功耗。
- 不写“完成实机部署”或“完成 Sim2Real”。
