# Unitree Go2 强化学习项目：简历与面试表述

## 推荐项目标题

**基于 Isaac Lab 的 Unitree Go2 强化学习运动控制与可复现评估**

## 一句话简介

在 Isaac Sim 5.1 / Isaac Lab 2.3.2 中复现 Unitree Go2 平地速度跟踪策略，完成 2048 环境 PPO 训练、统一多场景评估、checkpoint 选择、JIT/ONNX 导出和双视角视频展示。

## 中文简历版本

- 基于 Isaac Lab 与 RSL-RL 搭建 Unitree Go2 平地 velocity locomotion 训练链路，使用 2048 个并行环境完成 3000 次 PPO iteration、1.47 亿 simulation transitions，训练耗时约 2.0 小时，后期吞吐约 20.3k steps/s。
- 设计可复用的 headless evaluator，固定评估协议并按 episode 输出 CSV/JSON，比较 3 个 checkpoint × 3 个评估场景 seed，共评估 1152 个 episode，避免仅凭训练 reward 选择模型。
- 最终策略在 384 个 episode 上取得 0.0756 ± 0.0021 m/s 平面速度跟踪误差，99.48% episode 达到 20 秒时限，base-contact termination 为 0%。
- 解决 Isaac Sim 5.1、Isaac Lab 2.3.2 与项目代码的依赖及 API 兼容问题，固化训练、评估、视频录制、环境快照和 checksum 流程，并导出 TorchScript/ONNX 策略。

如果简历空间有限，优先保留前三条。

## English resume version

- Reproduced a Unitree Go2 flat-terrain velocity-locomotion policy with Isaac Lab and RSL-RL; trained PPO for 3,000 iterations across 2,048 parallel environments (147M simulation transitions) in 2.0 hours at ~20.3k steps/s.
- Built a deterministic headless evaluation pipeline with per-episode CSV/JSON metrics and compared 3 checkpoints across 3 evaluation-scenario seeds (1,152 episodes total) instead of selecting solely by training reward.
- Selected a final policy achieving 0.0756 ± 0.0021 m/s mean planar-velocity tracking error, a 99.48% 20-second episode completion rate, and zero base-contact terminations over 384 episodes.
- Resolved Isaac Sim 5.1 / Isaac Lab 2.3.2 compatibility issues, automated reproducibility snapshots and artifact checksums, and exported the policy to TorchScript and ONNX.

## 面试展开结构

### 1. 为什么不能只看训练 reward？

训练 reward 同时受到采样噪声、课程阶段和价值估计影响。项目先按训练窗口筛出 2400/2900/2999 三个候选，再使用相同评估 seed 和命令分布做配对比较。最终选择 `model_2999.pt`，因为它的 XY 跟踪误差最低、聚合 reward 最高且跨场景波动最小。

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

当前结果只证明平地速度跟踪 baseline。训练只使用 seed 42，三个 seed 是评估场景 seed；复杂地形、多训练 seed、MuJoCo Sim2Sim 和 Sim2Real 尚未完成，面试时不应扩大表述。

## 下一阶段可讲的改进方向

1. 恢复 rough/stairs/boxes 地形并设计 terrain curriculum；
2. 用多训练 seed 比较 baseline 与课程学习改进版；
3. 做 reward、扰动、观测噪声和 domain randomization 消融；
4. 使用 ONNX 在 MuJoCo 中完成 Sim2Sim 回归；
5. 评估控制频率、延迟、摩擦和质量扰动敏感性，为后续 Sim2Real 做准备。

## 不应使用的夸大表述

- 不写“完成复杂地形泛化”；当前只有 flat sub-terrain。
- 不写“完成多 seed 训练”；当前只有训练 seed 42。
- 不写“实现多机器人避碰”；grid 视频只是克隆环境展示。
- 不把机械功率代理写成真实电池功耗。
- 不写“完成实机部署”或“完成 Sim2Real”。
