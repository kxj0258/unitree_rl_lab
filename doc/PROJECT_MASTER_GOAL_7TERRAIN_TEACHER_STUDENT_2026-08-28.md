# Unitree Go2 七地形项目总目标与 Teacher–Student 路线（2026-08-28 固化）

## 1. 本文件地位

本文件固化项目长期总目标。后续会话、训练、评估、Demo 和代码设计不得只围绕当前 stairs
局部结果而遗忘本目标。若未来改变路线，必须新增带日期的决策文件，不覆盖本文件。

## 2. 完整项目目标

最终目标是训练并审计一个 Unitree Go2 recurrent perceptive student policy，使其在以下七类地形
及其难度 curriculum 上执行通用速度命令跟踪，并形成高质量连续 Demo：

1. flat；
2. random rough；
3. pyramid slope；
4. inverted-pyramid slope；
5. pyramid stairs；
6. inverted-pyramid stairs；
7. boxes/discrete obstacles。

最终 student 必须是单一统一策略；训练期间可以使用 terrain identity、privileged state 和多个
teacher，但最终 student 的部署观测不得依赖 simulator-only terrain label。

有效成功不能由 timeout 或 reward 单独定义，必须同时检查 command tracking、command-aligned
progress、世界净位移、path length、stall、terrain-specific progress（如高度增益）、termination、
功耗和视频。

## 3. 总体分两条工作线

### Track A：先用当前成果制作 Demo v1

不新增训练，使用已验证结果形成一版诚实的当前成果 Demo：

- 历史 HeightScan model10000 多地形 34 s 作品集片段；
- stairs model4999 的下楼/综合 stairs 片段；
- model4999 inverted-stairs 绕圈失败的 before 诊断；
- remediation model900 level7 连续上楼的 after 片段；
- 叠加 checkpoint、training seed、evaluation scenario seed、terrain/level、位移/yaw/stall/termination；
- 明确该 Demo 是多 checkpoint 当前成果汇编，不冒充一个统一 policy 的七地形结果。

Track A 的目的：先保留并展示已经完成的工程闭环，不等待新的从零 teacher/student 项目全部结束。

### Track B：从零重启七 teacher → 从零统一 student

在独立的新 experiment namespace 中，从随机初始化分别训练七个 terrain teachers。所有 teacher
共享相同 Go2 action semantics、控制频率、基础 command schema 和尽可能一致的 actor observation
接口；允许 critic/teacher 使用 terrain-specific privileged observations。

七个 teacher 完成固定分层、多个 evaluation scenario seeds 和 checkpoint 选择后，随机初始化一个
统一 recurrent perceptive student。训练时按 1/7 terrain family 平衡采样，由 ground-truth terrain
identity 仅用于选择对应 teacher 产生 action target；student 本身只接收统一可部署观测。

student 先做在线 multi-teacher distillation，再做 balanced PPO consolidation，并保留 teacher
regularization/rehearsal，最后在七地形统一 acceptance matrix 上选择 checkpoint。

## 4. 为什么方案可行

当前本地 RSL-RL 3.1.2 已提供：

- `DistillationRunner`；
- `Distillation` algorithm；
- `StudentTeacher`；
- `StudentTeacherRecurrent`；
- recurrent teacher loading 与 observation normalization；
- student rollout、teacher action label、MSE/Huber behavior loss。

本地 runner 由 student action 驱动环境，再让 teacher 对 student 遇到的 observation 给标签，因此
不是只在 teacher 固定轨迹上做离线 BC，能降低 covariate shift。官方实现一次只支持一个 teacher；
本项目需要在不升级 RSL-RL 的前提下新增 multi-teacher dispatcher 和审计逻辑。

开源/论文依据：

- RSL-RL 官方支持 PPO 与 Student–Teacher Distillation：
  https://github.com/leggedrobotics/rsl_rl
- Isaac Lab 官方环境列表已包含 RSL-RL DISTILLATION 与 RECURRENT DISTILLATION：
  https://isaac-sim.github.io/IsaacLab/develop/source/overview/environments.html
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* 使用仿真强化学习与
  proprioceptive controller 展示复杂自然地形鲁棒性： https://arxiv.org/abs/2010.11251
- Extreme Parkour 采用 privileged teacher 到 vision student 的课程/蒸馏路线并开源：
  https://github.com/chengxuxin/extreme-parkour 与 https://arxiv.org/abs/2309.14341
- RSL/IsaacLab 的 WLR 示例给出 teacher task → student task 的两阶段训练：
  https://github.com/leggedrobotics/wlr-competition-2025

这些工作支持 privileged teacher、可部署 student、recurrent/vision distillation 的总体方向；它们
不直接证明“七个完全独立 teacher”必然最优，因此本项目必须做自己的 ablation 和验收。

## 5. 对“每种地形从零 teacher”的优化约束

用户选择七个独立从零 teacher，作为主实验路线。为避免 teacher 无法合并，必须满足：

1. action dimension、joint order、action scale、control decimation 完全一致；
2. student-visible observation schema、顺序、单位、normalization 规范一致；
3. command schema 一致，至少覆盖相同的 vx/vy/yaw/heading 子集；
4. episode/termination 基础语义一致；
5. terrain-specific reward 可以不同，但共同 tracking/safety 项应有统一定义；
6. 每个 teacher 都必须覆盖其 terrain 的 levels 与命令方向，不能只学习单一演示轨迹；
7. “最优 teacher”由固定分层评估选择，不默认最后 checkpoint；
8. 开发阶段可以 training seed 42，正式声称鲁棒最优前应补多个独立 training seeds；
9. 保存 teacher action mean/distribution、normalizer、GRU 状态边界和 checkpoint SHA-256；
10. 不进行 checkpoint 权重平均。

完全独立从零训练能提供干净的 terrain-specific upper bound 和良好 ablation，但计算重复较大。
保留一个次要对照实验：同配置从共享 foundation 分叉。该对照不取代用户选择的从零主路线，用于
量化从零 teacher 是否真的优于 transfer teacher。

## 6. 七个 Teacher 阶段

建议按由易到难实施，但每个 teacher 都从随机初始化：

1. flat teacher；
2. random-rough teacher；
3. pyramid-slope teacher；
4. inverted-slope teacher；
5. pyramid-stairs teacher；
6. inverted-stairs teacher；
7. boxes teacher。

每个 teacher 的流程：task/config → 128-env smoke → 正式 PPO → TB/checkpoint/GPU 审计 →
checkpoint candidates → evaluation scenario seeds 7/42/123 → terrain-specific directional test → video →
SHA-256 manifest → teacher freeze。

Teacher 可以拥有 privileged critic/teacher observation，例如 clean base velocity、contacts、terrain
height/geometry、friction/mass；student-visible actor observation必须保持统一。若 teacher actor 使用
privileged 输入，multi-teacher runner 必须显式区分 teacher 与 student observation groups。

## 7. Multi-teacher 数据与在线蒸馏

### 7.1 平衡采样

每个 batch 的 environment quota 原则上按 terrain family `1/7` 平衡；每个 family 内再平衡：

- difficulty level；
- command bins（forward/reverse/lateral/yaw/heading）；
- success/failure recovery states；
- episode phase。

不能按各 teacher 自然成功率采样，否则 flat/容易 terrain 会支配 student loss。

### 7.2 Teacher dispatcher

训练时使用 simulator ground-truth terrain identity 选择 teacher，仅用于产生监督 target，不放入
student observation。所有 teacher 处于 eval mode，normalizer 固定，各自维护 recurrent hidden state，
在相应 env done 时 reset。

### 7.3 Student 从零开始

统一 student 随机初始化，建议延续 perceptive GRU actor：统一 proprioception + HeightScan/后续深度
特征，GRU hidden 256 起步。student 不加载 model3200/model4999/model900 权重，确保“from scratch
student”实验定义干净；这些旧模型仅属于 Track A 与历史证据。

### 7.4 在线而非纯离线 BC

第一阶段使用 student action rollout，teacher 对 student 实际访问到的 state 给 action target；优先
Huber/action-distribution loss，并记录每 terrain behavior loss。必要时使用 teacher/student action
混合退火，防止随机 student 初期立即跌倒导致数据退化。

第二阶段加入 balanced PPO consolidation：

`L_total = L_PPO + lambda_bc * L_teacher + lambda_aux * L_terrain_or_context`

`lambda_bc` 从高到低退火但不立即归零；PPO 修复 teacher imitation 的 compounding error，teacher
loss 防止 terrain 间灾难性遗忘。当前官方 RSL-RL Distillation 是单 teacher 且纯 behavior loss，
multi-teacher + PPO 联合目标需要项目内扩展，不能假称已有开关即可完成。

## 8. Composite Mixed-Terrain 阶段

仅在每 env 固定一种 terrain 上训练，不能证明 student 能在同一 episode 中跨 terrain boundary。
蒸馏稳定后必须新增 composite course：七类 patch 在一条路线中连接，随机化顺序、长度和难度。

训练时 teacher dispatcher 可按当前位置/local patch ground truth 选择 teacher；student 不接收该标签，
依赖 HeightScan 与 GRU 推断。必须检查边界处 action continuity、hidden-state 适应、stall 与跌倒。

## 9. 七地形统一 Acceptance Matrix

最终 teacher/student 均按统一协议审计：

- training seed 与 evaluation scenario seed 分列；
- 至少 evaluation scenario seeds 7/42/123；
- 7 terrain families × difficulty levels × command bins；
- fixed-direction、random-command、heading-locked；
- mixed-patch transition course；
- XY/yaw tracking；command-aligned distance；世界净位移/path ratio；
- slope/stairs 高度变化；boxes obstacle pass/collision；
- stall、power/energy、base contact、bad orientation、timeout；
- recurrent hidden reset；non-finite；吞吐/GPU；
- checkpoint/optimizer/iteration、TB、console/GPU、CSV/JSON/video SHA-256。

student 不能只比较平均分。每 terrain 必须达到相应 teacher 的预设保留比例，并设置 hard safety
veto；任何 terrain 的 bad orientation/base contact 超阈值，都不能被其他 terrain 的高分抵消。

## 10. 分阶段里程碑

### M0：当前成果 Demo v1

完成 current-results 多 checkpoint before/after Demo 和 metadata/manifest。

### M1：七 teacher 规范

冻结统一 observation/action/command/metric schema，完成七 task smoke。

### M2：七个从零 teacher

依次训练、评估、选择并冻结七个正式 teacher。

### M3：Multi-teacher recurrent distillation

实现 dispatcher、balanced sampler、teacher hidden state、student-from-scratch、per-terrain loss。

### M4：Balanced PPO consolidation

联合 teacher regularization 与 PPO，避免纯 BC 分布偏移和灾难性遗忘。

### M5：Composite mixed-terrain

同 episode 七地形 transition curriculum 与固定验收。

### M6：最终统一 Demo 与复现包

只使用一个 student checkpoint 完成七地形连续 Demo，生成报告、metadata、CSV、video、manifest。

### M7（可选）：Sim-to-real

domain randomization、actuator/latency、sensor mapping、export、safety gate、逐级真机验证。

## 11. 永久决策

- 当前结果先制作 Track A Demo v1；
- 长期主路线采用七个 terrain-specific teachers 各自从零训练；
- 统一 recurrent perceptive student 也从零初始化；
- student 使用平衡 multi-teacher 在线蒸馏，再用 PPO consolidation；
- 最终目标是单一 student 在七地形和 mixed transition course 上速度跟踪；
- 不用简单权重平均，不用顺序 fine-tuning 冒充融合；
- 所有阶段必须保留历史 artifacts、固定 seeds 身份和 SHA-256 证据；
- 在 M6 完成前，不宣称统一七地形最终策略已经完成。
