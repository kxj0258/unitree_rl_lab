# Unitree RL Lab 扩展优化规划

## 1. 文档用途

本文档用于指导后续基于个人仓库 `unitree_rl_lab` 的复现、优化、评测和项目经历建设。

个人仓库：<https://github.com/kxj0258/unitree_rl_lab>

上游仓库：<https://github.com/unitreerobotics/unitree_rl_lab>

核心定位：

> 基于 Isaac Lab、RSL-RL 和 PPO 的 Unitree Go2 四足机器人强化学习复现与多地形策略评测项目。

项目应继续围绕“训练可信、评估分层、策略可部署、失败可解释”推进，而不是单纯继续增加训练次数或地形数量。

## 2. 当前项目状态

根据个人仓库 README 和已有项目材料，目前已经具备较完整的仿真强化学习实验基础。

### 2.1 已完成的主要工作

- Unitree Go2 flat terrain velocity tracking baseline。
- random rough terrain 扩展。
- seven-terrain MultiTerrain 原型。
- HeightScan 迁移和地形相关配置研究。
- 多个独立 teacher policy 的训练与候选筛选。
- candidate、cross-seed、fixed-command 分层评估。
- 视频导出和评估结果归档。
- JIT / ONNX policy export 学习与部署接口梳理。
- online proxy dispatcher 和 imitation smoke test。
- 10,000-update recurrent proxy student 实验。
- 明确记录了 student 统一闭环暂未达到可用状态。
- Teacher v2 的 relative-heading command 和 gait-aware reward 对齐。
- Teacher v3 针对 pyramid stairs、inverted pyramid stairs 和 boxes 的专项修复。

### 2.2 已记录的主要结果

以下结果来自仓库 README 和历史材料，后续 Codex 必须以仓库中的原始记录、CSV、视频 manifest 和 checkpoint 重新核对，不能只复制下列数字。

#### Flat baseline

- 2048 个并行环境。
- training seed 42。
- 3000 PPO iterations。
- 147,456,000 simulation steps。
- 训练约 2 小时 2 分钟。
- 最后 500 iterations 约 20.3k steps/s。
- `model_2999.pt` 通过 scenario seeds 7、42、123 的候选筛选。
- 384 个评估 episode。
- mean XY velocity error：0.0756 ± 0.0021 m/s。
- XY velocity RMSE：0.1210 ± 0.0016 m/s。
- 382/384 episode 达到 20 秒时限。
- base-contact termination：0/384。

#### Rough terrain

- 50% flat + 50% random-uniform rough terrain。
- rough noise 约 0.01～0.06 m。
- 147,456,000 transitions。
- 1,536 个审计 episode，无 non-finite 值和 base-contact termination。
- `model_2900.pt` 的 384 个 formal episode：overall XY error 0.0950 ± 0.0008 m/s。
- random-rough XY error：0.1123 ± 0.0023 m/s。
- timeout rate：99.22%。
- 相对 flat-trained control，在同一 random-rough 子集上的 XY error 降低 55.7%。

#### MultiTerrain

覆盖：

- flat；
- random rough；
- positive/inverted slopes；
- positive/inverted stairs；
- boxes。

当前 survival continuation `model_3300.pt` 在 scenario seeds 7、42、123 下记录：

- XY error：0.272 ± 0.010 m/s；
- timeout rate：79.3% ± 4.3 个百分点。

项目已经明确：MultiTerrain policy 在 flat 上表现较好，但在 stairs、boxes 和 inverted terrain 上速度显著下降。因此当前结果证明了评测和地形覆盖流程，不证明已经解决复杂地形高速通过问题。

#### Teacher v3

- 每个地形独立训练 teacher。
- training seed 42。
- 2048 environments。
- 5000 PPO updates。
- 每个地形约 327,680,000 transitions。
- pyramid stairs `model_4999`、inverted pyramid stairs `model_4700`、boxes `model_4999` 为当前候选。
- pyramid 和 inverted stairs 通过 numeric level-5 gait gate。
- boxes 因 bad-orientation termination 和近零 signed progress 尚未通过 gate。
- 视觉步态复核、cross-seed 和 fixed-command 验证仍是后续门禁。

## 3. 目前最重要的问题

### 3.1 MultiTerrain 统一策略没有真正解决难地形

当前 MultiTerrain 结果显示，统一策略虽然覆盖多个地形，但不同地形间存在明显性能不均衡：

- flat 上跟踪较好；
- stairs、boxes、inverted terrain 上速度下降；
- timeout 率和速度误差不能单独代表步态质量；
- 单一 aggregate metric 会掩盖具体地形失败。

后续应优先做“按地形、等级、命令和终止原因分解”的诊断，而不是直接继续扩大训练规模。

### 3.2 teacher 成功不等于 student 成功

已有 recurrent proxy student 实验显示 imitation loss 下降，但统一 student policy 仍不可用。

这说明问题可能来自：

- teacher action 分布冲突；
- command/heading 语义不一致；
- 不同地形所需动作策略差异过大；
- teacher 的局部最优行为互相覆盖；
- recurrent state 训练与部署状态初始化不一致；
- student 只优化 action imitation，没有优化任务结果；
- 训练地形和评估地形分布不一致。

因此不应再以 imitation loss 作为 student 成功的主要证据，必须增加闭环任务指标。

### 3.3 评估门禁还需要进一步标准化

当前已有 candidate、cross-seed、fixed-command、gait gate 等概念，但后续需要将它们固化为机器可读的协议：

- checkpoint 候选筛选集与最终确认集分离；
- scenario seed 与 training seed 分离；
- 训练地形和评估地形分离；
- 每个 terrain level 有明确样本数；
- 每个 episode 保留终止原因；
- 视频和 CSV 具有一一对应关系；
- 所有选择规则提前写入配置，避免事后挑选最好结果。

### 3.4 部署边界不能被夸大

仓库有 JIT/ONNX 导出、MuJoCo sim2sim 和 C++ controller/Unitree SDK2 路径，但：

- 导出成功不等于部署闭环成功；
- sim2sim 通过不等于 sim2real 通过；
- 真实机器人结果目前不能直接写入简历；
- 观测顺序、关节顺序、动作缩放、控制频率和默认姿态必须逐项核对。

## 4. 总体扩展原则

后续工作遵守以下原则：

1. 先修复评估可信性，再优化策略。
2. 先固定观测/动作/命令合同，再做 teacher-student。
3. 先做单地形受控消融，再做七地形统一训练。
4. 训练指标、闭环指标、步态指标和部署指标分开记录。
5. 所有优化必须有 baseline、对照组和失败分析。
6. 上游代码保持独立，个人实验修改放在配置、脚本和扩展目录。
7. 不把单段成功视频当作泛化结论。
8. 不把训练吞吐当作实时控制延迟。

## 5. 推荐扩展路线

推荐优先级：

```text
P0 评估协议和证据整理
        ↓
P1 单地形/单任务受控修复
        ↓
P2 MultiTerrain 诊断与 curriculum 优化
        ↓
P3 teacher-student 重新设计
        ↓
P4 policy runtime / ONNX / MuJoCo sim2sim
        ↓
P5 真实机器人前的安全验证
```

## 6. P0：建立可信的评测与证据系统

### 目标

将目前已有的评估流程整理成可重复、可审计、不可事后挑选的 benchmark。

### 具体工作

建立统一 episode schema：

```yaml
run_id: string
training_seed: int
scenario_seed: int
checkpoint: string
checkpoint_sha256: string
task: string
terrain_type: string
terrain_level: int
command_profile: string
episode_index: int
duration_s: float
timeout: bool
termination_reason: string
mean_xy_velocity_error: float
xy_velocity_rmse: float
mean_yaw_velocity_error: float
gait_gate: object
video_path: optional string
git_commit: string
config_hash: string
```

建立固定评估层级：

1. smoke test：确认环境和 checkpoint 可运行；
2. candidate screen：快速筛选 checkpoint；
3. formal evaluation：固定 scenario seed 和 episode 数量；
4. cross-training-seed evaluation：验证训练随机性；
5. fixed-command evaluation：验证命令覆盖；
6. gait gate：验证步态质量和安全性；
7. deployment evaluation：验证导出和运行时一致性。

### 验收标准

- 每个结果可以追溯到 task、terrain、level、seed、checkpoint 和 commit；
- 评估脚本不会覆盖已有结果；
- 失败 episode 不被自动过滤；
- aggregate 指标可由 episode-level CSV 重新计算；
- 所有候选选择规则写入文档和配置。

## 7. P1：单地形受控优化

不要直接从七地形继续调参。先选择一个最困难但可复现的任务，例如 pyramid stairs 或 boxes。

### 7.1 观测消融

比较：

- proprioception only；
- 加入历史观测；
- 加入 HeightScan；
- 不同 HeightScan 分辨率；
- 不同 scan clipping；
- actor/critic 是否使用不同特权信息。

保持以下因素不变：

- PPO 算法；
- 总训练 transitions；
- 并行环境数；
- training seed；
- command 分布；
- reward 权重；
- evaluator。

### 7.2 reward 消融

逐项评估：

- tracking reward；
- body orientation；
- base height；
- foot contact；
- foot slip；
- action rate；
- energy；
- anti-circling progress；
- worst-leg participation/symmetry。

重点观察：

- 速度误差是否降低；
- bad orientation 是否减少；
- loaded-foot slip 是否减少；
- 步态是否更对称；
- 是否出现 reward hacking；
- 是否牺牲了其他地形的性能。

### 7.3 command 分布消融

比较：

- non-negative forward velocity；
- relative heading target；
- 不同 forward speed 范围；
- heading ±45° 内不同采样分布；
- command resampling interval；
- fixed-command 与随机 command 的泛化。

当前项目已经明确 reverse、lateral 和 in-place-turn 不属于最终部署命令范围，因此不能在结果中暗示策略支持这些能力。

## 8. P2：MultiTerrain 统一策略优化

### 8.1 从单一 aggregate reward 转向分层目标

建议将训练和评估拆成：

```text
通用目标：稳定、跟踪、低接触、低滑移
        +
地形特定目标：楼梯抬腿、箱体通过、倒置地形稳定
        +
命令目标：forward speed + relative heading
```

避免所有地形共享同一组 reward 权重却要求相同步态表现。

### 8.2 地形采样策略

建立 terrain-balanced sampler：

- 每种地形至少有明确采样比例；
- 每个 level 有最低采样量；
- 对失败率高的地形做受控 oversampling；
- 保留未见 level 作为最终评估；
- 不将最终测试集用于 curriculum 调节。

### 8.3 课程学习

建议逐步增加：

1. 地形高度/坡度；
2. 地形间断程度；
3. command speed；
4. heading 偏差；
5. 摩擦和质量随机化；
6. 外部扰动。

每一步都需要记录 curriculum 状态，避免只知道最终模型而不知道训练分布如何变化。

## 9. P3：teacher-student 重新设计

当前 student 失败是非常有价值的负结果，应围绕它设计严谨实验，而不是简单增加 imitation updates。

### 9.1 先检查 teacher 数据合同

确认所有 teacher 是否使用一致的：

- observation 顺序；
- action 维度；
- action scale；
- command 语义；
- heading 定义；
- control frequency；
- episode reset；
- terrain metadata。

### 9.2 teacher 数据采样

不要只保存 teacher action。每条样本还应保存：

```yaml
observation
action
command
terrain_type
terrain_level
teacher_id
episode_id
timestamp
termination_reason
teacher_value_or_confidence
```

### 9.3 student 训练对照

至少比较：

1. 纯 action imitation；
2. action imitation + terrain ID；
3. action imitation + command conditioning；
4. recurrent student；
5. mixture-of-experts student；
6. teacher routing + shared low-level policy；
7. imitation warm start + PPO fine-tuning。

### 9.4 student 评估

不能只看 imitation loss，必须同时报告：

- 每个地形的 success/timeout；
- velocity error；
- bad orientation；
- foot slip；
- gait gate；
- unseen scenario seed；
- fixed-command；
- rollout stability。

推荐先做“teacher routing + shared adapter”，再尝试完全统一的单一 student。这样可以判断失败到底来自共享表示不足，还是来自训练过程。

## 10. P4：策略导出与 sim2sim

### 10.1 导出合同

建立 policy manifest：

```yaml
policy_name: string
source_checkpoint: string
source_sha256: string
export_format: jit_or_onnx
observation_dim: int
observation_order: list
action_dim: int
action_scale: float
default_joint_position: list
joint_order: list
control_dt: float
command_contract: string
normalization: object
export_commit: string
```

### 10.2 数值一致性

对同一批 observation 比较：

- PyTorch output；
- JIT output；
- ONNX Runtime output；
- C++ runtime output。

记录：

- 最大绝对误差；
- 平均绝对误差；
- 每个关节误差；
- 动作限幅前后误差；
- 不同 batch 和序列长度的差异。

### 10.3 sim2sim 评估

在 MuJoCo 中固定：

- 同一 command；
- 同一初始姿态；
- 同一控制频率；
- 同一 episode 长度；
- 多个 scenario seed。

比较 Isaac Sim 与 MuJoCo 的：

- 速度跟踪误差；
- 姿态稳定性；
- 足端接触；
- 能耗和动作幅度；
- 终止原因；
- 步态周期；
- 延迟和控制频率。

只有 sim2sim 结果稳定后，才考虑真实机器人。

## 11. P5：真实机器人前的安全门禁

真实机器人 sim2real 不属于当前已完成事实。后续若要推进，至少需要：

- 明确机器人型号和控制接口；
- 完成关节顺序、单位和限幅核对；
- 低速、低幅度、短时长测试；
- 外部急停和软件退出状态；
- 站立策略和安全姿态；
- command 白名单；
- watchdog；
- 控制频率监控；
- 电池、电机温度和通信状态监控；
- 人工现场监管。

不得把仓库中存在的 SDK2 或 C++ controller 代码直接表述成已完成真实机器人部署。

## 12. 推荐的项目成果结构

建议最终形成：

```text
unitree_rl_lab/
├── config/
├── source/
├── deploy/
├── evaluation/
│   ├── episode_schema.json
│   ├── benchmark_protocol.md
│   ├── candidate_screen/
│   ├── formal_eval/
│   ├── cross_seed/
│   └── failure_analysis/
├── scripts/
│   ├── train/
│   ├── evaluate/
│   ├── export/
│   ├── compare_runtime/
│   └── report/
├── doc/
│   ├── observation_action_contract.md
│   ├── terrain_matrix.md
│   ├── teacher_student.md
│   ├── sim2sim.md
│   └── resume.md
└── results/
    ├── manifests/
    ├── csv/
    └── plots/
```

## 13. 可形成的项目经历

当前可写成：

> 基于 Isaac Lab、RSL-RL 和 PPO 复现 Unitree Go2 速度跟踪强化学习任务，完成 flat、random rough 和 seven-terrain MultiTerrain 的训练、checkpoint 筛选、分层评估、视频导出及 JIT/ONNX 策略导出；建立区分 training seed、scenario seed、terrain level、command profile、终止原因和 gait gate 的可复现实验协议，并针对复杂地形和 teacher-student 统一策略开展失败分析与受控优化。

完成后可进一步写：

> 针对 stairs、inverted terrain 和 boxes 的性能退化，建立地形均衡采样、HeightScan、gait-aware reward 和动作/命令合同消融，比较独立 teacher、共享 student、teacher routing 和 PPO fine-tuning 的闭环表现。

部署成果只有在实际验证后才能写成：

> 完成 JIT/ONNX/C++ runtime 的数值一致性验证和 MuJoCo sim2sim 评测。

真实机器人 sim2real 需要独立证据，不得提前写入。

## 14. 给 Codex 的可复制提示词

```text
你正在协助开发个人仓库 https://github.com/kxj0258/unitree_rl_lab。

请先阅读仓库 README、config/、source/、scripts/、evaluation/、deploy/ 和 doc/，并阅读本扩展规划文档。项目基于 Unitree Go2、Isaac Lab、RSL-RL 和 PPO，当前已有 flat、rough、seven-terrain MultiTerrain、teacher v1/v2/v3、候选筛选、cross-seed/fixed-command 评估、视频归档、JIT/ONNX 导出和 teacher-student 失败记录。

必须遵守以下事实边界：
- 不要把上游 Unitree RL Lab 的代码写成个人原创实现；
- 不要把规划、历史材料或单段视频写成已完成的泛化结果；
- 不要把 teacher 成功写成 student 成功；
- 不要把 ONNX/JIT 导出写成 sim2sim 或 sim2real 已完成；
- 不要把真实机器人部署写成已完成，除非仓库存在可核验的运行记录；
- 当前 boxes candidate 已有明确失败事实，应保留为负结果，不得删除或掩盖。

工作顺序：

第一步：只读审计
1. 检查 git status、最近提交和仓库结构；
2. 找出 flat、rough、MultiTerrain、teacher v3、student、evaluation、export 和 deploy 入口；
3. 对照 episode CSV、checkpoint manifest、视频 manifest 和 README，确认每个结果的真实证据；
4. 输出“已完成、证据位置、未完成、风险、推荐下一个最小 milestone”；
5. 不修改文件，不运行训练、仿真、测试、构建、安装或长时间脚本。

第二步：优先提出 P0 评估协议 milestone
- 建立统一 episode-level schema；
- 分离 training seed、scenario seed、candidate screen 和 final holdout；
- 记录 terrain_type、terrain_level、command_profile、termination_reason、gait_gate、checkpoint、commit 和 config hash；
- 确保 aggregate 指标可以从 CSV 重新计算；
- 确保失败样本不会被过滤；
- 先列出拟修改文件和每个文件的理由，等待我确认后再修改。

第三步：后续按以下顺序推进
P1：选择 pyramid stairs 或 boxes 做单地形受控观测/reward/command 消融；
P2：做 MultiTerrain terrain-balanced sampler 和 curriculum 对照；
P3：重新分析 teacher-student 失败，先验证数据合同，再比较 action imitation、command conditioning、terrain conditioning、teacher routing 和 PPO fine-tuning；
P4：建立 PyTorch/JIT/ONNX Runtime/C++ runtime 数值一致性和 MuJoCo sim2sim 评测；
P5：只有 sim2sim 和安全门禁通过后，才讨论真实机器人。

每个实验必须记录：
- 假设；
- baseline；
- 对照组；
- training seed；
- scenario seed；
- terrain/type/level；
- command profile；
- checkpoint 和 SHA-256；
- git commit；
- 配置 hash；
- episode-level CSV；
- success/timeout/termination；
- velocity error/RMSE；
- gait gate；
- 视频索引；
- 失败分析；
- 是否可写入简历。

现在只执行第一步：只读审计并输出报告。不要修改任何文件，不要运行项目代码。
```

## 15. 参考资料

- 个人仓库：<https://github.com/kxj0258/unitree_rl_lab>
- Unitree 官方上游：<https://github.com/unitreerobotics/unitree_rl_lab>
- Isaac Lab：<https://isaac-sim.github.io/IsaacLab/>
- 项目导学材料：`C:\work_kxj\soft_vs_code\outputs\resume_jd\导学-宇树强化学习.md`
- 项目面试材料：`C:\Users\kxj\Desktop\daily_kxj\秋招\项目经历深度介绍与面试问答.md`
