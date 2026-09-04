# Unitree Go2 项目主接管 V7：当前 Demo 与七地形 Teacher–Student 总路线

日期：2026-08-28（Asia/Shanghai）

这是本项目截至当前的最新主入口。它以 V6 为完整历史基线，并记录 V6 之后完成的 Demo v1、长期
决策和下一步。新会话必须从本文件第一行读到末尾，再按第 11 节顺序读取证据；不得仅凭聊天摘要
行动。

## 1. 永久约束

- 禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；
- 禁止覆盖、移动或删除 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果或视频；
- 训练、评估、录制和视频合成必须由助手给出完整命令、用户在终端执行并回传输出；助手不得启动；
- 安装、克隆、联网下载或扩大范围前必须取得用户明确授权；
- 不清理现有 dirty worktree；保留所有用户改动；
- 不修改或删除 `env_backup_unitree_rl.yml`；
- 不重复已经完成的训练、评估、录制或合成；
- 始终严格区分 training seed 与 evaluation scenario seed；
- timeout 只表示达到 episode 时间上限，不等于有效运动或任务成功；
- 本地读取、日志/TensorBoard/checkpoint metadata、目录盘点、视频 metadata 和 SHA-256 等只读操作
  已授权。

## 2. 仓库与环境身份

- 项目：`/home/kxj/rl_ws/unitree_rl_lab`；
- Unitree RL Lab HEAD：`247f48bdc6ab6648ac08981fd2d076fab81e90a3`；
- worktree 有大量有意保留的 modified/untracked 文件，不得清理；
- IsaacLab：`/home/kxj/rl_ws/IsaacLab`；
- IsaacLab HEAD：`39b494e6d8ae3f7adbb30ae1feaf45d21bdc7d19`，历史检查为 clean；
- Conda：`/home/kxj/miniforge3/envs/unitree_rl`；
- RSL-RL：3.1.2；
- 本文件生成前检查没有残留训练、评估、录制或 Isaac 进程。

## 3. 全历史成果链

### 3.1 HeightScan 历史原型

- 历史七地形 HeightScan `model_10000.pt`；
- SHA-256：`5f8768907c71141ea7c588f7b27b3248081daa8d00b7bcda31e8a1137faf2956`；
- 最终 34 秒四地形作品集：
  `evaluation/portfolio_go2_heightscan_model10000_level5_vx035_assetfollow_selected4_34s_2026-08-26.mp4`；
- 视频 SHA-256：`90a4909bb527eea70034e699cd9f3a81a1f119e619772309b0a06845eb2de887`。

HeightScan 是历史原型和作品集证据，不能冒充后续 v5 的统一最终策略。

### 3.2 v5 foundation

- 正式 checkpoint：`model_3200.pt`；
- training seed：42；
- SHA-256：`c23c592eb9401c0173e470672cfb67aa1bf12a2074cb8fbe93a438f2c730ed08`；
- 四种连续地形 foundation：flat、random rough、pyramid slope、inverted pyramid slope；
- 使用 evaluation scenario seeds 7/42/123 的固定分层评估。

### 3.3 stairs specialist

- training seed 始终为 42；evaluation seeds 7/42/123 不是训练 seed；
- 原正式 run TensorBoard 到 iteration 2244，但当时最新完整 checkpoint 是 `model_2200.pt`；
- `model_2200.pt` 的 SHA-256 与 optimizer state 已保存；
- 恢复训练必须加载 optimizer 并保留 iteration；curriculum state 未被 checkpoint 完整保存，恢复时会
  重新建立；GRU hidden state 是 rollout runtime state，不作为跨进程 episode state 恢复；
- 已正确从 model2200 追加 2800 iterations，使总计到 5000，而不是错误地再追加 5000；
- 最终综合 stairs `model_4999.pt`；
- SHA-256：`47b42309c03450c88f27ceaf66ed393abfcc5a488e7a508a369055ed4cfcee50`。

### 3.4 ascending remediation

发现 model4999 在 inverted-pyramid stairs 中会出现局部速度跟踪良好但净进展差、坑底绕行/停滞的
问题，因此不能用 timeout 或 path length 宣称成功。

- remediation 从 model4999 迁移 policy、GRU、normalization，但因目标分布改变而重置 optimizer 和
  iteration；
- 正式训练 1200 iterations、1024 env、training seed 42，exit code 0；
- `model_900.pt` 是当前最佳 heading-locked ascending specialist；
- model900 SHA-256：`57c66a27fb3be566ad28edec6e567bfc1f8eddea8db3115e0ed8009c9f38d1a3`；
- `model_1199.pt` 固定航向略强但随机命令和 hard-stairs 安全性退化，已拒绝作为综合模型；
- 不继续窄 remediation 到 3000。

model900 level 7 视频证明真实爬升：净前进约 7.22 m、高度增益约 0.808 m、stall 0、base contact 0、
bad orientation 0。视频 SHA-256：
`ff4192b91068c8a1725017078718647ae412ba64dcbe3db4577a47dcdbceb94e`。

## 4. 当前模型职责

1. HeightScan model10000：历史七地形原型与旧作品集；
2. v5 foundation model3200：四种连续地形正式 foundation；
3. stairs model4999：综合随机命令 stairs 正式模型；
4. remediation model900：heading-locked ascending specialist；
5. remediation model1199：过拟合诊断 checkpoint，不采用。

这些策略尚未融合。不得声称当前已有单一七地形统一策略。

## 5. Track A 已完成：当前成果 Demo v1

正式成片：

`evaluation/portfolio_go2_current_results_multicheckpoint_demo_v1_2026-08-28.mp4`

- composition exit code 0；
- 1280x720，50 FPS，2350 frames，47.0 s；
- 24,972,117 bytes；
- SHA-256：`3199ab908f2e7b1bd8091ccdb9c7281154b3eeb483d78325ff2851c7bcaa217f`；
- JSON sidecar 保存所有 source path/hash/trim；
- 成片 sidecar 与独立 manifest 均通过；
- 明确标注 multi-checkpoint showcase，不是 unified policy。

叙事顺序：四类 HeightScan 连续地形 → model4999 pyramid stairs 成功 → model4999 inverted stairs
失败诊断（净前进 0.89 m、stall 42.2%）→ model900 在更难 level 7 修复（净前进 7.22 m、stall 0）
→ 七 teacher/统一 student 路线。

视觉 QA 无黑帧、截断和编码异常。已知小瑕疵：前四段取自已有 overlay 的 34 秒作品集，叠加统一
banner 后顶部有轻微文字重影。当前文件固化为审计版 v1，不得覆盖；简历精修版必须另存 v1.1。

结果说明：`doc/RESULTS_CURRENT_MULTICHECKPOINT_DEMO_V1_2026-08-28.md`。

证据清单：`doc/repro/current_multicheckpoint_demo_v1_2026-08-28.sha256`；其自身 SHA-256 为
`f63dc75e33c31ac0698609e87c964d79ce41dd4e33807c6e87496ef1f81746eb`。

## 6. 用户批准的完整项目目标

最终目标是一个单一 recurrent perceptive student，在以下七类地形上实现可靠速度跟踪，并在同一
episode 的混合地形 transitions 上通过评估和形成最终 Demo：

1. flat；
2. random rough；
3. pyramid slope；
4. inverted pyramid slope；
5. pyramid stairs；
6. inverted pyramid stairs；
7. boxes。

总路线以 `doc/PROJECT_MASTER_GOAL_7TERRAIN_TEACHER_STUDENT_2026-08-28.md` 为准，后期不得遗忘或
用临时 fine-tuning 替换。

## 7. Track B：七 Teachers 到统一 Student

用户已批准：七种地形各自在独立 namespace 从随机初始化开始训练 teacher，选择每个地形的最优
策略；随后统一 student 也从随机初始化开始，使用平衡数据训练。

具体优化路线：

1. 所有 teachers 共享完全一致的 action semantics、command schema 和 student-visible observation
   schema；允许 teacher 在训练时使用 privileged information，但 student 输入不得依赖 terrain ID；
2. 七个 terrain families 按 1/7 平衡，并同时平衡 terrain level、command bins 与 recovery states；
3. 使用在线 multi-teacher distillation：student actions 驱动环境，按环境的 terrain family 在训练端
   选择相应 teacher label；terrain ID 只用于 dispatcher，不作为 student observation；
4. 蒸馏后进行 balanced PPO consolidation，并保留 teacher regularization/rehearsal 防止遗忘；
5. 最后加入同 episode composite mixed-terrain transitions；
6. 固定分层评估并使用多个 evaluation scenario seeds；teacher 的科学“最优”结论最终需要多个
   training seeds，不能只凭 seed42 或最后 checkpoint；
7. 保留 shared-foundation teacher 分支仅作为 ablation，不做权重平均，不把七个 checkpoint 直接合并。

本地 RSL-RL 3.1.2 已确认有 `DistillationRunner`、`Distillation`、`StudentTeacher`、
`StudentTeacherRecurrent`。原生实现是单 teacher、student actions 访问状态、纯 behavior loss；本项目
仍需实现 multi-teacher dispatcher 和 PPO+BC 联合目标，不能声称一个配置开关已经支持完整方案。

## 8. 下一步（尚未执行）

首要工作不是立即开始长训练，而是建立 Track B 的共享契约和最小闭环：

1. 审计并冻结七 teachers 的 action/command/student-observation schema；
2. 建立独立 experiment/task/run 命名和 artifact manifest 规则；
3. 实现 flat teacher 的 from-scratch 配置和训练包装脚本；
4. 先做静态检查，再由用户执行 smoke；
5. smoke 通过后由助手给出正式 flat teacher 命令；
6. 固定分层评估和 checkpoint 选择；
7. 再依次扩展其余六种地形；
8. teachers 达标后开发 multi-teacher distillation/student；
9. 最后 PPO consolidation、composite terrain 与统一 Demo。

尚未授权或运行任何 Track B 新训练。助手不得自行启动。

可选的近期展示工作是另做 Demo v1.1 消除 overlay 重影；它不能覆盖审计版 v1，也不应阻塞 Track B。

## 9. 收到新运行输出时的分析协议

每次先识别并复述：task、task variant、run name、training seed、evaluation scenario seed、source
checkpoint、current checkpoint、绝对目录和时间。

训练输出检查：

- exit status；checkpoint 连续性及 SHA-256；
- checkpoint iteration 与 optimizer state；
- TensorBoard iterations/scalars/non-finite；
- console 与 GPU CSV；
- tracking error、command-aligned progress、stall；
- curriculum、termination、吞吐、ETA/stall；
- GPU utilization/memory/temperature/power；
- NaN/Inf 和其他异常。

评估输出检查：terrain/level/command 覆盖、episodes、净位移、command-aligned progress、path length、
stall、tracking、termination、安全和多个 evaluation scenario seeds。视频检查 metadata、CSV、帧率、
帧数、时长、源 checkpoint/hash 和视觉任务完成度。timeout 永远不能代替有效进展。

## 10. 已完成事项，禁止重复

- HeightScan model10000 训练与 34 秒作品集；
- v5 foundation model3200 及三 evaluation-seed 固定分层评估；
- stairs 从 model2200 精确加载 optimizer/iteration 并仅追加 2800 到总 5000；
- stairs model4999 评估与 directional diagnostics；
- ascending-remediation smoke、1200 正式训练、model600/900/1199 评估；
- model900/model1199 原协议 evaluation seeds 7/42/123；
- model900 evaluation seed42 heading-locked level7 视频；
- 当前多 checkpoint Demo v1 合成。

不要续训窄 remediation 到 3000，不要用 model1199 替代 model4999，不要重新合成同名 Demo v1。

## 11. 新会话强制读取顺序

1. 本 V7 主接管文件，从头读到文件末尾；
2. `doc/repro/session_handoff_v7_current_demo_and_master_route_2026-08-28.sha256`；
3. `doc/PROJECT_MASTER_GOAL_7TERRAIN_TEACHER_STUDENT_2026-08-28.md`；
4. `doc/SESSION_HANDOFF_2026-08-28_V6_ASCENDING_REMEDIATION_ALL_HISTORY.md`；
5. `doc/RESULTS_CURRENT_MULTICHECKPOINT_DEMO_V1_2026-08-28.md`；
6. `doc/repro/current_multicheckpoint_demo_v1_2026-08-28.sha256`；
7. V6 指定的 V5 stairs、HeightScan handoff 与各自 manifest；
8. 按当前工作需要读取 env/PPO/curriculum/train/evaluate/record/compose 代码；
9. 按需读取 TensorBoard、checkpoint metadata/optimizer、console/GPU、evaluation summaries、video
   metadata/CSV 和 SHA-256；
10. 开始任何工作前只读检查 Unitree/IsaacLab HEAD 与 dirty 状态、关键证据和残留进程。

## 12. 新会话首次回复应确认的状态

新助手应先总结而不是立即给训练命令，并明确：

- 历史 HeightScan model10000 与 34 秒 Demo；
- v5 foundation model3200 与 evaluation scenario seeds 7/42/123；
- stairs training seed42、model2200 恢复语义和最终 model4999；
- model900 是 ascending specialist，model1199 被拒绝；
- Demo v1 已完成、SHA-256、multi-checkpoint 定位与 overlay 小瑕疵；
- 最终目标是七个 from-scratch teachers 和 from-scratch unified student；
- 当前下一步是 schema/namespace/flat-teacher smoke 设计，而不是自行启动训练。

之后等待用户发运行输出，按第 9 节分析并指导后续工作。
