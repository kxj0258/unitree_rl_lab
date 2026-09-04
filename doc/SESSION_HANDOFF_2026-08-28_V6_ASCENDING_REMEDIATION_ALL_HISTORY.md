# Unitree Go2 RL 主接管 V6：Ascending Remediation 与当前全状态（截至 2026-08-28）

## 1. 文件地位与永久约束

这是 2026-08-28 起最新主接管文件，继承且不覆盖：

1. `doc/SESSION_HANDOFF_2026-08-27_V5_STAIRS_ALL_HISTORY.md`；
2. `doc/SESSION_HANDOFF_2026-08-26_HEIGHTSCAN_DEMO_ALL_HISTORY.md`；
3. 更早 handoff 与 results。

新会话必须从头读到本文件末尾，再按第 12 节读取证据。永久约束：

- 不升级/重装 NVIDIA、Isaac Sim、Isaac Lab、PyTorch；
- 不覆盖、移动或删除 checkpoint、日志、GPU CSV、TB event、评估、视频；
- 训练/评估/录像由助手给完整命令，用户运行，助手不得自行启动；
- 安装、克隆、下载、扩大范围前取得用户明确授权；
- 不清理 dirty worktree；不修改/删除 `env_backup_unitree_rl.yml`；
- training seed 与 evaluation scenario seed 必须严格区分；
- timeout 不等于有效运动；
- 不重复已完成的训练、评估或录像；
- 不宣称已完成七地形统一策略或实机部署。

## 2. 仓库与工作区

- Unitree RL Lab：`/home/kxj/rl_ws/unitree_rl_lab`；
- HEAD：`247f48bdc6ab6648ac08981fd2d076fab81e90a3`；
- dirty worktree，含所有实验代码与文档，必须保留；
- IsaacLab：`/home/kxj/rl_ws/IsaacLab`；
- HEAD：`39b494e6d8ae3f7adbb30ae1feaf45d21bdc7d19`；
- 2026-08-28 检查 IsaacLab clean；
- Conda：`unitree_rl`。

新增/修改的 remediation 关键文件：

- `source/.../go2/v5_stairs_ascending_remediation_env_cfg.py`；
- `source/.../mdp/curriculums.py`；
- `source/.../agents/rsl_rl_ppo_cfg.py`；
- `source/.../go2/__init__.py`；
- `scripts/project/train_go2_v5_stairs_ascending_remediation.sh`；
- `scripts/project/evaluate_go2.py`；
- `scripts/project/record_go2_multiterrain_demo.py`；
- `scripts/project/record_go2_v5_stairs_ascending_model900_level7.sh`。

## 3. 历史正式成果

- Flat model2999、Rough model2900、MultiTerrain baseline model3300：详见旧 handoff；
- HeightScan 正式 model10000 SHA-256：
  `5f8768907c71141ea7c588f7b27b3248081daa8d00b7bcda31e8a1137faf2956`；
- HeightScan 34 s portfolio Demo SHA-256：
  `90a4909bb527eea70034e699cd9f3a81a1f119e619772309b0a06845eb2de887`；
- v5 foundation 正式 model3200 SHA-256：
  `c23c592eb9401c0173e470672cfb67aa1bf12a2074cb8fbe93a438f2c730ed08`；
- foundation training seed 42，evaluation scenario seeds 7/42/123。

## 4. 综合 Stairs 正式模型

原 stairs 正式 run TB 到 2244，最近完整 checkpoint 为 model2200，SHA-256
`db25354aadfaebc66f32ab0ee42a7bff28d15b2ae1d168901f959cebedbdc345`，`iter=2200`，
optimizer 25 states。

已正确从 model2200 加载 optimizer、保留 iteration，只追加 2800 iterations，使总计到 5000。
continuation exit code 0，总 transitions `91,750,400`。正式综合 stairs checkpoint：

`logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_16-17-34_v5_perceptive_gru_stairs_specialist_formal_continue_from2200_trainseed42_env1024_addi2800_total5000/model_4999.pt`

SHA-256：`47b42309c03450c88f27ceaf66ed393abfcc5a488e7a508a369055ed4cfcee50`。

model4999 通过 5 checkpoints × 3 evaluation scenario seeds 的 stairs 固定分层选择，仍是随机
commands 综合 stairs 正式模型。但 directional/heading diagnostics 证明它在 inverted stairs 会绕圈，
不能真正上楼。

## 5. Ascending Remediation 实现与训练

Task：`Unitree-Go2-Velocity-V5-Stairs-Ascending-Remediation`。

从 model4999 迁移 policy/GRU/normalization；因 objective 与分布改变，optimizer/iteration 重置。
训练 seed 42、1024 env、1200 iterations、39,321,600 transitions、LR `5e-5`。terrain 为
70% inverted stairs、20% pyramid stairs、5% flat、5% rough；固定 forward + heading feedback。
curriculum 新增 yaw-tracking gate，阻止绕圈策略依赖 body-frame progress 晋级。

正式 run：

`logs/rsl_rl/unitree_go2_velocity_v5_stairs_ascending_remediation/2026-08-27_23-01-17_v5_stairs_ascending_remediation_formal_v1_from4999_trainseed42_env1024_i1200`

exit code 0，1200 TB iterations，non-finite 0，checkpoints 0--1199 连续，optimizer state 完整。

## 6. Remediation checkpoint 结论

| checkpoint | SHA-256 | 当前身份 |
|---|---|---|
| model600 | `f8a8870cc55cc7b31d262c2cbc363ccd73c5704c5edc911d231018618545e37d` | 上楼显著改善，level7 尚弱 |
| model900 | `57c66a27fb3be566ad28edec6e567bfc1f8eddea8db3115e0ed8009c9f38d1a3` | 最佳 ascending specialist / 综合折中候选 |
| model1199 | `df7d0e769056f0dec1bb9f6fb0d5c127e2ebb66ffebda603d0ebfe8f3e003e7e` | 固定航向强，但随机 command 过拟合；拒绝综合使用 |

model900 heading-locked seed42 在 inverted stairs：净位移 `7.274 m`、yaw error `0.0527`；
level7 净位移 `7.167 m`、bad orientation 0。

随机 command 三 evaluation scenario seeds 7/42/123、共 192 episodes：model900 总体 timeout
98.44%、bad orientation 1.56%；显著改善 yaw 与 inverted displacement，但 XY error、stall、功耗
不如 model4999。因此不替换综合正式 model4999。

model1199 hard inverted levels 6--7 出现约 16.67% bad orientation，不采用。

## 7. Level-7 视频与可审计证据

目录：

`evaluation/video_v5_stairs_headinglocked_ascendingremediation_model900_trainseed42_evalseed42_inverted_level7_vx035_heading0_inityaw0_20s_2026-08-28_v1`

- training seed 42；evaluation scenario seed 42；
- model900；inverted stairs level7；vx 0.35；heading/initial yaw 0；
- z 从 `-0.4859` 到 `+0.3217 m`，增益约 `0.808 m`；
- forward displacement `7.2157 m`；path `7.2881 m`；stall 0；
- timeout 1、base contact 0、bad orientation 0；
- 1280×720、50 FPS、999 frames、19.98 s；CSV 1000 rows；
- 视频抽帧确认从坑底连续上楼并到达上层平面；
- video SHA-256：
  `ff4192b91068c8a1725017078718647ae412ba64dcbe3db4577a47dcdbceb94e`；
- tracking CSV SHA-256：
  `7fe1b2c0228b6375e495678654c5643e849bb48e7cbf8bd91b2e92af992087c0`。

## 8. 当前模型职责

1. v5 foundation model3200：四种连续地形 foundation；
2. stairs model4999：综合随机 command stairs 正式模型；
3. remediation model900：已验证 level7 上楼的 heading-locked ascending specialist；
4. remediation model1199：过拟合诊断 checkpoint，不作为正式模型；
5. HeightScan model10000：历史七地形原型与 34 s portfolio Demo。

不要声称这些已融合为一个七地形统一最终模型。

## 9. 技术路线的原始意图

V5 handoff 第 4 节已经确定路线：不让早期单策略直接同时解决七类难度跨度很大的地形；先训练
连续地形 perceptive recurrent foundation，再针对 stairs、boxes 等困难地形做 specialist，最后才
考虑：

- specialist mixture/router；
- teacher/student distillation；
- mixed-command consolidation；
- 或带 rehearsal 的统一 fine-tuning。

因此用户记忆的“在 model4999/共同 foundation 基础上做困难地形分支，最终再合并”与既定路线
基本一致。但 model4999 是 stairs 分支的综合 checkpoint，不一定应作为所有后续 specialist 的唯一
共同源；boxes 等分支可根据观测/动作兼容性从 foundation model3200 或经过验证的综合 parent 分叉。

## 10. 当前下一步与禁止误操作

第一优先级已经转为成果固化，不再续训当前窄 remediation 到 3000。当前尚未授权/执行新的训练。

用户在 2026-08-28 进一步确认长期总目标，完整、永久路线见：

`doc/PROJECT_MASTER_GOAL_7TERRAIN_TEACHER_STUDENT_2026-08-28.md`

近期先用当前多 checkpoint 成果制作诚实的 Demo v1；长期在独立 experiment namespace 中训练七个
terrain-specific from-scratch teachers，再随机初始化统一 recurrent perceptive student，按七地形
平衡采样做在线 multi-teacher distillation，并进行 balanced PPO consolidation。最终目标是单一
student 完成七地形与同 episode mixed-terrain transitions；在完成前不得宣称统一策略已完成。

近期只读/低风险工作：

1. 验证本 V6 handoff、results 与 manifest；
2. model900 heading-locked evaluation scenario seeds 7/123（尚未运行）；
3. 设计七地形最终架构与 acceptance gates；
4. 决定双模型/mixture 还是单 policy consolidation；
5. 设计 boxes specialist；
6. 制作 before/after portfolio 成片。

禁止：

- 从 model900 或 model1199 继续相同窄分布到 3000；
- 重复已完成 model900/model1199 随机 commands seeds 7/42/123；
- 重复 model900 heading seed42 或 level7 视频；
- 用 model1199 替换 model4999；
- 将 model900 描述为七地形最终模型。

## 11. 七地形尚未完成的部分

当前“七地形”历史原型来自 v4 HeightScan；v5 体系只正式完成连续四地形 foundation 与 stairs
specialist/ascending remediation。尚缺：

- boxes specialist；
- 对离散障碍/boxes 的专项 curriculum 与评估；
- 七类统一固定分层 acceptance matrix；
- specialist 合并/router/distillation；
- 合并后的 regression 与多个 evaluation scenario seeds；
- 最终统一 Demo；
- export、domain randomization、sim-to-real。

## 12. 新会话读取顺序与输出分析协议

读取顺序：

1. 本 V6 handoff；
2. `doc/PROJECT_MASTER_GOAL_7TERRAIN_TEACHER_STUDENT_2026-08-28.md`；
3. `doc/RESULTS_V5_STAIRS_ASCENDING_REMEDIATION_2026-08-28.md`；
4. `doc/repro/session_handoff_v6_ascending_remediation_2026-08-28.sha256`；
5. V5 stairs handoff；
6. HeightScan 全历史 handoff；
7. remediation/stairs/foundation env、PPO、curriculum、train/evaluate/record scripts；
8. remediation formal run TB/checkpoints/console/GPU；
9. model900/model1199 三-seed评估；
10. level7 video metadata/CSV/manifest；
11. 旧 PROJECT_STATUS、RESULTS、REPRODUCTION 与早期 handoff（按需）。

每次收到新输出先识别 task、variant、run name、training seed、evaluation scenario seed、source/current
checkpoint、SHA-256、绝对目录与时间。训练检查 exit、checkpoint/iter/optimizer、TB、console/GPU、
tracking、progress/stall、curriculum、termination、吞吐、GPU、non-finite。评估检查 terrain/level 覆盖、
command、位移/path/stall、termination、hash。视频检查帧率/帧数/时长、视觉任务完成度、metadata、
CSV 与 manifest。始终记住 timeout 不等于有效运动。

## 13. 接管摘要

项目已完成 HeightScan 七地形原型、v5 四地形 GRU foundation、综合 stairs specialist、精确 optimizer
续训、固定分层评估、绕圈漏洞诊断、ascending remediation、三-seed 回归和 level7 上楼视频验证。
当前正式综合 stairs 为 model4999；最佳 ascending specialist 为 model900；model1199 因过拟合拒绝。
近期先制作当前成果 Demo v1。长期路线已确定为七个地形 teacher 各自从零训练，再从零训练统一
student，使用平衡 multi-teacher 在线蒸馏与 PPO consolidation，最后完成同 episode mixed-terrain
七地形 Demo；不得盲目追加当前窄训练或遗忘该总目标。
