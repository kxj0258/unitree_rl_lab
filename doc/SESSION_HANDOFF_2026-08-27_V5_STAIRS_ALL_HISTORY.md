# Unitree Go2 RL 全历史接管：v5 Foundation 与 Stairs Specialist（截至 2026-08-27）

## 1. 本文件地位与永久约束

这是当前最新主接管文件。它继承并扩展：

`doc/SESSION_HANDOFF_2026-08-26_HEIGHTSCAN_DEMO_ALL_HISTORY.md`

新会话必须先完整读取本文件，再按第 12 节顺序按需读取证据。不要仅凭聊天摘要猜测状态，
也不要重复已完成的训练、评估或录制。

永久约束：

- 禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；
- 禁止覆盖、移动或删除任何 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果或视频；
- 新运行必须使用尚不存在的 run name、日志路径和输出目录；
- 训练、评估和视频录制只能由助手给出完整命令，用户在终端运行并回传输出；助手不得自行启动；
- 安装、克隆、联网下载或扩大项目范围前必须取得用户明确授权；
- 不清理 dirty worktree，不 reset/checkout 用户修改；
- 不修改或删除 `env_backup_unitree_rl.yml`；
- 始终严格区分 training seed 与 evaluation scenario seed；
- 单一 training seed 配合多个 evaluation scenario seeds 不得描述为多训练 seed；
- timeout 不等于有效运动，必须结合 tracking、command-aligned progress、位移、stall 和 termination；
- 不夸大 stairs、boxes 或七地形泛化能力，不宣称实机部署。

本地读取、HEAD/status、日志、checkpoint、TensorBoard、CSV/JSON、视频 metadata 和 SHA-256
等只读检查已经授权。

## 2. 仓库与工作区

- Unitree RL Lab：`/home/kxj/rl_ws/unitree_rl_lab`；
- Unitree RL Lab HEAD：`247f48bdc6ab6648ac08981fd2d076fab81e90a3`；
- IsaacLab：`/home/kxj/rl_ws/IsaacLab`；
- IsaacLab HEAD：`39b494e6d8ae3f7adbb30ae1feaf45d21bdc7d19`；
- 2026-08-27 接管时 IsaacLab clean；
- Conda 环境：`unitree_rl`；
- Unitree RL Lab 是有意保留的 dirty worktree。

每次新会话先执行只读检查：

```bash
cd /home/kxj/rl_ws/unitree_rl_lab
git status --short
git log -5 --oneline
git rev-parse HEAD
git -C /home/kxj/rl_ws/IsaacLab status --short
git -C /home/kxj/rl_ws/IsaacLab rev-parse HEAD
```

当前已修改或新增的关键文件包括（不能擅自丢弃）：

- `scripts/rsl_rl/train.py`；
- `scripts/project/evaluate_go2.py`；
- `scripts/project/evaluate_go2_v5_foundation_candidates.sh`；
- `scripts/project/train_go2_v5_foundation.sh`；
- `scripts/project/train_go2_v5_stairs_specialist.sh`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/v5_foundation_env_cfg.py`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/v5_stairs_specialist_env_cfg.py`；
- 旧 HeightScan、diagnostics、evaluator、recorder、composer 等 dirty 修改；
- `env_backup_unitree_rl.yml` 是用户文件，必须原样保留。

## 3. 已完成的历史成果

项目早期已完成 flat baseline、rough 50/50、rough 多 training-seed、七地形 MultiTerrain、
survival/tracking-aware 对照、HeightScan、固定分层评估和视频诊断。详细事实与路径见旧主 handoff。

仍需保留的正式历史成果：

- Flat 正式 `model_2999.pt`，SHA-256
  `a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928`；
- Rough seed-42 正式 `model_2900.pt`，SHA-256
  `9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889`；
- 无 HeightScan MultiTerrain baseline `model_3300.pt`，SHA-256
  `5756f08ca4df34d71bdd5d25fa3763b91c96c4af59f60c7598eec822322b33b3`；
- HeightScan 正式 `model_10000.pt`：
  `logs/rsl_rl/unitree_go2_velocity_multiterrain/2026-08-26_18-40-22_multiterrain_v4_heightscan_progressgate_fixedcmd_low_lr1e5_continue_to10000_from6000_seed42_env1024_addi4001/model_10000.pt`；
- HeightScan `model_10000.pt` SHA-256：
  `5f8768907c71141ea7c588f7b27b3248081daa8d00b7bcda31e8a1137faf2956`；
- 最终 34 秒作品集 Demo：
  `evaluation/portfolio_go2_heightscan_model10000_level5_vx035_assetfollow_selected4_34s_2026-08-26.mp4`；
- Demo SHA-256：
  `90a4909bb527eea70034e699cd9f3a81a1f119e619772309b0a06845eb2de887`。

HeightScan 的准确结论仍是：flat 稳定；rough/slope 有有效运动但 tracking error 明显；stairs、
boxes 和 inverted terrain 仍存在保守、缓慢、间歇停滞。该结果适合作为原型与失败诊断，
不能描述为七种复杂地形均成功跟踪。

## 4. 新架构路线与理由

用户明确接受从零重构和重新训练，不要求围绕旧七地形 checkpoint 做短期修补。讨论相关开源
腿式机器人项目后，选择高收益路线：先训练稳定、可感知、循环式通用 foundation，再做结构化
障碍 specialist，最后才考虑融合/蒸馏。核心思想与常见成功路线一致：

1. 不让一个早期单策略同时解决七类难度跨度很大的地形；
2. 先在连续、可渐进的地形上学会稳定速度跟踪与地形感知；
3. 对 stairs/boxes 使用专项 curriculum、采样和奖励；
4. 以固定分层评估挑选 checkpoint，而非默认最后 checkpoint 最好；
5. 后续若融合，优先考虑 specialist mixture、teacher/student 或蒸馏，而不是继续盲目加 iteration。

当前 Stage A 为四地形 foundation；Stage B 首先训练 stairs specialist。Boxes specialist 尚未实现。

## 5. v5 Foundation 实现

注册 task：`Unitree-Go2-Velocity-V5-Foundation`。

设计：

- 地形：flat、random rough、pyramid slope、inverted pyramid slope，平衡采样；
- actor：45 维 proprioception + 117 维 height scan = 162 维；
- critic：177 维 privileged observation；
- actor/critic：GRU，hidden dim 256，1 layer；
- actor/critic observation normalization 开启；
- PPO：1024 env，32 steps/env/iteration，LR `3e-4` adaptive；
- command-aligned progress reward、commanded stall penalty；
- tracking + command progress 门控 terrain curriculum；
- training seed 42，从随机初始化训练；
- 5000 iterations = 163,840,000 transitions。

Foundation smoke：128 env × 5 iterations，exit code 0，无非有限值。

Foundation 正式 run：

`logs/rsl_rl/unitree_go2_velocity_v5_foundation/2026-08-26_23-13-40_multiterrain_v5_perceptive_gru_foundation_balanced4_fromscratch_seed42_env1024_i5000`

- training seed：42；
- exit code：0；
- 5000 TensorBoard iterations；
- 14 个主要训练标量，non-finite 0；
- 总时长：03:22:12；
- 最终 reward 53.95；episode length 1000；
- curriculum level 4.6041；
- XY error 0.0706；yaw error 0.1437；
- timeout 0.9932；base contact 0；bad orientation 0.0068；
- GPU 平均利用率约 80%，实际采样峰值显存 6935/12227 MiB，最高温度 69°C；
- 曲线约在 2000--3000 iterations 后进入平台，未继续到 8000/10000。

关键 checkpoint：

| checkpoint | SHA-256 |
|---|---|
| `model_2100.pt` | `c1eb927ca1931119e851a0fef7b8294d22a983df1ff596d879b491594d5d1a2c` |
| `model_3200.pt` | `c23c592eb9401c0173e470672cfb67aa1bf12a2074cb8fbe93a438f2c730ed08` |
| `model_4999.pt` | `514694386e332de00f9421013e2f2cae4a1f5c520abca52cdc37cdf272cc9f8d` |

## 6. v5 Foundation 固定分层评估

Evaluator 新增 `v5_foundation_stratified`：4 terrains × 8 levels，32 env × 2 episodes =
64 episodes/checkpoint。修复了 recurrent evaluation 的 hidden-state reset：每次 env step 后调用
`runner.alg.policy.reset(dones)`。

评估 checkpoint：2100、3200、4999；evaluation scenario seeds：7、42、123；共 9 次评估、
576 episodes。注意：training seed 始终只有 42，7/42/123 是 evaluation scenario seeds。

三 seed 聚合：

| checkpoint | XY error | yaw error | timeout | bad orientation | power |
|---|---:|---:|---:|---:|---:|
| model2100 | 0.055468 ± 0.003020 | 0.084825 ± 0.002742 | 0.994792 | 0.005208 | 55.53 |
| model3200 | **0.050854 ± 0.001070** | **0.081719 ± 0.000636** | **1.000000** | **0** | 50.879 |
| model4999 | 0.053259 ± 0.000314 | 0.088938 ± 0.000389 | 1.000000 | 0 | **40.396** |

正式 foundation 选择 `model_3200.pt`，而不是 final checkpoint。其四地形聚合：

- flat XY `0.046612 ± 0.002277`，timeout 100%；
- random rough XY `0.055681 ± 0.002372`，timeout 100%；
- pyramid slope XY `0.051779 ± 0.000989`；
- inverted pyramid slope XY `0.049344 ± 0.001597`。

评估目录：

`evaluation/v5_foundation_stratified_model{2100,3200,4999}_trainseed42_evalseed{7,42,123}_2026-08-27`

## 7. Stage B Stairs Specialist 实现

注册 task：`Unitree-Go2-Velocity-V5-Stairs-Specialist`。

设计：

- 15% flat rehearsal；
- 15% random rough rehearsal；
- 35% ascending pyramid stairs；
- 35% descending/inverted pyramid stairs；
- 8 difficulty rows，20 terrain columns；
- stair height `0.03--0.14 m`，step width `0.30 m`；
- fixed 20 s command，x `0.20--0.60 m/s`，y `±0.10`，yaw `±0.40`；
- tracking curriculum threshold 0.55，minimum progress speed 0.15；
- 加入 swing-foot-height 与 feet-stumble 项；
- 降低 flat orientation 和 joint-position 约束以允许楼梯步态；
- 保持 foundation action scale `0.25`，确保迁移语义连续；
- PPO LR `1e-4` adaptive；
- 从正式 foundation `model_3200.pt` 迁移 policy、GRU、normalization；
- 第一次 specialist 训练重置 optimizer 和 iteration。

`scripts/rsl_rl/train.py` 新增：

- `--resume_path`：允许精确的跨 experiment checkpoint；
- `--reset_optimizer_on_resume`；
- `--reset_iteration_on_resume`。

`scripts/project/train_go2_v5_stairs_specialist.sh`：

- 强制 SHA-256 校验；
- 拒绝复用 run name、console/GPU log；
- 支持 foundation transfer 模式；
- 支持 `CONTINUE_CHECKPOINT` + `EXPECTED_CONTINUE_SHA256` 的真正续训模式；
- continuation 模式加载 optimizer、保留 checkpoint iteration，不添加 reset 参数。

## 8. Stairs smoke 结果与 action scale 修正

第一次 smoke 使用 action scale `0.35`：128 env × 5，exit code 0，无 NaN/Inf，但末端
bad orientation 11.67%。原因是相对 foundation `0.25` 放大了 40% 关节目标，破坏迁移语义。
该 smoke 仅作负诊断，不得继续。

修正版 action scale `0.25` smoke：

`logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_11-18-51_v5_perceptive_gru_stairs_specialist_smoke_actionscale025_from_foundation3200_trainseed42_env128_i5`

- training seed 42；128 env × 5；exit code 0；
- optimizer 未加载，specialist iteration 从 0；
- 37 TensorBoard scalar tags，non-finite 0；
- mean reward 4.60；XY error 0.0100；
- base contact 0；bad orientation 3.20%；
- 明显优于 action scale 0.35 smoke，正式训练采用 0.25。

## 9. Stairs 正式训练、中止点和完整核验

用户将预算从 4000 改为 5000。原始正式 run：

`logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_11-22-47_v5_perceptive_gru_stairs_specialist_formal_from_foundation3200_trainseed42_env1024_i5000`

外部日志：

- Console：`/home/kxj/rl_ws/unitree_go2_v5_perceptive_gru_stairs_specialist_formal_from_foundation3200_trainseed42_env1024_i5000_2026-08-27_11-22-41.log`；
- GPU CSV：`/home/kxj/rl_ws/gpu_v5_perceptive_gru_stairs_specialist_formal_from_foundation3200_trainseed42_env1024_i5000_2026-08-27_11-22-41.csv`。

身份：

- task variant：`v5_stairs_specialist`；
- training seed：42；
- evaluation scenario seed：N/A；
- source：formal foundation `model_3200.pt`；
- optimizer fresh，specialist iteration 从 0；
- 1024 env，32 steps/iteration，目标 5000 iterations。

用户在 terminal 显示 iteration 2240 后按一次 Ctrl-C。wrapper 最终显示 exit code 0，进程已完全
退出，无残留训练进程。TensorBoard 实际记录到 iteration 2244；最近完整保存 checkpoint 是
`model_2200.pt`，因此只丢失 44 次未保存更新。

中止前 iteration 2240：

- reward 56.42；episode length 987.22；
- track lin 2.3015；track yaw 0.6100；command progress 0.4106；stall -0.0054；
- curriculum level 4.6607；
- XY error 0.0693；yaw error 0.3240；
- timeout 0.9707；base contact 0；bad orientation 0.0293；
- timeout 伴随高 tracking/progress，不能简单判为静止成功，但 yaw error 偏高需要后续评估；
- TensorBoard 37 scalar tags，non-finite 0。

GPU CSV：1461 samples；平均 GPU utilization 83.89%，平均显存 6968 MiB，采样峰值
11105/12227 MiB，最高 70°C，平均 power 93.71 W，无 GPU error。

最新续训来源：

`logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_11-22-47_v5_perceptive_gru_stairs_specialist_formal_from_foundation3200_trainseed42_env1024_i5000/model_2200.pt`

SHA-256：

`db25354aadfaebc66f32ab0ee42a7bff28d15b2ae1d168901f959cebedbdc345`

checkpoint 内容：`model_state_dict`、`optimizer_state_dict`、`iter=2200`、`infos=None`；optimizer
包含 25 个 state entries。环境状态、episode 状态、GRU rollout hidden state 和 terrain curriculum
运行时分布不在 checkpoint 中。

恢复影响：

- 策略/GRU 权重、normalization、optimizer 会完整恢复；
- GRU 临时 hidden state 清零，通常一两个 rollout 内重建；
- 仿真环境重新生成，正在进行的 episode 丢失；
- curriculum 会从配置的初始低等级重新开始，而非立即回到平均 4.66；
- 已学策略能力不丢失，预计较快重新爬升，但最终轨迹不会与不中断训练逐 bit 相同；
- 恢复后必须重点看前 100--200 iterations 的 reward、tracking、bad orientation 和 curriculum。

## 10. 当前唯一下一步：从 model_2200 续训到总计 5000

截至本文件生成时，续训尚未启动。不要重复 foundation、smoke 或从零正式 stairs 训练。

RSL-RL 的 `max_iterations` 在 resume 时表示“额外 iterations”，不是目标总数。checkpoint 从
2200 开始，要使循环覆盖 2200...4999，必须设置 `MAX_ITERATIONS=2800`。若错误设置 5000，
会训练到约 7200。

用户应运行：

```bash
cd /home/kxj/rl_ws/unitree_rl_lab
source /home/kxj/miniforge3/etc/profile.d/conda.sh
conda activate unitree_rl

CONTINUE_CHECKPOINT=/home/kxj/rl_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_11-22-47_v5_perceptive_gru_stairs_specialist_formal_from_foundation3200_trainseed42_env1024_i5000/model_2200.pt \
EXPECTED_CONTINUE_SHA256=db25354aadfaebc66f32ab0ee42a7bff28d15b2ae1d168901f959cebedbdc345 \
NUM_ENVS=1024 \
SEED=42 \
MAX_ITERATIONS=2800 \
RUN_STAGE=stairs_specialist_formal_continue_from2200_to5000 \
V5_STAIRS_RUN_NAME=v5_perceptive_gru_stairs_specialist_formal_continue_from2200_trainseed42_env1024_addi2800_total5000 \
scripts/project/train_go2_v5_stairs_specialist.sh
```

启动时必须确认：

```text
Resume optimizer loaded: True
New-run starting iteration: 2200
Learning iteration 2200/5000
```

任一不一致应立即中止。注意 RSL-RL 会在新 run 中再次使用 `model_2200.pt` 文件名，但新目录
不会覆盖原 checkpoint；这是 iteration 计数语义造成的正常现象。

## 11. 续训完成后的预定工作

不要默认 `model_4999.pt` 最优。续训完成后：

1. 识别 continuation run name、training seed 42、source checkpoint model_2200、目录和时间；
2. 检查 exit status、checkpoint 序列、最后 checkpoint、TensorBoard、console/GPU log；
3. 检查 resume 初期 curriculum 从低 level 回升到约 4.6 的速度；
4. 检查 reward、XY/yaw tracking、command progress/stall、termination、吞吐和非有限值；
5. 将原始 run 与 continuation run 的 iteration 轴按 checkpoint iteration 连接，不能把它们当两次
   独立 training seeds；
6. 从原始和 continuation checkpoint 中按曲线选候选，建议至少覆盖 `model_1200`、`model_2200`、
   continuation 中期和 `model_4999`，实际候选应基于完整 TensorBoard 决定；
7. 实现 stairs 专用固定分层 evaluation protocol，至少分 ascending/descending stairs、difficulty
   levels、多个 evaluation scenario seeds；
8. 有效运动判据必须包含 command-aligned progress/位移/stall，不以 timeout 单独判成功；
9. 选出 stairs specialist 正式 checkpoint 后，才讨论录制或 boxes specialist；
10. 不重复 v4 HeightScan 作品集录制，除非用户明确要求新作品。

## 12. 新输出分析协议与证据读取顺序

每次收到新输出先识别：

- task 与 task variant；
- run name；
- training seed；
- evaluation scenario seed（训练时应为 N/A）；
- source checkpoint 和当前 checkpoint；
- checkpoint SHA-256；
- log/evaluation/video 目录与时间；
- terrain protocol/level、command、env/episode 数。

训练检查：exit status、checkpoint 连续性、checkpoint 内 iter/optimizer、TensorBoard tag/step、
console/GPU log、tracking、progress/stall、curriculum、termination、吞吐、GPU 状态、NaN/Inf。

评估检查：episode 数、CSV/JSON 一致性、terrain/level 覆盖、command 分布、tracking、有效位移、
stall、termination、checkpoint hash 和非有限值。

视频检查：clip 数、FPS/frames/分辨率/时长、主体可见性、metadata/CSV、termination、SHA-256。

关联证据读取顺序：

1. 本文件；
2. `doc/SESSION_HANDOFF_2026-08-26_HEIGHTSCAN_DEMO_ALL_HISTORY.md`；
3. `doc/SESSION_HANDOFF_2026-08-26_ALL_HISTORY.md`；
4. 当前 v5 foundation/stairs 的 env、PPO、train/evaluate scripts；
5. 当前 v5 log directories、TensorBoard、checkpoint 和外部 console/GPU logs；
6. `doc/PROJECT_STATUS.md`；
7. `doc/RESULTS_MULTITERRAIN_CONTINUATION_2026-08-26.md`；
8. `doc/RESULTS_GO2_DEMOS_2026-08-26.md`；
9. `doc/COMMAND_SCRIPT_AUDIT_2026-08-26.md`；
10. `doc/REPRODUCTION.md` 与 `doc/RESUME_PROJECT.md`；
11. `doc/repro/go2_multiterrain_stage2_final_artifacts_2026-08-26.sha256`；
12. 只有需要早期细节时再读 2026-08-24/25 handoff 和对应 RESULTS。

本文件综合本次可见对话、旧 handoff、当前代码、日志、TensorBoard、checkpoint、评估与视频
证据。不可见且从未落盘的历史对话无法逐字恢复，应以本地 artifact 和 SHA-256 为最终依据。

## 13. 新对话启动提示词

将第 13 节之后由当前助手提供的启动提示词原样粘贴到新对话。新助手必须先总结接管状态，
不得在尚未收到用户新输出时擅自启动训练、评估或录制。
