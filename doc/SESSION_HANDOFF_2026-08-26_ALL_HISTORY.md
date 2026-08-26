# Unitree Go2 RL 项目总接管（截至 2026-08-26）

## 1. 新会话接管规则

这是当前项目的主接管文件，综合本项目历次会话已经写入仓库的阶段文档，以及
2026-08-26 会话中完成的 MultiTerrain 续训、固定协议评估、动态视频和复现整理。
新会话开始后应先完整读取本文件，再按需读取第 10 节列出的证据文档；不要仅凭聊天摘要
猜测实验状态。

所有本地文件读取和只读检查均已授权，可自行连续执行并分析。训练、评估和视频录制采用
“助手给出精确命令，用户在终端执行并回传输出”的方式。未经用户针对下一项工作的明确
授权，不要安装、克隆或进行其他扩展性修改。

永久约束：

- 禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；
- 禁止覆盖、移动或删除 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果或视频；
- 新训练/评估/视频目录必须使用不存在的新名称；
- 始终严格区分 training seed 与 evaluation scenario seed；
- 不把单 training-seed 的多 evaluation-seed 结果表述为多训练 seed；
- 不宣称已经实机部署，也不夸大 MultiTerrain 的高速跟踪能力。

## 2. Git 与环境基线

- Unitree RL Lab 工作目录：`/home/kxj/rl_ws/unitree_rl_lab`；
- 本文件创建前 HEAD：`b3b7c2c653ce0c84534cd6de9f7ef961e0459ec8`；
- 对应提交：`docs: consolidate Go2 Stage-2 reproduction workflow`；
- IsaacLab：`/home/kxj/rl_ws/IsaacLab`；
- IsaacLab HEAD：`39b494e6d8ae3f7adbb30ae1feaf45d21bdc7d19`；
- Conda 环境：`unitree_rl`；
- 软件栈和兼容修改见 `doc/PROJECT_STATUS.md` 与 `doc/repro/`。

每次接管首先运行：

```bash
cd /home/kxj/rl_ws/unitree_rl_lab
git status --short
git log -5 --oneline
git rev-parse HEAD
git -C /home/kxj/rl_ws/IsaacLab status --short
git -C /home/kxj/rl_ws/IsaacLab rev-parse HEAD
```

不要擅自清理任何已有修改。

## 3. 项目目标与已完成阶段

最终目标是形成可复现、可审计、适合机器人强化学习/具身智能秋招展示的 Unitree Go2
速度跟踪项目。目前完成：

1. Flat baseline：training seed 42，2048 envs，3000 PPO iterations；
2. Rough 50/50：flat/random rough，正式训练和固定分层评估；
3. Rough 多 training-seed：42、7、123，各 3000 iterations，并完成统一评估；
4. MultiTerrain：七类地形原型、survival continuation、tracking-aware 对照、固定分层评估；
5. Flat/Rough/MultiTerrain 视频录制、动态遥测叠加、完整 SHA-256 清单；
6. 训练、评估、视频、审计、归档及 Stage-2 编排脚本已保存到 Git。

## 4. 已选模型与核心结论

### Flat baseline

- checkpoint：`logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt`；
- SHA-256：`a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928`；
- 三个 evaluation scenario seeds、384 episodes：XY error `0.0756 ± 0.0021 m/s`，timeout `99.48%`。

### Rough

- seed-42 正式选择：`model_2900.pt`，SHA-256
  `9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889`；
- 多 training-seed 固定 checkpoint 主比较采用各自 `model_2999.pt`；
- training seeds 42/7/123 的固定-checkpoint 聚合：overall XY
  `0.0956 ± 0.0009 m/s`，rough XY `0.1131 ± 0.0013 m/s`，timeout `1144/1152`；
- 这是目前最适合作为简历主结果和主 Demo 的稳健性证据。

### MultiTerrain

- 七类地形：flat、random rough、正/倒 slope、正/倒 stairs、boxes；
- training seed 始终为 42；
- 入选 survival baseline continuation `model_3300.pt`：
  `logs/rsl_rl/unitree_go2_velocity_multiterrain/2026-08-26_04-32-39_multiterrain_v3_challenging_noheightscan_baseline_continue_from2995_seed42_env1024_addi500/model_3300.pt`；
- SHA-256：`5756f08ca4df34d71bdd5d25fa3763b91c96c4af59f60c7598eec822322b33b3`；
- scenario seeds 7/42/123：XY error `0.2722 ± 0.0095 m/s`，yaw error
  `0.1838 ± 0.0113 rad/s`，timeout `79.29% ± 4.29 pp`；
- tracking-aware 分支是有效负结果，没有超过 survival baseline `model_3300`；
- `model_3494` 虽局部误差更低，但整体存活更差，所以没有选择最终 iteration。

MultiTerrain 的准确结论是：策略获得七地形覆盖并在 flat 上实现良好速度跟踪，但在 stairs、
boxes、inverted terrain 上明显保守降速。不能表述为“所有复杂地形高速精确跟踪”。

## 5. MultiTerrain 训练审计限制

Survival continuation 实际训练至 iteration 3494，并保存 checkpoint、TensorBoard 和 GPU CSV；
但训练期间执行中的 Shell 文件被修改，训练进程结束后 wrapper 读取错位命令并返回 127，预期
console log 不完整。因此该 run 可依据 checkpoint、TensorBoard、GPU CSV 和固定评估使用，
但不能宣称 console log 完整。

Tracking-aware 正式 run exit code 0，console/TensorBoard/GPU CSV/checkpoint 完整，无数值型
NaN/Inf。不要删除此前安全退出的诊断日志。

## 6. 最终动态 Demo

### MultiTerrain 七地形长版

- 目录：`evaluation/video_multiterrain_stratified_baseline_model3300_trainseed42_evalseed42_level5_vx035_20s_follow_2026-08-26`；
- 成片：`go2_multiterrain_model3300_vx035_level5_20s_follow_dynamic_demo.mp4`；
- 141.86 秒，1280x720，50 FPS，7093 帧可解码；
- SHA-256：`5b64a6722651b33977048453c57f20017c39754985c45ad13a41ccbbe9f93939`；
- 动态画面显示 command vx、actual vx、XY error、累计路径、termination 及历史曲线；
- 跟随相机已修复机器人走出画面的问题。

固定 `vx=0.35 m/s`、level 5、evaluation scenario seed 42 的 post-warmup 结果：

| Terrain | Actual vx | XY error | Distance | Termination |
|---|---:|---:|---:|---|
| flat | 0.335 | 0.060 | 6.46 m | timeout |
| pyramid slope | 0.224 | 0.186 | 4.76 m | timeout |
| random rough | 0.152 | 0.220 | 3.40 m | timeout |
| pyramid stairs | 0.077 | 0.288 | 2.46 m | bad orientation |
| inverted slope | 0.064 | 0.294 | 2.43 m | timeout |
| boxes | 0.040 | 0.331 | 2.81 m | timeout |
| inverted stairs | 0.023 | 0.339 | 2.64 m | timeout |

### Rough 三 training-seed 长版

- 目录：`evaluation/video_rough_multitraining_model2999_evalseed42_level5_vx035_20s_follow_2026-08-26`；
- 成片：`go2_rough_model2999_three_training_seeds_vx035_level5_20s_follow_dynamic_demo.mp4`；
- 61.94 秒，1280x720，50 FPS，3097 帧可解码；
- SHA-256：`3114114451694c63535542d79340dfb7e4328e89189622e018ab45ed51972b42`；
- evaluation scenario seed 均为 42，training seeds 分别为 42、7、123；
- actual vx 分别为 0.249、0.271、0.257 m/s；XY error 分别为
  0.117、0.106、0.107 m/s；三段均 timeout，无 base contact/bad orientation。

用户已人工观看并确认两部长版成片合格。作品集后续应制作 30--45 秒 highlight；长版保留
为完整技术证据，不应覆盖。

## 7. 已保存的复现入口

关键能力均已脚本化，不能再依赖聊天记录或 Shell history：

- `scripts/project/train_go2_baseline.sh`；
- `scripts/project/train_go2_rough.sh`；
- `scripts/project/train_go2_multiterrain.sh`；
- `scripts/project/evaluate_go2.py`；
- `scripts/project/evaluate_go2_rough_matrix.sh`；
- `scripts/project/evaluate_go2_rough_training_seed_matrix.sh`；
- `scripts/project/record_go2_video.sh`；
- `scripts/project/record_go2_multiterrain_demo.py` 与 `.sh`；
- `scripts/project/record_go2_rough_multitraining_demo.sh`；
- `scripts/project/compose_go2_demo_videos.py`；
- `scripts/project/analyze_tensorboard.py`；
- rough audit/archive 脚本；
- `scripts/project/reproduce_go2_multiterrain_stage2.sh`。

最后一个脚本固化了本轮曾由终端拼接的 Stage-2 精确命令，默认 `EXECUTE=0`，仅打印命令；
只有显式 `EXECUTE=1` 才运行，并要求新的 `REPRO_TAG` 或不存在的 `OUTPUT_ROOT`。使用前先看：

```bash
cd /home/kxj/rl_ws/unitree_rl_lab
scripts/project/reproduce_go2_multiterrain_stage2.sh --help
```

## 8. 新运行输出的分析协议

用户会继续粘贴训练、评估或视频命令输出。收到后必须先识别 run，不能看到 seed 数字就直接
归类。按以下字段核对：

1. task/task variant；
2. run name 与输出目录；
3. training seed；
4. evaluation scenario seed；
5. resume checkpoint 或被评估 checkpoint（含绝对路径和 SHA-256）；
6. terrain protocol、terrain level、command、env/episode 数；
7. 开始/结束时间和命令退出状态。

训练输出继续检查：checkpoint 连续性与最终文件、TensorBoard event、console/GPU log、
tracking error/reward、curriculum、termination、吞吐、GPU 利用率/显存/温度，以及 NaN/Inf。

评估输出继续检查：预期 episode 数、CSV/JSON 一致性、terrain type/level 覆盖、assignment
稳定性、command 与 actual tracking error、termination 分解、非有限值、checkpoint hash。

视频输出继续检查：原始 clip 数、帧数/FPS/分辨率/时长、机器狗是否持续在画面内、遥测字段、
JSON/CSV、成片 SHA-256 和 `sha256sum -c`。机器狗可见性仍需用户本地人工确认。

若输出不足，先给出只读核验命令；若需要实际训练/评估/录制，则给用户完整可复制命令，由
用户运行并反馈。不要自行启动这些长任务。

## 9. 下一阶段优先级

1. 制作 30--45 秒作品集 highlight，同时保留两部长版证据；
2. 解耦 command curriculum 与 terrain curriculum，避免速度命令和地形难度同步升级；
3. 对 stairs/boxes 做 terrain-aware sampling、失败回放和 reward 受控消融；
4. 从 survival baseline `model_3300` 出发，以同一 `multiterrain_stratified` 协议比较，
   不进行无目标追加 iterations；
5. 完成后再考虑 ONNX/MuJoCo Sim2Sim。当前禁止擅自安装或克隆，不做 Sim2Real 宣称。

每次实验只改变少量明确因素，并提前写清假设、对照、成功标准、训练预算和停止条件。需要
明确保留负结果，避免只展示最好一次。

## 10. 必须关联阅读的证据文档

接管时建议依次完整读取：

1. `doc/SESSION_HANDOFF_2026-08-26_ALL_HISTORY.md`（本文件）；
2. `doc/PROJECT_STATUS.md`；
3. `doc/RESULTS_MULTITERRAIN_CONTINUATION_2026-08-26.md`；
4. `doc/COMMAND_SCRIPT_AUDIT_2026-08-26.md`；
5. `doc/REPRODUCTION.md`；
6. `doc/RESUME_PROJECT.md`；
7. `doc/repro/go2_multiterrain_stage2_final_artifacts_2026-08-26.sha256`。

需要追溯早期阶段时再读：

- `doc/SESSION_HANDOFF_2026-08-25_ROUGH_MULTITRAINING_FINAL.md`；
- `doc/SESSION_HANDOFF_2026-08-25_MULTITERRAIN_PROTOTYPE.md`；
- `doc/RESULTS_ROUGH_MULTITRAINING.md`；
- `doc/RESULTS_MULTITERRAIN_PROTOTYPE.md`。

## 11. 新对话启动提示词

将下面整段粘贴到新对话：

```text
我正在继续 /home/kxj/rl_ws/unitree_rl_lab 的 Unitree Go2 强化学习项目。

请先完整读取：
/home/kxj/rl_ws/unitree_rl_lab/doc/SESSION_HANDOFF_2026-08-26_ALL_HISTORY.md

然后严格按该文件第 1、8、9、10 节接管，并自行执行必要的只读检查，确认 Unitree RL Lab
与 IsaacLab 的 HEAD、工作区状态、关键脚本、实验目录和 SHA-256 清单。类似本地文件读取和
只读检查已经自动授权，你可以持续检查并基于结果分析。

永久约束：禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；禁止覆盖、移动或
删除任何 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果或视频。训练、评估
和视频录制采用你给出完整命令、我在终端运行并回传输出的方式；不要自行启动这些任务。
安装、克隆或扩大项目范围前必须取得我的明确授权。

我随后会继续发送运行输出。每次先依据 task、task variant、run name、training seed、
evaluation scenario seed、checkpoint、目录和时间识别 run，再检查退出状态、checkpoint、
TensorBoard、console/GPU log、tracking error、curriculum、termination、吞吐、GPU 状态和
非有限值。始终严格区分 training seed 与 evaluation scenario seed。先总结当前已接管状态，
不要重复已完成的训练或覆盖已有产物，然后等待/分析我发送的新输出并指导下一步。
```

## 12. 说明

本文件能完整整合的是：当前会话上下文，以及历次会话已经沉淀到本地仓库的 handoff、结果、
复现与状态文档。未写入仓库、且当前上下文中已经不可见的历史聊天原文无法逐字恢复；关键
实验事实已由本地 checkpoint、CSV/JSON、日志、视频、Git 历史和 SHA-256 清单作为主证据。
