# Unitree Go2 RL 全历史接管：HeightScan 与作品集 Demo（截至 2026-08-26）

## 1. 接管规则与永久约束

这是当前最新的主接管文件。它以
`doc/SESSION_HANDOFF_2026-08-26_ALL_HISTORY.md` 为前史基线，并加入后续全部诊断、
HeightScan 开发、checkpoint 迁移、训练、评估和作品集视频结果。新会话应先完整读取本文件，
再按第 11 节读取关联证据；不要只依赖聊天摘要猜测状态。

本地文件读取和只读检查已自动授权。训练、评估和视频录制必须采用“助手给出完整命令，
用户在终端运行并回传输出”的方式，助手不得自行启动。永久约束：

- 禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；
- 禁止覆盖、移动或删除任何 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果或视频；
- 新运行必须使用尚不存在的 run name、日志路径和输出目录；
- 安装、克隆、联网下载或扩大项目范围前必须得到用户明确授权；
- 始终严格区分 training seed 与 evaluation scenario seed；
- 不把单 training-seed 的多 evaluation-seed 结果写成多训练 seed；
- 不宣称实机部署，不夸大楼梯、boxes 或所有复杂地形的能力；timeout 不等于有效运动。

不要重复已经完成的训练、评估或录制，不要擅自清理工作区中的任何修改。

## 2. 仓库、环境与当前工作区

- Unitree RL Lab：`/home/kxj/rl_ws/unitree_rl_lab`；
- 当前 HEAD（本文件生成前）：`247f48bdc6ab6648ac08981fd2d076fab81e90a3`；
- IsaacLab：`/home/kxj/rl_ws/IsaacLab`；
- IsaacLab HEAD：`39b494e6d8ae3f7adbb30ae1feaf45d21bdc7d19`，本轮只读检查为 clean；
- Conda 环境：`unitree_rl`；
- 禁止改变核心 NVIDIA/Isaac/PyTorch 软件栈。

Unitree RL Lab 当前是有意保留的 dirty worktree。以下是本轮 HeightScan/诊断相关修改，
不要 reset、checkout、覆盖或丢弃：

- `scripts/project/analyze_tensorboard.py`；
- `scripts/project/compose_go2_demo_videos.py`；
- `scripts/project/evaluate_go2.py`；
- `scripts/project/record_go2_multiterrain_demo.py`；
- `scripts/project/record_go2_multiterrain_demo.sh`；
- `scripts/project/train_go2_multiterrain.sh`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/multiterrain_velocity_env_cfg.py`；
- 新文件 `scripts/project/analyze_go2_multiterrain_diagnostics.py`；
- 新文件 `scripts/project/diagnose_go2_multiterrain_directional.sh`；
- 新文件 `scripts/project/migrate_go2_heightscan_checkpoint.py`。

`env_backup_unitree_rl.yml` 是用户已有的无关未跟踪文件，本轮未修改，必须继续保留。

每次新会话先运行只读检查：

```bash
cd /home/kxj/rl_ws/unitree_rl_lab
git status --short
git log -5 --oneline
git rev-parse HEAD
git -C /home/kxj/rl_ws/IsaacLab status --short
git -C /home/kxj/rl_ws/IsaacLab rev-parse HEAD
```

## 3. 全项目历史阶段摘要

项目目标是形成可复现、可审计、适合机器人强化学习/具身智能求职展示的 Unitree Go2
速度跟踪项目。此前已完成：

1. Flat baseline：training seed 42，2048 env，3000 PPO iterations；
2. Rough 50/50：flat/random rough 正式训练与固定分层评估；
3. Rough 多 training-seed：42、7、123，各 3000 iterations；
4. 七类 MultiTerrain：flat、random rough、正/倒 slope、正/倒 stairs、boxes；
5. survival baseline、tracking-aware 负对照、固定分层评估与动态遥测视频；
6. 本轮增强诊断、HeightScan 地形感知、progress-gated curriculum、长续训和作品集短片。

历史正式结果：

- Flat `model_2999.pt`：SHA-256
  `a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928`；三个 evaluation
  scenario seeds、384 episodes，XY error `0.0756 ± 0.0021 m/s`，timeout `99.48%`。
- Rough seed-42 正式 checkpoint `model_2900.pt`：SHA-256
  `9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889`。
- Rough training seeds 42/7/123 固定 checkpoint 聚合：overall XY
  `0.0956 ± 0.0009 m/s`，rough XY `0.1131 ± 0.0013 m/s`，timeout `1144/1152`。
- 无 HeightScan MultiTerrain survival baseline `model_3300.pt`：SHA-256
  `5756f08ca4df34d71bdd5d25fa3763b91c96c4af59f60c7598eec822322b33b3`；scenario seeds
  7/42/123 的 XY error `0.2722 ± 0.0095 m/s`，yaw error `0.1838 ± 0.0113 rad/s`，
  timeout `79.29% ± 4.29 pp`。
- tracking-aware 分支是有效负结果，没有超过 survival baseline。

## 4. 本轮诊断结论

为避免把 timeout 当作成功，本轮增强固定地形录制诊断，加入：机器人姿态、初始朝向投影
位移、base/foot contact force、termination、滑窗 stall fraction 和逐步 tracking CSV。

无 HeightScan `model_3300` 在 direction diagnostics 中表现为：

- pyramid slope 正反方向可产生有效位移；
- inverted slope 明显慢或停滞；
- stairs 普遍停滞，正 pyramid stairs 在约 18.82 秒 bad orientation；
- inverted stairs 约 76%--78% 时间停滞；
- boxes 约 49% 时间停滞。

结论：主要问题不是单纯跌倒，而是策略在结构化障碍前采取“保命但不推进”的局部最优；
因此先加入地形感知是有意义的。

## 5. HeightScan 实现

新增 task：`Unitree-Go2-Velocity-MultiTerrain-HeightScan`。

核心设计：

- actor/critic 追加 `mdp.height_scan`；
- scan grid 约 `1.2 m × 0.8 m`，resolution `0.1 m`，共 117 rays，向前偏置 `0.45 m`；
- policy observation 从 45 增至 162，critic observation 从 60 增至 177；
- actor scan noise 为约 `±0.02`，clip `[-1, 1]`；
- 新 curriculum `multiterrain_levels_progress_tracking_survival` 同时要求 timeout、tracking 阈值
  `0.45` 和最小净位移 `0.75 m`，小于 `0.1 m/s` 的 standing command 豁免；
- command curriculum 被禁用，使中等速度命令与 terrain curriculum 解耦；
- `HeightScanPPORunnerCfg` 使用固定 learning rate `1e-5`；
- 训练 wrapper 支持 `TASK_VARIANT=height_scan`；
- 修复 Isaac 将 checkpoint 参数当 regex 时误选 `.sha256` sidecar 的问题，改为精确锚定 regex；
- evaluator、recorder、diagnostic wrapper 均支持 HeightScan；
- composer 后续增加 `--terrains`、`--start_seconds`、`--duration_seconds`，默认行为保持兼容。

## 6. Checkpoint 迁移和训练谱系

当前 HeightScan 不是从零训练，而是从成熟无 HeightScan baseline `model_3300.pt` 迁移后补训。
迁移脚本保持旧 actor/critic 权重完全一致，新 117 列输入权重置零。

第一次迁移保留旧 optimizer 的 smoke 发生灾难性 bad orientation（最终约 93.6%），该 run 仅作
负诊断，不能继续使用。稳定 v2 迁移将 optimizer state/step 清零并设 lr `1e-5`：

- bootstrap v2 SHA-256：
  `462e2a1283d185c5c6774fe434022411589499fe65dbc057e867333dbdd2b659`；
- 128 env × 5 smoke 稳定，无 base contact/bad orientation；
- 1024 env × 100 pilot 最终 `model_3403.pt`：
  `logs/rsl_rl/unitree_go2_velocity_multiterrain/2026-08-26_16-24-32_multiterrain_v4_heightscan_progressgate_fixedcmd_low_lr1e5_pilot100_from3304_seed42_env1024/model_3403.pt`；
  SHA-256 `89757e19f5502d3cb53fcfb2f976c8c56b1cfe47eca47b4d283c64f6cd232031`。

随后同一 training seed 42 连续补训：

1. `model_3403 → model_6000`，1024 env，2598 iterations；最终 SHA-256
   `0867f75b611264715859b0b8094479879c4a1b1f84dab5b841e79b65a896fcfa`；
2. `model_6000 → model_10000`，1024 env，4001 iterations；最终路径：
   `logs/rsl_rl/unitree_go2_velocity_multiterrain/2026-08-26_18-40-22_multiterrain_v4_heightscan_progressgate_fixedcmd_low_lr1e5_continue_to10000_from6000_seed42_env1024_addi4001/model_10000.pt`；
   SHA-256 `5f8768907c71141ea7c588f7b27b3248081daa8d00b7bcda31e8a1137faf2956`。

第二段训练 exit code 0，4001 TensorBoard iterations、14 scalar tags、non-finite 0。后期约：

- reward `20.35`；episode length `948`；
- XY training error `0.282`；yaw error `0.307`；
- timeout `88.5%`；bad orientation `11.3%`；base contact `0.17%`；
- terrain curriculum `5.30`；
- throughput `12.7k--13.2k steps/s`；
- GPU 平均利用率约 84%，平均显存约 5.3 GiB，最高温度 68°C；无 GPU/数值错误。

训练已进入平台期。当前不要继续盲目增加 iterations，也不要默认 final iteration 永远最好。

## 7. 固定分层评估与正式候选

所有以下 HeightScan 评估均为 training seed 42、evaluation scenario seed 42、
`multiterrain_stratified`、70 env × 2 episodes = 140 episodes。CSV 均为 140 行，无 NaN/Inf。

| Checkpoint | XY error | yaw error | timeout | bad orientation | base contact |
|---|---:|---:|---:|---:|---:|
| model_3403 | 0.2633 | 0.1681 | 84.3% | 13.6% | 2.1% |
| model_6000 | 0.2432 | 0.1360 | 77.1% | 21.4% | 1.4% |
| model_8500 | 0.2415 | 0.1322 | 79.3% | 20.0% | 0.7% |
| model_9700 | 0.2226 | 0.1382 | 80.7% | 19.3% | 0.0% |
| model_10000 | **0.2187** | 0.1372 | **84.3%** | **15.0%** | 0.7% |

`model_10000` 是当前综合正式候选。相对 `model_3403`，XY error 下降约 17%，timeout 持平。
evaluation scenario seed 42 下按地形的重要结论：

- flat：timeout 100%，XY `0.164`；
- random rough：timeout 100%，XY `0.230`；
- pyramid slope：timeout 100%，XY `0.154`；
- inverted pyramid slope：timeout 95%，XY `0.207`；
- boxes：timeout 70%，XY `0.212`；
- inverted pyramid stairs：timeout 80%，XY `0.306`；
- pyramid stairs：timeout 45%，XY `0.257`，仍是主要短板。

评估命令由用户运行；用户决定暂不增加 evaluation scenario seeds 7/123。因此不能将
`model_10000` 描述成跨 evaluation seed 已验证的结果。不同 task/checkpoint 即使 seed 相同，
episode command 分布也不保证完全配对，跨 task 的小差异要谨慎解释。

## 8. HeightScan Demo 与量化结论

固定参数均为 training seed 42、evaluation scenario seed 42、terrain level 5。

### vx=0.20 m/s

七段均 timeout、无跌倒，但多数接近原地踏步：flat 实际 `0.025 m/s`，rough `0.004 m/s`，
pyramid stairs `0.012 m/s`。这批视频保留为低速死区证据，不用作最终作品集。

### vx=0.35 m/s，固定相机

flat 解除停滞但走出镜头，因此不作为最终主素材。诊断显示：

| Terrain | Actual vx | Forward displacement | Stalled fraction | Termination |
|---|---:|---:|---:|---|
| flat | 0.293 | 5.643 m | 0.000 | timeout |
| random rough | 0.089 | 1.827 m | 0.218 | timeout |
| pyramid slope | 0.107 | 2.374 m | 0.359 | timeout |
| inverted slope | 0.071 | 1.385 m | 0.427 | timeout |
| boxes | 0.034 | 1.038 m | 0.411 | timeout |
| pyramid stairs | 0.026 | 0.903 m | 0.616 | timeout |
| inverted stairs | 0.030 | 0.722 m | 0.800 | timeout |

### vx=0.35 m/s，asset-follow 相机

最终原始素材目录：

`evaluation/video_multiterrain_heightscan_model10000_trainseed42_evalseed42_level5_vx035_20s_assetfollow_2026-08-26`

七段、每段约 20 秒、1280×720、50 FPS，清单 `sha256sum -c` 全部通过；机器人持续在画面中，
无 NaN/Inf、base contact 或 bad orientation。量化数据与固定相机版一致。

准确表述：flat 有稳定跟踪；rough 与 slope 有有效运动但误差明显；stairs/boxes/inverted
terrain 仍保守、缓慢或间歇停滞。不要用“七种复杂地形均精确跟踪”描述本项目。

## 9. 最终作品集短片

主成片：

`evaluation/portfolio_go2_heightscan_model10000_level5_vx035_assetfollow_selected4_34s_2026-08-26.mp4`

- 内容顺序：flat、random rough、pyramid slope、inverted pyramid slope；
- 每段从原视频第 2 秒开始截取 8 秒，另有 2 秒片头；
- 总时长 34.0 秒，1280×720，50 FPS，1700 frames；
- 大小 19,764,258 bytes；
- SHA-256：`90a4909bb527eea70034e699cd9f3a81a1f119e619772309b0a06845eb2de887`；
- `.json` 和 `.sha256` sidecar 已生成，`sha256sum -c` 通过；
- 画面包含 task/terrain/seeds、command vx、actual vx、XY error、累计距离、termination 和动态曲线；
- 抽帧检查确认主体、标签、曲线和切换正常。

这份短片是当前简历主 Demo。七地形长素材作为完整技术证据，stairs/boxes 作为失败分析和
后续优化依据，禁止删除或覆盖。

建议简历表述：

> 为 Unitree Go2 PPO 策略加入 117 维前视高度扫描与进度门控课程学习，在七类程序化地形上
> 完成固定分层评估；策略在平地实现稳定速度跟踪，并在 rough、坡面地形保持无跌倒运动，
> 同时通过逐步诊断定位楼梯与离散障碍上的保守停滞问题。

## 10. 后续优先级

当前可视为“可用于简历展示的 terrain-aware locomotion 原型”。新会话不要重复训练或录制。
建议顺序：

1. 更新项目结果文档、README/PROJECT_STATUS 和最终 SHA-256 manifest，明确正结果和局限；
2. 审查并提交当前有意保留的 HeightScan/诊断代码修改，但必须保留用户无关文件；
3. 若继续算法优化，针对 pyramid stairs、inverted stairs、boxes 做小预算受控实验；
4. 优先改变采样/进度奖励或障碍专项 curriculum，预先定义成功标准，不再无目标追加 iteration；
5. 需要严格科研对照时，再考虑从零训练 HeightScan 与 warm-start 的同预算对照；
6. ONNX/MuJoCo Sim2Sim 只能在用户明确授权安装/克隆或扩展范围后进行；当前不做 Sim2Real 宣称。

## 11. 新输出分析协议与证据文档

收到用户粘贴的新输出后，必须先识别：task、task variant、run name、training seed、evaluation
scenario seed、checkpoint 绝对路径及 SHA-256、目录、时间、terrain protocol/level、command、
env/episode 数。然后检查：

- 训练：退出状态、checkpoint 连续性、TensorBoard、console/GPU log、reward/tracking、curriculum、
  termination、吞吐、显存/温度/利用率、NaN/Inf；
- 评估：episode 数、CSV/JSON 一致性、地形与 level 覆盖、command 分布、tracking、termination、
  checkpoint hash 和非有限值；
- 视频：clip 数、FPS/帧数/分辨率/时长、主体可见性、metadata/CSV、termination、SHA-256。

关联阅读顺序：

1. 本文件；
2. `doc/SESSION_HANDOFF_2026-08-26_ALL_HISTORY.md`；
3. `doc/PROJECT_STATUS.md`；
4. `doc/RESULTS_MULTITERRAIN_CONTINUATION_2026-08-26.md`；
5. `doc/RESULTS_GO2_DEMOS_2026-08-26.md`；
6. `doc/COMMAND_SCRIPT_AUDIT_2026-08-26.md`；
7. `doc/REPRODUCTION.md` 与 `doc/RESUME_PROJECT.md`；
8. `doc/repro/go2_multiterrain_stage2_final_artifacts_2026-08-26.sha256`；
9. 需要早期细节时再读 `doc/SESSION_HANDOFF_2026-08-25_*` 和对应 RESULTS 文档。

本文件整合当前可见会话上下文及历次会话已经沉淀到本地的 handoff、日志、CSV/JSON、
checkpoint、视频和 Git 证据。不可见且从未落盘的历史聊天不能逐字恢复，但关键事实均尽量以
本地证据和 SHA-256 为准。

## 12. 新对话启动提示词

将以下整段原样粘贴到新对话：

```text
我正在继续 /home/kxj/rl_ws/unitree_rl_lab 的 Unitree Go2 强化学习项目。

请先完整读取：
/home/kxj/rl_ws/unitree_rl_lab/doc/SESSION_HANDOFF_2026-08-26_HEIGHTSCAN_DEMO_ALL_HISTORY.md

然后按该文件第 1、2、10、11 节接管，并按第 11 节顺序按需读取关联证据。请自行执行必要的
只读检查，确认 Unitree RL Lab 与 IsaacLab 的 HEAD、dirty workspace、关键脚本、训练/评估/
视频目录和 SHA-256。类似本地读取与只读检查已自动授权，可以持续检查并基于结果分析。

永久约束：禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；禁止覆盖、移动或
删除任何 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果或视频。训练、评估
和视频录制必须由你给出完整命令、我在终端运行并回传输出；你不要自行启动。安装、克隆、
联网下载或扩大项目范围前必须取得我的明确授权。不要清理当前 dirty worktree，也不要修改
或删除 env_backup_unitree_rl.yml。

我会继续发送运行输出。每次先依据 task、task variant、run name、training seed、evaluation
scenario seed、checkpoint、目录和时间识别 run，再检查退出状态、checkpoint、TensorBoard、
console/GPU log、tracking error、curriculum、termination、吞吐、GPU 状态和非有限值。始终
严格区分 training seed 与 evaluation scenario seed，timeout 不等于有效运动。

先向我总结你已接管的当前状态，明确正式 checkpoint model_10000、固定分层评估、最终 34 秒
作品集 Demo、尚未解决的 stairs/boxes 停滞和当前未提交代码修改；不要重复已完成的训练、
评估或录制。然后等待并分析我发送的新输出，基于证据总结并指导下一步。
```
