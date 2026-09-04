# Unitree Go2 项目 V9 主接管：七地形 Teacher v1、完整验证、Demo v2 与最终命令接口纠偏（含全部历史索引）

日期：2026-08-30（Asia/Shanghai）

本文件是 V8 之后的最新主接管文件。它既给出此前项目历史的不可丢失结论，也完整记录 2026-08-29 至
2026-08-30 新完成的七地形 Teacher v1 训练、三阶段闭环验证、可视化 Demo 和最终命令接口纠偏。
新会话必须从第一行读到文件末尾，随后严格执行第 15 节的读取顺序，不得只读摘要。

## 1. 永久约束

1. 禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch。
2. 禁止覆盖、移动或删除 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果、原始视频或合成视频。
3. 训练、评估、录制和视频合成只能由助手给出完整命令，由用户在自己的终端执行并回传输出；助手不得自行启动。
4. 安装、克隆、联网下载或扩大范围前必须得到用户明确授权。
5. 不清理 dirty worktree，不修改或删除 `env_backup_unitree_rl.yml`。
6. 不重复已完成的训练、评估、录制或同名视频合成。
7. 始终严格区分 training seed 和 evaluation scenario seed。
8. exit code 0 只说明程序完成；timeout 不等于有效运动或成功。
9. 新结果必须先确认 task、variant、run name、training seed、evaluation scenario seed、source/current checkpoint、绝对目录和时间，再分析证据。
10. 每次收到运行输出，都要检查 exit status、SHA、checkpoint iteration/optimizer、TensorBoard、console/GPU、tracking、progress、stall、terrain/level/command 覆盖、净位移、path length、安全指标和非有限值。

## 2. 仓库与当前运行状态

- Unitree RL Lab：`/home/kxj/rl_ws/unitree_rl_lab`
- HEAD：`247f48bdc6ab6648ac08981fd2d076fab81e90a3`
- worktree：dirty；包含此前和本轮项目开发，禁止清理。
- IsaacLab：`/home/kxj/rl_ws/IsaacLab`
- HEAD：`39b494e6d8ae3f7adbb30ae1feaf45d21bdc7d19`
- IsaacLab worktree：2026-08-30 最后核验时 clean。
- 2026-08-30 最后核验时，没有残留训练、评估、录制或视频合成进程。
- 环境：`/home/kxj/miniforge3/envs/unitree_rl`
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，12227 MiB；七 Teacher 训练中实际约使用 4.65–5.36 GiB，最高可见温度约 67°C。

## 3. 不可丢失的全部历史结论

以下结论由 V8 及其前序 handoff 固化，不得在新会话中改写：

1. HeightScan model10000 与历史 34 秒 Demo 已完成并保留。
2. V5 foundation 从零训练使用 training seed 42；model3200 依据 evaluation scenario seeds 7/42/123 选定。
3. stairs 历史分支从 model2200 恢复，语义是总迭代继续到最终 model4999，不是从零训练 4999 次。
4. ascending remediation 的 model900 被接受；model1199 因退化被拒绝。
5. 47 秒 Demo v1 是 multi-checkpoint showcase，不是统一策略。
6. boxes safety model900 只是 bounded proxy teacher，不是完整 boxes Teacher。
7. 七个 terrain dispatch entries 当时实际只有四个 unique proxy policies。
8. dispatcher online v2 和 recurrent student imitation smoke v2 已通过。
9. proxy student model10000 从零训练、training seed 42、10000 imitation updates；SHA-256：
   `f70ce64aa44a54ced84660e51c114d7b105f04f1fdfc876d93fb54eb9912cc5b`。
10. proxy student 低 imitation loss 只证明模仿学习；完整 closed-loop 验证后来已完成并明确失败：
    - 21/21 runs，2940 episodes，evaluation seeds 7/42/123；
    - bad-orientation rate 91.9048%；timeout 8.0612%；
    - mean command-aligned progress 0.131204 m/s；stall 17.8515%；
    - 完整 manifest 自身 SHA：`b46ce39bc4f10ecbb55c770a0e1c814fa6e04c8684f747eab34c5d6da0eb659a`；
    - 该历史 manifest 错误包含自身条目；其他 JSON/CSV/console 校验通过，禁止覆盖历史 manifest。
11. proxy student 六场景诊断视频已完成，manifest SHA：
    `7ec73d146a92f543c369efe6fe8a10cbeb0908cc229cdcfe90d627ac2e7af353`。
12. proxy student 是有价值的负结果基线，不得作为成功统一策略或最终 Demo。
13. 长期科学目标仍是：七个 terrain-specific teachers 分别从零训练和严格验证，再从零训练真正统一的 recurrent perceptive student。

历史详情见：

- `doc/SESSION_HANDOFF_2026-08-29_V8_PROXY_STUDENT_ABLATION_AND_PENDING_VALIDATION_ALL_HISTORY.md`
- `doc/RESULTS_PROXY_STUDENT_MODEL10000_CLOSED_LOOP_2026-08-29.md`
- V8 第 12 节列出的 V7/V6/V5/HeightScan/总目标/各 manifest。

## 4. Teacher v1 的最初设计和训练预算

新增配置：

- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/terrain_teacher_v1_env_cfg.py`
- `config/terrain_teacher_v1_contract.json`
- 七个 Gym task 注册位于 `.../go2/__init__.py`
- runner：`TerrainTeacherV1PPORunnerCfg`
- 单 Teacher 训练脚本：`scripts/project/train_go2_terrain_teacher_v1.sh`
- 七 Teacher 顺序训练脚本：`scripts/project/train_go2_7terrain_teachers_v1.sh`

共同训练参数：

- 从随机初始化开始；training seed 42；
- 1024 environments；
- 32 rollout steps/update；
- 5000 PPO updates/teacher；
- 163,840,000 transitions/teacher；七个合计 1,146,880,000 transitions；
- recurrent GRU hidden 256、1 layer；
- actor/critic MLP `[512, 256, 128]`，ELU；
- actor observation 162（含 117 HeightScan），critic observation 177，actions 12；
- PPO epochs 5、mini-batches 8、learning rate `3e-4` adaptive、gamma 0.99、lambda 0.95、clip 0.2、desired KL 0.01、entropy 0.01；
- checkpoint 每 100 updates 保存，最后额外保存 model4999。

5000 updates 是首轮预算上限，不是自动合格线。它对应的 transitions 略高于 Isaac Lab 官方 Go2 rough 常见的
4096 env × 24 steps × 1500 updates = 147,456,000，但复杂任务是否充分只能由闭环验证判断。

## 5. Teacher v1 实际训练命令域

Teacher v1 实际训练的是三维机身速度命令 `[vx, vy, yaw_rate]`，连续随机采样：

| terrain | vx (m/s) | vy (m/s) | yaw rate (rad/s) |
|---|---:|---:|---:|
| flat | [-0.60, 0.80] | [-0.35, 0.35] | [-0.80, 0.80] |
| random_rough | [-0.30, 0.65] | [-0.20, 0.20] | [-0.60, 0.60] |
| pyramid_slope | [-0.20, 0.55] | [-0.15, 0.15] | [-0.50, 0.50] |
| inverted_pyramid_slope | [-0.20, 0.55] | [-0.15, 0.15] | [-0.50, 0.50] |
| pyramid_stairs | [0.15, 0.50] | [-0.08, 0.08] | [-0.30, 0.30] |
| inverted_pyramid_stairs | [0.15, 0.50] | [-0.08, 0.08] | [-0.30, 0.30] |
| boxes | [0.15, 0.50] | [-0.08, 0.08] | [-0.30, 0.30] |

注意：连续随机采样意味着负 vx/非零 vy 分量确实在训练分布中，但精确的“纯横移”或“纯原地转向”点并不是离散课程片段。

## 6. 七个 Teacher v1 训练结果

用户执行了七个从零训练任务，全部进程 exit 0。每个 run 均有 51 个 checkpoint、TensorBoard event、console 和 GPU CSV；
最终 checkpoint 内部 `iter=4999`，均含完整 optimizer state；关键 TensorBoard 标量无 NaN/Inf，无 OOM、traceback 或中途 stall。

| terrain | run directory | model4999 SHA-256 |
|---|---|---|
| flat | `logs/rsl_rl/unitree_go2_velocity_teacher_v1_flat/2026-08-29_16-48-29_teacher_v1_flat_fromscratch_trainseed42_env1024_i5000` | `5695c610874c1577390bf98ac909cb4b161f3b4cef0905cd12930e7fd4e1edc5` |
| random_rough | `logs/rsl_rl/unitree_go2_velocity_teacher_v1_random_rough/2026-08-29_19-35-33_teacher_v1_random_rough_fromscratch_trainseed42_env1024_i5000` | `8d139c4be23994ca474b9876c41fd03740956cda016328b01aaa52fb454178da` |
| pyramid_slope | `logs/rsl_rl/unitree_go2_velocity_teacher_v1_pyramid_slope/2026-08-29_22-31-02_teacher_v1_pyramid_slope_fromscratch_trainseed42_env1024_i5000` | `e33d9fe93ad5070132abb2c3baacb4cf94485361b1a02aab8cfcb775da28ee55` |
| inverted_pyramid_slope | `logs/rsl_rl/unitree_go2_velocity_teacher_v1_inverted_pyramid_slope/2026-08-30_01-28-47_teacher_v1_inverted_pyramid_slope_fromscratch_trainseed42_env1024_i5000` | `f72c8e63c0333234314b2aaa8f97daad6982d56d4f7f5676a7d1afd8906dd01b` |
| pyramid_stairs | `logs/rsl_rl/unitree_go2_velocity_teacher_v1_pyramid_stairs/2026-08-30_04-15-42_teacher_v1_pyramid_stairs_fromscratch_trainseed42_env1024_i5000` | `368746eb5f2d94327ac955a6cbec2b0124607031ecf8ac10ab79b5dbb6fa03ea` |
| inverted_pyramid_stairs | `logs/rsl_rl/unitree_go2_velocity_teacher_v1_inverted_pyramid_stairs/2026-08-30_07-13-57_teacher_v1_inverted_pyramid_stairs_fromscratch_trainseed42_env1024_i5000` | `588ce05a44a96ffee6f7a236bf96affd80552ea4886ad4fdf3a426aa76ed7aa2` |
| boxes | `logs/rsl_rl/unitree_go2_velocity_teacher_v1_boxes/2026-08-30_10-23-44_teacher_v1_boxes_fromscratch_trainseed42_env1024_i5000` | `d73843ab31d92c585ec26381142ffafcc5efea31cb4de07ca533f44e6edb8b38` |

训练曲线提示：random_rough 的平滑 reward 在约 update1861 达峰后回落；inverted_pyramid_stairs 后期 yaw error 约 0.31。
所以从未默认七个 model4999 都是最佳。

## 7. 三阶段 closed-loop 验证

Evaluator 新增 `teacher_v1_stratified`：关闭运行时 curriculum，固定单一 Teacher terrain，按 evaluation scenario seed 生成地形，
20 env 中每个 level 0–9 分配两个 env；每环境两集时，每项共 40 episodes。记录 tracking、净位移、path length、stall、功率和安全终止。

### 7.1 候选筛选：evaluation seed 42

- 21 candidates；每项 40 episodes；level 0–9 全覆盖；全部 JSON/CSV/console 完整、无非有限值。
- 目录：`evaluation/teacher_v1_candidate_screen_trainseed42_evalseed42_v1`
- manifest SHA：`3e17e0bed20659fc314590f6f2386c29be0eb29362d7c26f4d6494e6fe4b5ed8`

### 7.2 Finalists 随机命令跨 seed

- 9 finalists；evaluation seeds 7/123，并与第一阶段 seed42 合并分析。
- 目录：`evaluation/teacher_v1_finalists_trainseed42_crossseed_random_commands_v1`
- manifest SHA：`1bb6da9253dcc5fae34510a9ebe3cd91e9670451b1439b8775dd4dbb074390a0`

由随机命令跨 seed 选出的七个固定命令 finalists：

| terrain | selected checkpoint | SHA-256 |
|---|---:|---|
| flat | model4999 | `5695c610874c1577390bf98ac909cb4b161f3b4cef0905cd12930e7fd4e1edc5` |
| random_rough | model4999 | `8d139c4be23994ca474b9876c41fd03740956cda016328b01aaa52fb454178da` |
| pyramid_slope | model4000 | `076a3691b8dd91bf4ab918a7e0e4c20987e7319a352ab783853fcae67367a1a4` |
| inverted_pyramid_slope | model4000 | `00b8653af4ea8a0a88b4c2e14518c7851564abaef82041536734f6fcc4a7c491` |
| pyramid_stairs | model4700 | `0fcdad89af3c7f27a2ab97f1094b9e67b4aa5be84377bef8db54025215359e23` |
| inverted_pyramid_stairs | model4999 | `588ce05a44a96ffee6f7a236bf96affd80552ea4886ad4fdf3a426aa76ed7aa2` |
| boxes | model4999 | `d73843ab31d92c585ec26381142ffafcc5efea31cb4de07ca533f44e6edb8b38` |

### 7.3 固定命令矩阵

- 七个 finalist × 每地形七个域内命令 × seeds 7/42/123 = 147 runs；
- 每项 40 episodes；总计 17,640 episodes；
- level 0–9 全覆盖；全部数值有限；全部 evidence SHA 校验通过；
- 目录：`evaluation/teacher_v1_finalists_trainseed42_fixed_command_matrix_seeds7_42_123_v1`
- manifest SHA：`11b0829c053b53e05310be2bd6850fe4dcf43ded78ef1eca2d8d0fb7d5a7ad85`

结论：

- flat model4999：通过该宽域矩阵，840/840 无安全终止；前进/后退/横移/yaw 跟踪良好。
- pyramid_slope model4000：基本通过；reverse 1/120 失败。
- inverted_pyramid_slope model4000：基本通过；mixed 2/120 失败。
- random_rough model4999：暂不通过；fixed forward 7/120 安全终止（5.8%）。
- pyramid_stairs model4700：暂不通过；forward-correct-right 5/120、forward-high 4/120 等少量早停。
- inverted_pyramid_stairs model4999：不通过；系统性 yaw tracking 缺陷。即使 yaw command=0：
  - forward-low mean yaw error 约 0.234 rad/s；
  - forward-high 约 0.463 rad/s；
  - 左右 correction 约 0.327–0.329 rad/s；
  - forward-turn ±0.25 时 yaw error 约 0.160–0.161 rad/s。
- boxes model4999：暂不通过；mixed 7/120 失败，高等级 7–9 聚合风险明显，level8–9 尤其突出。

不能把训练 reward、episode length 或 timeout 当作可靠 Teacher 证明。

## 8. 最终部署命令接口已纠偏

用户在本轮明确确认最终部署接口是：

```text
vx：非负前进速度
vy：恒为 0
yaw_rate：转向角速度
```

理由：实际部署的高度感知主要面向机器人前方，后方和侧方高度信息不可靠；实际控制只需要前进速度和转向。

重要结论：Teacher v1 的 flat/rough/slopes 包含负 vx 和非零 vy；七个任务都包含非零 vy。因此 Teacher v1 比最终接口更宽。
它们仍可作为宽域实验基线，并可在最终接口子域中做可视化，但不应未经重新对齐和修复就蒸馏为最终 Student。

用户还确认 Demo/展示不包含：后退、横移、原地旋转；只允许 `vx>0, vy=0`，yaw 非零时必须同时前进。

尚需在新会话正式确定 Teacher v2 contract 的两点：

1. 正式训练是否包含 `vx=0` 的静止状态，以及是否允许任何原地转向；Demo 已明确不展示原地转向。
2. yaw 上限是否与 vx 耦合。推荐低速允许较大 yaw、高速降低 yaw，避免矩形命令盒产生“最高速度+最大 yaw”的极小转弯半径。

## 9. yaw 范围解释

`yaw=[-0.8,0.8]` 是角速度，不是朝向角：

- `0.8 rad/s = 45.84 deg/s`；
- 保持 1 秒约转 45.8°，2 秒约 91.7°，约 7.85 秒转一圈；
- 总朝向并不限制在 ±45°。

当前 v1 yaw 范围：flat ±0.8 rad/s，rough ±0.6，slopes ±0.5，stairs/boxes ±0.3。
转弯半径近似 `R=vx/|yaw_rate|`，所以必须考虑 vx/yaw 联合约束。

## 10. Teacher v1 Demo v2

为七个 selected checkpoints 录制了 49 个原始片段：每地形七段，均严格采用展示接口 `vx>0, vy=0`：

- low/mid/high straight forward；
- forward + gentle left/right；
- forward + strong left/right。

没有后退、横移或原地转向。每段保存 raw MP4、逐帧 CSV、metadata、console 和 SHA。

原始取证根目录：

`evaluation/video_teacher_v1_forward_steering_command_domain_demo_v1`

原始录制及 v1 合成总 evidence manifest SHA：

`0fed2c6017efb5321ce107c3860b969a5e34ddd450f4395133cc950ddc3cfe1c`

合成 v1 曾发现 HUD 语义缺陷：旧 recorder 的 stall 按相对初始世界朝向的位移判断，会把正常弧线转弯误标为 stall。
原始视频和速度 trace 正确，未重录；只修改 composer，在 v2 中：

- `body-forward distance` 通过机身坐标实际 vx 积分；
- warmup 后，仅当 command vx≥0.1 且 actual vx<0.05 时显示 stall；
- 明确 body-forward distance 不是世界坐标净位移。

最终 v2 目录：

`evaluation/video_teacher_v1_forward_steering_command_domain_demo_v1/composed_v2`

最终 v2 manifest SHA：

`d8fdd017500d576c8dcd728d3b3440e7adaabd8e37262fe7aabff27f9d145f8d`

视频规格：1280×720、50 FPS；七个独立完整版各 60.86 s；总览 173.2 s；全部八个视频 SHA 校验通过。

总览：

`.../composed_v2/teacher_v1_seven_terrain_overview_demo.mp4`

七个独立版：

- `teacher_v1_flat_full_demo.mp4`
- `teacher_v1_random_rough_full_demo.mp4`
- `teacher_v1_pyramid_slope_full_demo.mp4`
- `teacher_v1_inverted_pyramid_slope_full_demo.mp4`
- `teacher_v1_pyramid_stairs_full_demo.mp4`
- `teacher_v1_inverted_pyramid_stairs_full_demo.mp4`
- `teacher_v1_boxes_full_demo.mp4`

所有视频明确标注：七个独立 checkpoints，不是 unified policy；已知问题用 `KNOWN ISSUE` 保留，禁止包装成全部成功。

## 11. 本轮新增/修改的关键代码

- `source/.../go2/terrain_teacher_v1_env_cfg.py`
- `source/.../go2/__init__.py`：七 Teacher task registration
- `source/.../agents/rsl_rl_ppo_cfg.py`：TerrainTeacherV1PPORunnerCfg
- `config/terrain_teacher_v1_contract.json`
- `scripts/project/train_go2_terrain_teacher_v1.sh`
- `scripts/project/train_go2_7terrain_teachers_v1.sh`
- `scripts/project/evaluate_go2.py`：新增 `teacher_v1_stratified`
- `scripts/project/evaluate_go2_terrain_teacher_v1_candidates.sh`
- `scripts/project/evaluate_go2_terrain_teacher_v1_finalists_crossseed.sh`
- `scripts/project/evaluate_go2_terrain_teacher_v1_fixed_command_matrix.sh`
- `scripts/project/record_go2_multiterrain_demo.py`：支持七 Teacher task
- `scripts/project/record_and_compose_go2_terrain_teacher_v1_demos.sh`
- `scripts/project/compose_go2_terrain_teacher_v1_demos.py`：双语标题、三轴 HUD、v2 stall 修复

静态检查均已通过：Python compile、bash `-n`、`git diff --check`、task registration/config smoke、dry-run 命令计数。

## 12. 当前科学判断

1. Proxy Student 失败证明 teacher-driven one-step recurrent imitation 即使 loss 很低，也可能 closed-loop 崩溃。
2. 七个从零 Teacher v1 的训练工程链路和大规模验证链路已经建立。
3. Teacher v1 中 flat 最稳定；两种 slopes 接近可用；rough、两种 stairs、boxes 仍有明确安全/方向问题。
4. inverted_pyramid_stairs 的 yaw 问题是系统性的，不是只在最高等级出现。
5. 更重要的是，Teacher v1 的命令域与最终部署接口不一致。不能只修四个失败 Teacher 后直接进入 Student；应先冻结 v1，再定义最终接口对齐的 Teacher v2。
6. Teacher v1 的训练、验证和 Demo 都不得删除或覆盖，它们是宽域 ablation 与失败驱动设计证据。

## 13. 推荐的下一步顺序

新会话不得立刻启动训练。先完成：

1. 完整读取 V9、V9 manifest、V8 及 V8 第12节历史链、全部 Teacher v1 evidence manifests 和当前代码。
2. 只读确认 HEAD/dirty、所有关键 SHA、当前无残留进程。
3. 将 Teacher v1 结果写成冻结 results 文档和 registry（如果尚未存在），不得覆盖历史文件。
4. 与用户确认最终 Teacher v2 命令 contract：
   - `vy=0`；
   - `vx` 各地形非负范围；
   - 是否包含 `vx=0` 静止；
   - 是否允许原地转向；
   - yaw 是否采用随 vx 收紧的耦合 envelope；
   - 各地形明确验收阈值。
5. 参考当前固定矩阵，设计只覆盖最终接口的 Teacher v2 curriculum 和 reward：先修命令定义，再处理 rough/stairs/boxes 安全和 inverted-stairs yaw。
6. 开发完成后只提供命令，仍由用户执行训练。
7. 七个最终接口 Teacher 均通过 seeds 7/42/123、levels 0–9、域内速度/转向矩阵后，才建立最终 teacher registry。
8. 最后从零训练真正统一的 recurrent perceptive student，并进行同等 closed-loop 验证。

## 14. 收到新运行输出时的固定分析协议

先识别并复述：

- task / variant / run name；
- training seed；
- evaluation scenario seed；
- source checkpoint / current checkpoint；
- 绝对输出目录；
- 起止时间。

然后检查：

- exit status；
- checkpoint SHA、iteration、optimizer state；
- TensorBoard 前中后趋势和候选峰值；
- console/GPU、stall、OOM、NaN/Inf；
- terrain type/level 和 command 覆盖；
- command/actual vx、vy、yaw；
- command-aligned progress、净位移和 path length；
- timeout、base contact、bad orientation；
- 不允许总体均值掩盖高等级或特定转向失败；
- 不允许把单个 Demo 片段替代完整验证。

## 15. 新会话强制读取顺序

必须按以下顺序完整读取：

1. 本 V9 主接管文件，从第一行到文件末尾；
2. `doc/repro/session_handoff_v9_teacher_v1_validation_demos_and_interface_realignment_2026-08-30.sha256`；
3. V8 主接管文件及其 manifest；
4. V8 第12节列出的 V7、项目总目标、V6、V5 stairs、HeightScan handoff 及全部 manifests；
5. `doc/RESULTS_PROXY_STUDENT_MODEL10000_CLOSED_LOOP_2026-08-29.md` 与对应 evidence；
6. Teacher v1 training contract、七个 run 的 checkpoint metadata/optimizer、TensorBoard、console 和 GPU CSV；
7. candidate screen 目录中的全部 JSON、CSV、console 和 manifest；
8. cross-seed finalist 目录中的全部 JSON、CSV、console 和 manifest；
9. fixed-command matrix 目录中的全部 147 组 JSON、CSV、console 和 manifest；
10. Demo 根目录中的全部49组 metadata、CSV、console、raw video manifests，以及 `composed_v2` manifest/metadata；
11. 当前 evaluator、Teacher task registration/config/runner、训练/评估/录制/合成脚本；
12. 如果 Teacher v2 或后续目录已存在，先完整只读核验，禁止重复同名工作。

## 16. 新会话首次回复必须明确

首次回复必须明确总结：

1. HeightScan model10000/34秒历史 Demo、foundation model3200 seeds 7/42/123、stairs model2200 恢复语义/最终4999、ascending model900 接受/model1199 拒绝；
2. 47秒 Demo v1 是 multi-checkpoint showcase；boxes model900 只是 bounded proxy teacher；
3. 七 dispatch entries 只有四 unique proxy policies；dispatcher online v2 和 imitation smoke v2 已通过；
4. proxy student model10000 的 SHA、training seed42、10000 updates，以及完整闭环 91.9% bad orientation 的失败结论；
5. 七个 Teacher v1 已从零、seed42、各5000 updates完成，训练证据健康；
6. selected checkpoints 是 flat4999、rough4999、pyramid slope4000、inverted slope4000、pyramid stairs4700、inverted stairs4999、boxes4999；
7. 固定命令矩阵共17,640 episodes；flat通过、slopes基本通过，rough/stairs/boxes仍有问题，inverted stairs yaw 系统性失败；
8. 最终部署接口已经明确为非负 vx + yaw、vy=0；Teacher v1 是更宽命令域基线，不应未经重新对齐直接蒸馏；
9. 49 raw clips、7个独立 Demo + 1总览 v2 已完成，v2 manifest SHA 为 `d8fdd017...`，并明确不是统一策略；
10. 下一步先冻结 v1、确认 Teacher v2 contract，再开发，绝不自行启动训练/评估/录制/合成。

## 17. 已完成事项，禁止重复

- 不重复 proxy student 10000-update 训练、完整 closed-loop matrix 或诊断录制。
- 不重复七个 Teacher v1 的 5000-update 训练。
- 不重复 candidate screen、cross-seed random-command finalists 或 147-run fixed-command matrix。
- 不重复49个 Teacher v1 raw clips。
- 不重复 `composed` v1 或 `composed_v2` 同名视频合成。
- 不覆盖任何上述 manifest、JSON、CSV、console、checkpoint、event、GPU CSV 或视频。

