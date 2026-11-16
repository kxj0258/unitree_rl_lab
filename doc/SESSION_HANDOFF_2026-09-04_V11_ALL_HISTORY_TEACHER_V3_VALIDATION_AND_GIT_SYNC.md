# Unitree Go2 项目 V11 主接管：全历史、Teacher v3 验证与 Git 同步

日期：2026-09-04（Asia/Shanghai）

这是当前最新、面向新对话的主接管文件。它以 V9 的完整历史索引为基础，合并 V10 和 2026-08-30 至 2026-09-04 的 Teacher v2/v3 开发、训练、评估、双视角视频及 Git 同步结果。新对话必须从第一行完整读取到文件末尾，再按第 12 节读取历史和证据；不能只读摘要。

## 1. 最终目标与已冻结决策

最终目标是：按照真实可部署命令契约，从零训练并严格验证七个 terrain-specific teachers，再从零训练一个真正统一的 recurrent perceptive student。七类地形为 flat、random rough、pyramid slope、inverted pyramid slope、pyramid stairs、inverted pyramid stairs、boxes。

最终命令契约：

- `vx`：非负前进速度；
- `vy=0`；
- 使用 heading command，目标朝向相对命令采样时当前朝向限制在 ±45°；
- 只允许行进中转向；不要求后退、横移或原地旋转。

Teacher v1 的训练域包含负 `vx`、非零 `vy` 和 yaw-rate，因此与最终接口不完全一致。Teacher v1 必须冻结为可追溯宽域 baseline，不得直接作为最终统一 student 的正式教师集合。

## 2. 永久操作约束

1. 禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch。
2. 禁止覆盖、移动或删除 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果、原始视频或合成视频。
3. 训练、评估、录制和视频合成只能由助手给出完整命令，由用户在自己的终端执行并回传输出；助手不得自行启动。
4. 安装、克隆、联网下载或扩大范围前必须取得用户明确授权。
5. 不清理 dirty worktree；不修改或删除 `env_backup_unitree_rl.yml`。
6. 不重复已完成的训练、评估、录制或同名视频合成。
7. 始终严格区分 training seed 和 evaluation scenario seed。
8. exit code 0 只表示程序正常结束；timeout 不等于有效运动或任务成功。
9. 以后给用户的交互式终端命令末尾禁止使用 `exit "$status"`，否则会关闭用户终端。只打印 status 并返回 shell prompt。
10. 收到终端输出后，先识别 task、variant、run name、training seed、evaluation scenario seed、source/current checkpoint、绝对输出目录和时间；再检查 exit、SHA、iteration/optimizer、TensorBoard、console/GPU、tracking、progress、stall、terrain/level/command 覆盖、净位移、path length、安全指标和非有限值。

## 3. 仓库、环境和 Git 状态

- Unitree RL Lab：`/home/kxj/rl_ws/unitree_rl_lab`
- 当前 HEAD：`896f60823fe26904d77f14ef1dc8b9798b91185d`
- commit：`feat: add seven-terrain teacher-student training pipeline`
- branch：`main`，跟踪 `origin/main`
- origin：`git@github.com:kxj0258/unitree_rl_lab.git`
- upstream：`https://github.com/unitreerobotics/unitree_rl_lab.git`
- GitHub：`https://github.com/kxj0258/unitree_rl_lab`
- 当前唯一未跟踪项：`env_backup_unitree_rl.yml`；必须继续保持未修改、未删除、未提交。
- IsaacLab：`/home/kxj/rl_ws/IsaacLab`
- IsaacLab HEAD：`39b494e6d8ae3f7adbb30ae1feaf45d21bdc7d19`
- IsaacLab 当前 clean。
- Conda env：`/home/kxj/miniforge3/envs/unitree_rl`
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，12227 MiB。
- 最后检查时没有训练、评估、录制或视频合成进程；`packagekitd`/`webkit` 名称匹配不是 Isaac Kit 任务。

2026-09-04 已将 97 个代码、脚本和文档文件提交并推送。checkpoint、logs、evaluation、videos 被排除，没有推送。推送时从官方 upstream 精确补取了历史 LFS 对象 `doc/sim2sim.gif`，OID/SHA-256 `391dde01aa8c308f2ce8b02acfeb612c195f4078aca02e8ecb14052f37be5209`。

## 4. 不可丢失的历史结论

1. HeightScan `model10000` 与历史 34 秒 Demo 已完成并保留。
2. Foundation `model3200` 使用 evaluation scenario seeds 7/42/123 评估选择。
3. Stairs 历史训练使用 training seed 42；从 `model2200` 恢复的语义是继续总 iteration，最终到 `model4999`，不是额外从零训练 4999 次。
4. Ascending `model900` 被接受；`model1199` 因退化被拒绝。
5. 47 秒 Demo v1 是 multi-checkpoint showcase，不是统一策略。
6. Boxes safety `model900` 只是 bounded proxy teacher。
7. 七个旧 dispatch entries 实际只有四个 unique proxy policies。
8. Dispatcher online v2 与 student imitation smoke v2 已通过。
9. Proxy student `model10000` 从零训练、training seed 42、10,000 imitation updates，SHA-256 `f70ce64aa44a54ced84660e51c114d7b105f04f1fdfc876d93fb54eb9912cc5b`。
10. Proxy student imitation loss 下降不代表闭环成功。完整 21-run、2940-episode 验证很差：bad-orientation 91.9048%、timeout 8.0612%、mean command-aligned progress 0.131204 m/s、stall 17.8515%。它是负结果 baseline，不是最终统一策略。
11. 七个 Teacher v1 已分别从零训练 5,000 updates，并完成候选筛选、跨 evaluation seed 随机命令和固定命令矩阵。
12. Teacher v1 Demo composed_v2 已完成：七个独立完整版加一个总览版，保留 KNOWN ISSUE，采用中英双语标题与英文数值 HUD。

Teacher v1 最终候选：flat `model4999`、random rough `model4999`、pyramid slope `model4000`、inverted pyramid slope `model4000`、pyramid stairs `model4700`、inverted pyramid stairs `model4999`、boxes `model4999`。Flat 最稳定；slopes 基本通过；rough/stairs/boxes 有安全失败；inverted stairs 有系统性 yaw 缺陷。完整数值见 V9。

## 5. Teacher v2：接口纠偏和结果

Teacher v2 采用 relative heading command ±45°、非负前进和 `vy=0` 路线。七个任务均已注册。共同工作包括 HeightScan、heading-aligned progress、横向漂移/绕圈抑制、步态和安全奖励。

已接受并保留的 Teacher v2：

| Terrain | Candidate | 状态 |
|---|---:|---|
| flat | `model4000` | gait 人工检查通过，随机命令与固定矩阵完成 |
| random rough | `model4800` | gait 人工检查通过，随机命令与固定矩阵完成 |
| pyramid slope | `model4999` | gait 人工检查通过，随机命令与固定矩阵完成 |
| inverted pyramid slope | `model4700` | gait 人工检查通过，随机命令与固定矩阵完成 |

原始 Pyramid Stairs V2 `model4999` 和 `model3500` 都出现前腿蹦、后腿拖行/左后腿少参与。第一次 remediation 仍长期翘左后腿。Inverted Stairs 和 Boxes remediation 可以运动，但仍有前腿跳跃、后腿拖行或不自然步态。因此不继续在旧 checkpoint 上盲目训练，改为从零 Teacher v3 专项修复。

## 6. Teacher v3：设计、正式训练与 checkpoint

三个 V3 任务：

- `Unitree-Go2-Velocity-TeacherV3-PyramidStairs`
- `Unitree-Go2-Velocity-TeacherV3-InvertedPyramidStairs`
- `Unitree-Go2-Velocity-TeacherV3-Boxes`

核心改进：HeightScan、最弱腿参与约束、四足 duty symmetry、worst loaded-foot slide、terrain-relative swing peak、terrain-scaled gait clock、heading-aligned progress、lateral drift/stall/anti-circle gate；Inverted Stairs 增加 elevation progress；Boxes 增加 support-edge penalty 和低/中等级混合课程。

正式训练均为从零初始化、training seed 42、2048 env、5000 PPO updates、32 steps/env/update、16 mini-batches，每个 327,680,000 transitions。训练无 OOM、NaN/Inf；checkpoint `iter=4999` 且 optimizer state 完整。

| Terrain | Formal run | Final SHA-256 |
|---|---|---|
| Pyramid Stairs | `logs/rsl_rl/unitree_go2_velocity_teacher_v3_pyramid_stairs/2026-09-02_10-57-34_teacher_v3_pyramid_stairs_worstleg_symmetry_heightscan_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1` | `a46577f3f52c3faa4afd69a8a9d238af4717f347f91946d3b4d1c530caabb196` |
| Inverted Stairs | `logs/rsl_rl/unitree_go2_velocity_teacher_v3_inverted_pyramid_stairs/2026-09-02_16-54-00_teacher_v3_inverted_pyramid_stairs_worstleg_symmetry_heightscan_ascent_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1` | `598cd3e28127e302f5e6eff2e0e2aa3fa253a0a57f14cca3af570618ab8ce908` |
| Boxes | `logs/rsl_rl/unitree_go2_velocity_teacher_v3_boxes/2026-09-02_21-52-51_teacher_v3_boxes_worstleg_symmetry_edgeaware_mixedlevel_heightscan_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1` | `db5148a5daa4c21d24af7513ba49578354dfbccfbfaddb6b4107474f5a3e48ac` |

2048-env 配置在正式训练前通过 1-update 和 10-update 稳定性测试。早期第一次失败不是显存问题，而是 V3 curriculum 函数签名漏掉 `duty_symmetry_term`/`stall_term`；代码已修复，失败证据保留。

## 7. Teacher v3 candidate screen

协议：training seed 42 的固定 checkpoints；evaluation scenario seed 123；随机命令；`teacher_v3_stratified`；100 env × 2 episodes；单地形 level 0--9 均匀覆盖；运行期 curriculum 冻结。

输出根目录：

`evaluation/teacher_v3_candidate_screen_trainseed42_evalseed123_randomcommand_2026-09-03`

九个 candidates、1,800 episodes 全部完成；CSV 无非有限值；manifest 全部校验通过。manifest 自身 SHA-256：

`bce0bf9fd69775e99d7f12accd5f01dce0af57568541826abdf2af3bddcc1919`

选择：

- Pyramid Stairs `model4999`：200/200 timeout、zero safety termination、XY error 0.0431 m/s。
- Inverted Stairs `model4700`：200/200 timeout、zero safety termination、XY error 0.0447 m/s；SHA-256 `de8250790ae1e04437d014803edce17771045cd80e73735e37b81e79326fc640`。
- Boxes 暂选 `model4999`，但 timeout 82.5%、bad orientation 16%，只是待 gait gate 的 provisional candidate。

## 8. Teacher v3 双视角 gait gate

协议：evaluation scenario seed 42、terrain level 5、`vx=0.40 m/s`、`vy=0`、relative heading 0、20 秒、warmup 100 steps。每个候选同时录制：

- `terrain_fixed`：观察路径、爬升、越顶和越障；
- `asset_follow`：观察四腿参与、拖行、翘腿和跳跃。

总 manifest：`evaluation/teacher_v3_dual_view_gait_gate_trainseed42_evalseed42_2026-09-03.sha256`；SHA-256 `5926ade0400d7695b0549b57538e29d0d0c86a30021d93bb788b19cdd5c09775`，所有条目通过。

| Candidate | Signed progress | Path length/efficiency | Stall | Duty spread | Safety/result |
|---|---:|---:|---:|---:|---|
| Pyramid Stairs `model4999` | 7.398 m | 7.436 m / 99.49% | 0% | 0.053 | 正常 timeout；数值通过，人工视频尚待用户确认 |
| Inverted Stairs `model4700` | 7.327 m | 7.386 m / 99.21% | 0% | 0.057 | 正常 timeout；数值通过，人工视频尚待用户确认 |
| Boxes `model4999` | -0.043 m | 7.378 m / -0.58% | 11.11% | 0.110 | bad-orientation termination；失败 |

Boxes 虽有 7.378 m path length，却没有有效前向净进展，因此不能以“在运动”或 timeout 代替成功。它可能绕行、反复偏转、回退、踩空或摔倒，必须结合固定/跟随视频定位。

当前对话尚未收到用户对六段最新视频的人工观察结论。新对话首先应让用户报告两种楼梯是否仍有后腿拖行/翘起、前腿跳跃，以及 Boxes 的具体失败方式；在此之前不要开始 cross-seed/fixed matrix，也不要训练 student。

## 9. 当前代码与文档状态

主要实现：

- `source/.../go2/terrain_teacher_v1_env_cfg.py`
- `source/.../go2/terrain_teacher_v2_env_cfg.py`（含 V2 remediation 和 V3 classes）
- `source/.../mdp/rewards.py`
- `source/.../mdp/curriculums.py`
- `source/.../mdp/commands/velocity_command.py`
- `source/.../agents/rsl_rl_ppo_cfg.py`
- `scripts/project/evaluate_go2.py`（含 `teacher_v1/v2/v3_stratified`）
- `scripts/project/train_go2_discrete_teacher_v3_serial.sh`
- `scripts/project/evaluate_go2_discrete_teacher_v3_candidates.sh`
- `scripts/project/record_go2_discrete_teacher_v3_dual_view.sh`
- `scripts/project/analyze_go2_gait_contacts.py`

最新结果文档：`doc/RESULTS_TEACHER_V2_V3_2026-09-04.md`。README、PROJECT_STATUS、REPRODUCTION、RESUME_PROJECT 已在 commit `896f608` 更新并推送。

V9 manifest 目前对 `scripts/project/evaluate_go2.py` 和 `scripts/project/record_go2_multiterrain_demo.py` 报 checksum mismatch，这是因为二者在 V9 后继续演进，不表示历史评估产物损坏。禁止改写 V9 manifest；V9 文档、V8 链和旧 evidence manifests 仍保留。V10 manifest 对 V10 文档集当前全部通过。

## 10. 下一步严格顺序

1. 获取用户对六段 V3 双视角视频的人工结论。
2. Pyramid/Inverted Stairs：只有数值与人工 gait 都通过，才进入 evaluation scenario seeds 7/42/123 random-command 与 fixed-command matrix。
3. Boxes：先根据视频和 tracking 定位失败，设计最小受控修复；不得直接接受 `model4999`，也不得无分析地继续训练。
4. 为所有合格结果生成不可覆盖的 metadata/CSV/console/video SHA manifests。
5. 最终形成满足同一部署命令契约的七 teacher registry。
6. 七个 teacher 均通过后，从零训练真正统一的 recurrent perceptive student；不得复用失败 proxy student 冒充最终策略。
7. 最终 student 必须做七地形、levels、commands、seeds、mixed transitions、hidden reset、safety 和视频的完整闭环验证。

## 11. 收到新终端输出的固定处理模板

首先逐项识别：

- task、terrain、variant、run name；
- training seed 与 evaluation scenario seed（分开）；
- initialization/source checkpoint/current checkpoint；
- checkpoint iteration、SHA 和 optimizer；
- 绝对输出目录、起止时间、exit status。

随后检查：

- TensorBoard 前/中/后趋势、候选峰值、非有限值；
- console traceback/OOM/CUDA、GPU utilization/memory/temperature/power；
- episode/terrain type/level/command 覆盖；
- command versus actual velocity/heading；
- command-aligned progress、世界净位移、path length/efficiency、stall；
- base contact、bad orientation、timeout；
- 四足 duty factor、最大连续 air/contact、loaded-foot slide、diagonal mismatch；
- 不允许总体均值掩盖高 level 或特定命令失败；
- 最后给出证据边界、是否通过和下一条完整命令。命令不得自动关闭终端。

## 12. 新对话强制读取顺序

新对话开始后，必须要求助手在采取任何行动前完成只读接管：

1. 完整读取本 V11，从第一行到文件末尾；
2. 校验 `doc/repro/session_handoff_v11_all_history_teacher_v3_validation_and_git_sync_2026-09-04.sha256`；
3. 完整读取 V10、`doc/RESULTS_TEACHER_V2_V3_2026-09-04.md` 和 V10 manifest；
4. 完整读取 V9 和 V9 第 15 节规定的 V8、V7、项目总目标、V6/V5/HeightScan 历史 handoff；
5. 读取 proxy student closed-loop、Teacher v1 训练/三阶段评估、Demo v1/v2 manifests；
6. 读取 Teacher v2/v3 当前代码、formal run params/TensorBoard/checkpoints、candidate screen 和 dual-view gait evidence；
7. 只读确认 Unitree RL Lab/IsaacLab HEAD 与 dirty、remote、所有关键 SHA 和残留进程；
8. 不启动训练、评估、录制或合成，先向用户复述接管状态并等待最新人工视频观察或运行输出。

## 13. 新对话必须明确复述的核心事实

首次回复至少明确：HeightScan model10000/34秒 Demo；foundation model3200/seeds 7/42/123；stairs seed42/model2200 resume/model4999；ascending 900 接受/1199 拒绝；47秒 Demo 为 multi-checkpoint；boxes safety 900 为 bounded proxy；七 dispatch 仅四 unique；dispatcher online v2 与 imitation smoke v2 通过；proxy student model10000 的 seed/updates/SHA 和闭环失败；Teacher v1 七个从零训练及冻结原因；最终 heading ±45°/非负 vx/vy=0 接口；Teacher v2 四个接受候选；Teacher v3 三个正式训练和候选；两个 stairs 数值 gait 通过但人工检查待定；Boxes V3 gait 失败；最终目标仍是真正统一 recurrent perceptive student。

## 14. 可直接复制到新对话的提示词

```text
我正在继续 /home/kxj/rl_ws/unitree_rl_lab 的 Unitree Go2 七地形 Teacher–Student 强化学习项目。

请先完整读取最新主接管文件：
/home/kxj/rl_ws/unitree_rl_lab/doc/SESSION_HANDOFF_2026-09-04_V11_ALL_HISTORY_TEACHER_V3_VALIDATION_AND_GIT_SYNC.md

必须从第一行读取到文件末尾，不能只读摘要或局部章节。然后读取并校验：
/home/kxj/rl_ws/unitree_rl_lab/doc/repro/session_handoff_v11_all_history_teacher_v3_validation_and_git_sync_2026-09-04.sha256

请严格按照 V11 第12节继续读取 V10、V9、V8、V7、项目总目标、V6/V5/HeightScan 历史 handoff、proxy student 闭环证据、Teacher v1/v2/v3 的训练与全部评估证据、Demo manifests 和当前相关代码。不要跳过历史文件或证据目录。

开始工作前做必要只读检查：Unitree RL Lab 与 IsaacLab HEAD/dirty、remote、关键 checkpoint SHA/iteration/optimizer、TensorBoard、candidate screen、双视角 gait evidence、manifest 和残留进程。完成接管后不要自行启动训练、评估、录制或合成，先总结当前状态并等待我发送最新人工视频观察或终端运行结果。

永久约束：禁止升级或重装 NVIDIA/Isaac Sim/Isaac Lab/PyTorch；禁止覆盖、移动或删除 checkpoint、日志、GPU CSV、TensorBoard、评估和视频；运行任务必须由你给完整命令、我执行并回传；安装/克隆/下载前先授权；不清理 dirty worktree；不修改或删除 env_backup_unitree_rl.yml；不重复已完成运行；严格区分 training seed 与 evaluation scenario seed；timeout 不等于成功；以后给我的命令不能在末尾使用 exit，必须保留终端窗口。

以后每次收到终端输出，先识别 task、variant、run name、training seed、evaluation scenario seed、source/current checkpoint、绝对目录和时间，再检查 exit、SHA、iteration/optimizer、TensorBoard、console/GPU、tracking、coverage、progress、净位移、path length、stall、安全、四足 gait 和非有限值，然后基于完整本地证据总结并指导下一步。
```
