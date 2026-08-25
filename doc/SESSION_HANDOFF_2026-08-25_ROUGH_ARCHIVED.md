# Unitree Go2 RL：rough 阶段归档后会话接管说明

更新时间：2026-08-25（Asia/Shanghai）

这是本轮长会话结束时的最新接管文件。它承接：

- `doc/SESSION_SUMMARY_2026-08-25_BASELINE_COMPLETE.md`；
- `doc/SESSION_SUMMARY_2026-08-25_ROUGH_COMPLETE.md`；
- `doc/RESULTS_BASELINE.md`；
- `doc/RESULTS_ROUGH.md`。

新会话应先完整读取本文件，再做只读仓库和正式产物检查，不要从聊天记忆猜测当前状态。

## 1. 协作方式与安全边界

默认协作方式：助手每次提供一批明确的终端命令，用户运行后返回完整输出；助手根据实际输出识别 run、分析、总结并指导下一步。只有用户明确说“你直接运行/实现”时，助手才直接执行或修改。

禁止事项：

- 不升级或重新安装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；
- 不覆盖、移动或删除现有 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果和视频；
- 不把评估场景 seed 误称为独立训练 seed；
- 不把 flat baseline、rough pilot、rough 正式 run 或后续新 run 混淆；
- 未经用户明确授权，不直接修改文件或启动训练/评估。

## 2. 当前固定软件与仓库状态

- Ubuntu 22.04.5，kernel 6.8.0-138-generic；
- NVIDIA Driver 580.173.02 Open；
- Conda env：`/home/kxj/miniforge3/envs/unitree_rl`；
- Python 3.11.16；
- PyTorch 2.7.0+cu128；
- Isaac Sim 5.1.0.0；
- Isaac Lab v2.3.2，当前本地 HEAD `39b494e`，基线提交 `37ddf62`；
- RSL-RL 3.1.2；
- Unitree RL Lab upstream baseline `4960b84`；
- Unitree RL Lab 本文件创建前的功能归档 HEAD `7429bc0`；提交本接管文件后以实际 `git log` 为准；
- 本文件创建前两个仓库的 `git status --short` 均为空。

Unitree RL Lab 的关键提交链：

```text
7429bc0 feat: archive reproducible Go2 rough workflow
61843d8 feat: add stratified Go2 rough evaluation
dea17cc feat: add Go2 rough terrain task
2454bee docs: finalize Go2 baseline reproduction
372ae7e feat: add reproducible Go2 evaluation workflow
4960b84 upstream baseline
```

## 3. 已完成的 flat baseline

- Task：`Unitree-Go2-Velocity`；
- 2048 environments、training seed 42、3000 PPO iterations；
- 正式 checkpoint：`model_2999.pt`；
- SHA-256：`a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928`；
- 正式 checkpoint 路径：

```text
logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt
```

Canonical flat evaluation：384 episodes，XY error `0.0756 ± 0.0021 m/s`，timeout `382/384`（99.48%），base-contact termination `0/384`。

Flat baseline 的完整归档由 `doc/repro/final_artifacts_2026-08-25.sha256` 校验。

## 4. 本轮完成的 rough 任务与训练

新增任务 `Unitree-Go2-Velocity-Rough`：

- 10 × 20 terrain grid；
- 精确 50% flat / 50% random-uniform rough；
- random-rough noise range 0.01–0.06 m，step 0.01 m；
- difficulty range 0.0–1.0；
- 训练期 terrain curriculum 启用；
- 保持 45 维 policy observation，不加入 height scan；
- 不包含 stairs、boxes、slopes 或离散障碍。

训练流程完成了 smoke、2048×100 scale check、2048×500 pilot 和正式 2048×3000 run。正式 run：

- Training seed：42；
- Iterations：3000；
- Simulation transitions：147,456,000；
- Wall time：2:11:41；
- Exit code：0；
- Checkpoint 数：31；
- Console/TensorBoard 非有限值：0；
- 后期吞吐约 18.8k steps/s。

正式训练目录：

```text
logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000
```

正式 console 与 GPU 日志：

```text
/home/kxj/rl_ws/unitree_go2_roughmix50_curriculum_formal_seed42_env2048_i3000_2026-08-25_04-40-14.log
/home/kxj/rl_ws/gpu_roughmix50_curriculum_formal_seed42_env2048_i3000_2026-08-25_04-40-14.csv
```

正式 rough checkpoint 不是最后的 `model_2999.pt`，而是独立评估选出的：

```text
logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000/model_2900.pt
SHA-256: 9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889
```

候选 checkpoint 哈希：

```text
model_2400.pt bef2d39df9222038be615f92877e3c594fd0deacb52848f6938edbeba3fa1701
model_2900.pt 9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889
model_2999.pt c3d8f8fd5b24aa78b870f1c90c74ca50271d68b13c41b29123261afc0ed95a38
```

## 5. Rough 正式分层评估

评估协议为 `rough_stratified v1`：精确 50/50 flat/random_rough，两类均覆盖 terrain level 0–9，运行期 terrain/command curriculum 更新关闭，使用确定性 inference policy。

正式矩阵：

- flat `model_2999` control；
- rough `model_2400`、`model_2900`、`model_2999`；
- 评估场景 seeds 7、42、123；
- 每个 run 64 environments × 2 episodes = 128 episodes；
- 共 12 runs、1536 episodes。

全部 12 runs exit 0，非有限值 0，base-contact termination 0。

正式 rough `model_2900` 的 384-episode 指标：

- Reward：`31.8819 ± 0.1171`；
- Overall XY error：`0.0950 ± 0.0008 m/s`；
- Random-rough XY error：`0.1123 ± 0.0023 m/s`；
- Overall yaw error：`0.0702 ± 0.0005 rad/s`；
- Timeout：`381/384`（99.22%）；
- Base contact：`0/384`；
- Bad orientation：`3/384`（0.78%）。

相对 flat-trained control 的相同 random-rough subset：

- Reward 提升 78.0%；
- XY error 降低 55.7%；
- Yaw error 降低 57.5%；
- Action delta 降低 6.8%；
- Mechanical-power proxy 降低 40.7%；
- Timeout rate 增加 43.23 个百分点。

`model_2900.pt` 的 overall/rough XY error、action delta、功率代理和跨 seed XY 波动优于 `model_2999.pt`；`model_2999.pt` 只保留为 reward/yaw 略优的 late-training alternative。

## 6. 本轮固化的复现与归档入口

- `scripts/project/train_go2_rough.sh`：rough smoke/pilot/formal 训练，保存 console/GPU 日志并拒绝覆盖；
- `scripts/project/evaluate_go2.py`：单 checkpoint 的 flat/rough 分层评估；
- `scripts/project/evaluate_go2_rough_matrix.sh`：固定 12-run checkpoint 矩阵，支持 dry-run；
- `scripts/project/audit_go2_rough_evaluation.py`：审计 run/episode 数、checkpoint hash、terrain type/level、assignment、非有限值、termination 和 console fatal marker；
- `scripts/project/archive_go2_rough.py`：创建或验证六个 rough repro 文件，拒绝覆盖。

正式归档文件：

```text
doc/repro/OFFICIAL_ROUGH_CHECKPOINT.txt
doc/repro/rough_checkpoints_2026-08-25.txt
doc/repro/rough_eval_multiseed_2026-08-25.csv
doc/repro/rough_evaluation_inventory_2026-08-25.txt
doc/repro/rough_evaluation_sha256_2026-08-25.txt
doc/repro/rough_final_artifacts_2026-08-25.sha256
```

本轮已验证：

- Shell/Python 语法通过；
- 12-run dry-run 路径正确；
- 1536-episode audit 通过；
- 六个归档文件可以精确重建；
- 48 项 rough 评估哈希、13 项 rough 正式产物哈希及原 flat baseline 哈希全部通过；
- Markdown 本地链接和 whitespace 检查通过。

归档提交：`7429bc0 feat: archive reproducible Go2 rough workflow`，提交后工作区干净。CSV 的 CRLF→LF 信息只是 Git 换行标准化警告，不影响数据、哈希清单或按行归档验证。

## 7. 收到下一轮训练输出时必须执行的分析顺序

用户表示将在下一轮训练结束后，把完整或分段终端输出发送到新会话。本文件没有记录该新 run 的确切 task、seed、iteration、路径或状态，因此新助手不能预先假定它属于哪次训练。

收到输出后先从以下证据识别 run：

1. task 名、run name、训练目录和开始时间；
2. training seed、environment 数、max iterations；
3. console/GPU log 路径；
4. checkpoint 目录与最后保存的 iteration；
5. Git HEAD 和当时工作区状态。

识别完成后，必须明确说明它是 flat baseline、rough seed-42 formal、rough pilot，还是一个新的训练 seed/run，不得把它并入既有 baseline 指标。

随后依次检查：

1. 进程/命令退出状态是否为 0，是否真正回到 shell prompt；
2. checkpoint 数、最后 checkpoint、TensorBoard event、console log、GPU CSV 是否存在且非空；
3. reward、episode length、XY/yaw tracking error 与 curriculum 是否收敛或平台；
4. time-out、bad-orientation、base-contact 等 termination 组成；
5. 吞吐、collection/learning 时间、训练 wall time、GPU 利用率/显存/温度；
6. `nan`、`inf`、Traceback、exception、failed/error 等非有限值或错误；
7. 该 run 是否有资格进入统一评估，候选 checkpoint 应如何选择。

训练统计不能替代统一离线评估。新训练 run 若通过完整性检查，下一步应使用与正式 rough 相同的 `rough_stratified v1` 协议比较；若它是新的 training seed，必须与 seed-42 模型分开标注，并最终按 training seed 聚合，而不是把评估场景 seed 当作训练 seed。

## 8. 当前项目边界与后续优先级

当前完成了 flat baseline 和 50/50 flat/random-uniform rough 扩展，但两者都只有 training seed 42。尚未完成：

- 多训练 seed 的方差评估；
- stairs/boxes/slopes 或任意复杂地形泛化；
- height-scan policy；
- terrain/reward/扰动/观测噪声消融；
- rough 正式视频和 rough policy 导出；
- MuJoCo Sim2Sim；
- Sim2Real 或实机部署。

优先级：先完成至少 3 个 training seeds 的可比训练与评估，再决定 terrain/observation/reward 消融；之后才扩展新地形和 Sim2Sim。

## 9. 新会话首次只读检查

```bash
conda activate unitree_rl
cd /home/kxj/rl_ws/unitree_rl_lab

sed -n '1,420p' doc/SESSION_HANDOFF_2026-08-25_ROUGH_ARCHIVED.md
git log --oneline -6
git status --short
git -C /home/kxj/rl_ws/IsaacLab log --oneline -3
git -C /home/kxj/rl_ws/IsaacLab status --short

scripts/project/archive_go2_rough.py --verify
sha256sum --check doc/repro/rough_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/rough_final_artifacts_2026-08-25.sha256
sha256sum --check doc/repro/final_artifacts_2026-08-25.sha256
```

这些命令均为只读验证。不要为了接管会话而重新训练、重新评估或重新生成归档。

## 10. 新对话推荐开场提示词

```text
我正在继续 Unitree RL Lab 的 Go2 强化学习仿真项目。

请先完整读取最新的本地接管总结：

/home/kxj/rl_ws/unitree_rl_lab/doc/SESSION_HANDOFF_2026-08-25_ROUGH_ARCHIVED.md

然后按文件第 9 节执行只读检查，确认 Unitree RL Lab/IsaacLab 的提交与工作区状态，并验证 flat/rough 正式归档。不要启动训练或评估，也不要修改文件。

请先简要复述：

1. 当前固定的软件与仓库版本；
2. 已完成的 flat baseline、rough 正式训练与各自 official checkpoint；
3. rough_stratified v1 评估协议和最终指标；
4. 已保存的复现脚本、正式归档和项目边界；
5. 收到下一轮训练输出后的检查顺序。

禁止升级或重新安装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；禁止覆盖、移动或删除现有 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果和视频。

继续采用以下协作方式：你每次给我一批明确的终端命令，我运行后把完整结果发给你；你根据实际结果分析、总结并指导下一步。只有我明确说“你直接运行/实现”时，你才直接修改或执行。

我接下来会发送新一轮训练结束后的完整或分段终端输出。请先依据 task、run name、seed、目录和时间识别它属于哪个 run，不要与 flat baseline、rough pilot 或已完成的 rough seed-42 formal run 混淆。然后检查退出状态、checkpoint/TensorBoard/log/GPU 产物、tracking error、curriculum、termination、吞吐、GPU 状态和非有限值，并判断是否进入统一分层评估。
```
