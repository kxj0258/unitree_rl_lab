# Unitree Go2 RL：rough 多训练 Seed 最终会话交接

更新时间：2026-08-25（Asia/Shanghai）

这是本次长对话结束时的最终接管总结。它在既有正式归档上补充了完整的会话脉络、运行身份、检查结论和下一轮协作方法。新会话必须先完整读取本文件，再根据第 10 节执行只读验证；不要依赖旧聊天记忆，也不要为了接管而重新训练或评估。

## 1. 协作方式与不可变安全边界

默认协作方式：助手每次给出一批明确的终端命令，由用户运行并返回完整输出；助手依据实际输出分析、总结并指导下一步。只有用户明确说“你直接运行/实现”时，助手才直接执行或修改。

禁止升级或重新安装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch。禁止覆盖、移动或删除任何已有 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果和视频。不得把接管检查变成训练、评估或归档重建。

## 2. 固定软件与仓库版本

- Ubuntu 22.04.5，kernel 6.8.0-138-generic；
- NVIDIA Driver 580.173.02 Open；
- Conda env：`/home/kxj/miniforge3/envs/unitree_rl`；
- Python 3.11.16；PyTorch 2.7.0+cu128；
- Isaac Sim 5.1.0.0；
- Isaac Lab v2.3.2，仓库 HEAD `39b494e`，基线 `37ddf62`；
- RSL-RL 3.1.2；
- Unitree RL Lab upstream baseline `4960b84`；
- rough 多训练 seed 正式归档提交：`8079665fa05a77986c63cc5d6599dd3464c16377`（`feat: archive Go2 rough multi-training-seed results`）。

本交接文件自身的提交应以新会话执行 `git log` 的实际结果为准。归档提交完成后曾确认 Unitree RL Lab 与 IsaacLab 工作区均干净；新会话仍需重新做只读确认。

## 3. 已完成的 flat baseline 与 seed-42 rough 正式实验

Flat baseline：

- Task：`Unitree-Go2-Velocity`；training seed 42；2048 env；3000 PPO iterations；
- run：`logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000`；
- official：`model_2999.pt`；
- SHA-256：`a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928`；
- canonical evaluation：XY error `0.0756 ± 0.0021 m/s`，yaw error `0.0575 ± 0.0007 rad/s`，timeout `382/384`，base contact `0/384`。

首个 rough formal：

- Task：`Unitree-Go2-Velocity-Rough`；training seed 42；2048 env；3000 iterations；
- 50% flat / 50% random-uniform rough，terrain curriculum，45 维 policy observation，无 height scan；
- run：`logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000`；
- historical official：`model_2900.pt`；
- SHA-256：`9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889`；
- 384-episode official evaluation：reward `31.8819 ± 0.1171`，overall XY `0.0950 ± 0.0008 m/s`，rough XY `0.1123 ± 0.0023 m/s`，yaw `0.0702 ± 0.0005 rad/s`，timeout `381/384`，base contact `0/384`，bad orientation `3/384`。

历史 seed-42 official `model_2900` 保持不变，不因后续固定 iteration 的跨训练 seed 结论而追溯修改。

## 4. 本次对话新增的两个 rough formal run

两个新 run 与 seed-42 使用相同 task、配置和训练预算；各有 147,456,000 transitions、31 checkpoints，退出状态均为 0。

### Training seed 7

- run：`logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_13-53-47_roughmix50_curriculum_formal_seed7_env2048_i3000`；
- console：`/home/kxj/rl_ws/unitree_go2_roughmix50_curriculum_formal_seed7_env2048_i3000_2026-08-25_13-53-40.log`；
- GPU CSV：`/home/kxj/rl_ws/gpu_roughmix50_curriculum_formal_seed7_env2048_i3000_2026-08-25_13-53-40.csv`；
- wall time `2:14:55`，最后约 `18.5k steps/s`；
- checkpoint hashes：`model_2400` `3a04a27fd4ebd50643a6cf894ce5a61e384091f76559bf910290ac94a65517a2`，`model_2900` `d31467c4a0feee120336d3e52b77154e3d96de654891e780601be8ccde2a4c5a`，`model_2999` `353ea7a9d1cfa3e35fdc8678901ede952c54e7f4fb54782ad86fd2f02352f4af`；
- console/TensorBoard non-finite 0；GPU CSV 1629 个有效样本、malformed 0、non-finite 0；稳态 GPU 平均利用率 86.78%，显存约 5.81 GiB，平均 63.57°C、64.62 W；
- official：`model_2999.pt`。

### Training seed 123

- run：`logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_17-23-39_roughmix50_curriculum_formal_seed123_env2048_i3000`；
- console：`/home/kxj/rl_ws/unitree_go2_roughmix50_curriculum_formal_seed123_env2048_i3000_2026-08-25_17-23-32.log`；
- GPU CSV：`/home/kxj/rl_ws/gpu_roughmix50_curriculum_formal_seed123_env2048_i3000_2026-08-25_17-23-32.csv`；
- wall time `1:53:56`，最后约 `22.0k steps/s`；
- checkpoint hashes：`model_2400` `fd6289ba4c35e0ebb8af6167a130f8bc2d4b1b8504865ddfdae8b2f5218d6c91`，`model_2900` `26649d0c4e7edf283e76740e22b4b0e9a7bfa9abf2d23b24a89333cf363dee1d`，`model_2999` `d83a237bbf9918b3f81d269b1ce4684659756ee30c902b384f92d9dfced0945c`；
- console/TensorBoard non-finite 0，GPU 检查正常；
- official：`model_2999.pt`。`model_2900` 只是 tracking alternative，不是该 seed 的 official。

对话中首次运行 GPU CSV 解析器时出现过 `KeyError: 'temperature.gpu'`，原因是粘贴/脚本缩进与字段解析问题；修正解析命令后 CSV 表头、1629 个样本和全部数值均通过。这不是训练错误，也不能记作实验 non-finite/failure。

## 5. `rough_stratified v1` 统一评估协议

- evaluation scenario seeds：7、42、123；它们不是 training seeds；
- 每个 run 为 64 environments × 2 episodes = 128 episodes；
- 精确 50% flat / 50% random_rough；两类均覆盖 terrain level 0–9；
- 运行期 terrain/command curriculum 更新关闭；
- deterministic inference policy；
- 每个 training seed 评估 `model_2400/2900/2999`。

三个 training seeds × 三个 checkpoints × 三个 evaluation scenario seeds = 27 rough runs、3456 episodes。原 seed-42 目录另保留 3 个 flat control runs；三个评估目录合计 120 个文件：

```text
evaluation/rough_stratified_v1_2026-08-25
evaluation/rough_stratified_v1_trainseed7_2026-08-25
evaluation/rough_stratified_v1_trainseed123_2026-08-25
```

27 个 rough runs 全部 exit 0；episode 数、checkpoint hash、terrain type/level、environment assignment、termination 和 console marker 审计通过；非有限值和 base-contact termination 均为 0。

## 6. Official checkpoint 与最终指标

Per-training-seed official：

| Training seed | Official | SHA-256 |
|---:|---|---|
| 42 | `model_2900.pt` | `9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889` |
| 7 | `model_2999.pt` | `353ea7a9d1cfa3e35fdc8678901ede952c54e7f4fb54782ad86fd2f02352f4af` |
| 123 | `model_2999.pt` | `d83a237bbf9918b3f81d269b1ce4684659756ee30c902b384f92d9dfced0945c` |

Per-seed selected official 的 1152 episodes 聚合（mean ± sample SD across independent training seeds）：

- overall reward `32.0299 ± 0.1508`，rough reward `30.5034 ± 0.1521`；
- overall XY `0.0953 ± 0.0010 m/s`，rough XY `0.1124 ± 0.0008 m/s`；
- overall yaw `0.0706 ± 0.0005 rad/s`，rough yaw `0.0802 ± 0.0005 rad/s`；
- overall action delta `0.8207 ± 0.0079`，power proxy `53.57 ± 1.01 W`；
- timeout `1144/1152`（99.31%），base contact `0/1152`，bad orientation `8/1152`（0.69%）。

固定 iteration 主结果推荐 `model_2999`：overall reward `32.0408 ± 0.1351`，overall XY `0.0956 ± 0.0009 m/s`，rough XY `0.1131 ± 0.0013 m/s`，yaw `0.0704 ± 0.0008 rad/s`，timeout `1144/1152`，base contact `0/1152`，bad orientation `8/1152`。固定 `model_2900` 的 XY 只略优（overall `0.0946`、rough `0.1125`），但 reward、yaw、action delta、power、timeout、bad orientation 与跨 training-seed 稳定性整体不如 `model_2999`。

因此必须同时保留两个不冲突的结论：历史 seed-42 official 是 `model_2900`；跨 training-seed 的固定 iteration 主结果推荐 `model_2999`。

## 7. 已保存的复现脚本与正式归档

关键脚本：

- `scripts/project/train_go2_rough.sh`；
- `scripts/project/evaluate_go2.py`；
- `scripts/project/evaluate_go2_rough_matrix.sh`；
- `scripts/project/evaluate_go2_rough_training_seed_matrix.sh`；
- `scripts/project/audit_go2_rough_evaluation.py`；
- `scripts/project/audit_go2_rough_multitraining.py`；
- `scripts/project/archive_go2_rough.py`；
- `scripts/project/archive_go2_rough_multitraining.py`。

结果文档：`doc/RESULTS_ROUGH.md`、`doc/RESULTS_ROUGH_MULTITRAINING.md`。

多训练 seed 六文件正式归档：

```text
doc/repro/OFFICIAL_ROUGH_CHECKPOINTS_MULTITRAINING.txt
doc/repro/rough_multitraining_selected_official_2026-08-25.csv
doc/repro/rough_multitraining_fixed_checkpoints_2026-08-25.csv
doc/repro/rough_multitraining_evaluation_inventory_2026-08-25.txt
doc/repro/rough_multitraining_evaluation_sha256_2026-08-25.txt
doc/repro/rough_multitraining_final_artifacts_2026-08-25.sha256
```

正式归档实现时已通过：27-run/3456-episode audit；120/120 evaluation hashes；35/35 final-artifact hashes；旧 rough 48/48 evaluation、13/13 final-artifact hashes；flat 14/14 hashes；Shell/Python 语法、dry-run、拒绝覆盖、Markdown 本地链接及 `git diff --check`。没有启动训练/评估，也没有移动、覆盖或删除已有产物。

## 8. 当前项目边界与后续优先级

已完成的是 flat baseline，以及三个独立 training seeds 下的 50/50 flat/random-uniform rough 复现、统一分层评估、方差分析与正式归档。

尚未完成：stairs、boxes、slopes 或一般复杂地形；height-scan policy；terrain curriculum / observation noise / reward 等受控消融；rough 正式视频与导出；MuJoCo Sim2Sim；Sim2Real 或实机部署。

推荐下一阶段先设计小规模、一次只改变一个因素的 terrain-curriculum、observation-noise、reward 消融，再扩展新地形。任何新地形必须定义与 `rough_stratified v1` 分离的新评估协议，不能把指标直接混入当前正式结果。

## 9. 收到下一轮训练输出后的检查顺序

收到完整或分段终端输出时，不要先凭指标猜测身份。按证据依次确认：

1. task、run name、training seed、训练目录、开始/结束时间；
2. env 数、max iterations、console/GPU log 路径、Git HEAD；
3. 明确它是 flat baseline、rough pilot、seed-42/7/123 formal，还是全新的 run；
4. 退出状态是否为 0，是否回到 shell prompt；
5. checkpoint 数和最后 checkpoint、TensorBoard event、console log、GPU CSV 是否存在且非空；
6. reward、episode length、XY/yaw tracking error、terrain/command curriculum 的趋势；
7. timeout、bad orientation、base contact 等 termination 组成；
8. throughput、collection/learning time、wall time、GPU utilization/memory/temperature/power；
9. `nan`、`inf`、Traceback、RuntimeError、exception、CUDA/OOM、failed 等非有限值或致命错误；
10. 完整性、稳定性和安全性通过后，才判断是否进入对应的统一分层评估。

训练末尾统计不能替代离线评估。新的 training seed 必须独立标注并按 training seed 聚合；evaluation scenario seed 只能衡量同一 policy 的场景随机性，绝不能当作独立训练样本。

## 10. 新会话首次只读接管命令

```bash
conda activate unitree_rl
cd /home/kxj/rl_ws/unitree_rl_lab

sed -n '1,520p' doc/SESSION_HANDOFF_2026-08-25_ROUGH_MULTITRAINING_FINAL.md
git log --oneline -8
git status --short
git -C /home/kxj/rl_ws/IsaacLab log --oneline -3
git -C /home/kxj/rl_ws/IsaacLab status --short

scripts/project/archive_go2_rough.py --verify
scripts/project/archive_go2_rough_multitraining.py --verify
scripts/project/audit_go2_rough_multitraining.py \
  --verify-selected-csv doc/repro/rough_multitraining_selected_official_2026-08-25.csv \
  --verify-fixed-csv doc/repro/rough_multitraining_fixed_checkpoints_2026-08-25.csv
sha256sum --check doc/repro/rough_multitraining_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/rough_multitraining_final_artifacts_2026-08-25.sha256
```

这些命令只用于接管验证。新助手应先简要复述版本、completed runs、official checkpoints、协议、最终指标、归档和项目边界，再等待用户发送下一批运行结果。

## 11. 建议的新对话首条消息

复制下面文字到新对话即可：

```text
我正在继续 Unitree RL Lab 的 Go2 强化学习仿真项目。

请先完整读取本地最终接管总结：
/home/kxj/rl_ws/unitree_rl_lab/doc/SESSION_HANDOFF_2026-08-25_ROUGH_MULTITRAINING_FINAL.md

然后严格按该文件第 10 节执行只读接管检查，确认 Unitree RL Lab、IsaacLab 的提交与工作区状态，并验证 flat、rough seed-42 和 rough 多训练 seed 正式归档。不要启动训练或评估，不要修改文件。

请先简要复述：固定软件与仓库版本；flat baseline 和三个 rough formal training seeds；各自 official checkpoint；rough_stratified v1 协议与最终指标；复现脚本、正式归档、项目边界和下一阶段优先级。

继续采用以下协作方式：你每次给我一批明确的终端命令，我运行后把完整结果发给你；你依据实际结果分析、总结并指导下一步。只有我明确说“你直接运行/实现”时，你才直接执行或修改。

禁止升级或重新安装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；禁止覆盖、移动或删除已有 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果和视频。

我随后会发送新一轮运行的完整或分段终端输出。请先依据 task、run name、training seed、目录和时间识别它属于哪个 run，不要与 flat baseline、rough pilot 或已完成的 seed-42/7/123 formal runs 混淆。然后按接管总结第 9 节检查退出状态、checkpoint/TensorBoard/log/GPU 产物、tracking error、curriculum、termination、吞吐、GPU 状态和非有限值，并判断是否进入对应的统一分层评估。请始终区分 training seed 与 evaluation scenario seed。
```
