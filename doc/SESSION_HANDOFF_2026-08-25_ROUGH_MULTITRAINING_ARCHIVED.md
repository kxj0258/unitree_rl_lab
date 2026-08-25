# Unitree Go2 RL：rough 多训练 Seed 归档后接管说明

更新时间：2026-08-25（Asia/Shanghai）

这是三个 rough training seeds 全部训练、评估、审计和归档后的最新接管文件。新会话应先读本文件，再读 `RESULTS_ROUGH_MULTITRAINING.md`；不要从旧聊天记忆推断当前状态。

## 1. 协作与安全边界

默认方式：助手给出一批明确终端命令，用户运行并返回完整输出；只有用户明确说“你直接运行/实现”时，助手才直接执行或修改。

禁止升级或重新安装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；禁止覆盖、移动或删除 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果和视频。

## 2. 固定软件与仓库

- Ubuntu 22.04.5，kernel 6.8.0-138-generic；
- NVIDIA Driver 580.173.02 Open；
- Conda env：`/home/kxj/miniforge3/envs/unitree_rl`；
- Python 3.11.16；PyTorch 2.7.0+cu128；
- Isaac Sim 5.1.0.0；
- Isaac Lab v2.3.2，HEAD `39b494e`，基线 `37ddf62`；
- RSL-RL 3.1.2；
- Unitree RL Lab upstream baseline `4960b84`；
- 多 seed 归档前 Unitree RL Lab HEAD `6689e76`，归档提交以实际 `git log` 为准。

## 3. 已完成训练

Flat baseline 保持不变：training seed 42、official `model_2999.pt`，SHA-256 `a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928`。

Rough 三个 formal training seeds 均为 2048 env、3000 iterations、147,456,000 transitions：

| Training seed | Wall time | Official | SHA-256 |
|---:|---:|---|---|
| 42 | 2:11:41 | `model_2900.pt` | `9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889` |
| 7 | 2:14:55 | `model_2999.pt` | `353ea7a9d1cfa3e35fdc8678901ede952c54e7f4fb54782ad86fd2f02352f4af` |
| 123 | 1:53:56 | `model_2999.pt` | `d83a237bbf9918b3f81d269b1ce4684659756ee30c902b384f92d9dfced0945c` |

三个 formal run 共 442,368,000 transitions、93 checkpoints，exit 均为 0，console/TensorBoard 非有限值均为 0。

## 4. 评估与最终结论

协议为 `rough_stratified v1`。三个 training seeds × 三个 checkpoints × 三个 evaluation scenario seeds，共 27 rough runs、3456 episodes；另保留原 flat control 3 runs。所有 rough runs 审计通过，base-contact termination 和非有限值均为 0。

Per-seed selected official 的 1152 episodes 聚合：

- Reward：`32.0299 ± 0.1508`；
- Overall XY：`0.0953 ± 0.0010 m/s`；
- Rough XY：`0.1124 ± 0.0008 m/s`；
- Overall yaw：`0.0706 ± 0.0005 rad/s`；
- Timeout：`1144/1152`（99.31%）；
- Base contact：`0/1152`；
- Bad orientation：`8/1152`（0.69%）。

固定 iteration 比较推荐 `model_2999`：它除 XY 略逊于 `model_2900` 外，在 reward、yaw、action delta、power、timeout、bad orientation 与跨 training-seed 稳定性上整体更优。历史 seed-42 official `model_2900` 不追溯修改。

## 5. 新增复现与归档入口

- `scripts/project/evaluate_go2_rough_training_seed_matrix.sh`：单个 training seed 的 9-run 矩阵，明确区分 training/evaluation seed，支持 dry-run 并拒绝覆盖；
- `scripts/project/audit_go2_rough_multitraining.py`：27-run、3456-episode 审计与 selected/fixed CSV 重建；
- `scripts/project/archive_go2_rough_multitraining.py`：新的六文件 create/verify，拒绝覆盖；
- `doc/RESULTS_ROUGH_MULTITRAINING.md`：完整结果与边界。

新归档文件：

```text
doc/repro/OFFICIAL_ROUGH_CHECKPOINTS_MULTITRAINING.txt
doc/repro/rough_multitraining_selected_official_2026-08-25.csv
doc/repro/rough_multitraining_fixed_checkpoints_2026-08-25.csv
doc/repro/rough_multitraining_evaluation_inventory_2026-08-25.txt
doc/repro/rough_multitraining_evaluation_sha256_2026-08-25.txt
doc/repro/rough_multitraining_final_artifacts_2026-08-25.sha256
```

旧 seed-42 六文件归档和 `archive_go2_rough.py` 保持原语义并继续通过验证。

## 6. 当前边界与后续优先级

多 training-seed 首要目标已经完成。仍未完成复杂地形、height scan、消融、rough 视频/导出、MuJoCo Sim2Sim、Sim2Real 或实机。

推荐下一阶段先设计小规模、一次只改变一个因素的 terrain curriculum / observation-noise / reward 消融；之后再扩展 stairs、boxes、slopes。扩展地形前必须定义与 `rough_stratified v1` 分离的新评估协议。

## 7. 新会话首次只读检查

```bash
conda activate unitree_rl
cd /home/kxj/rl_ws/unitree_rl_lab

sed -n '1,420p' doc/SESSION_HANDOFF_2026-08-25_ROUGH_MULTITRAINING_ARCHIVED.md
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

不要为了接管而重新训练、重新评估或重新生成归档。
