# Unitree Go2 RL Rough：会话总结与接管说明

更新时间：2026-08-25（Asia/Shanghai）

本文件承接 `SESSION_SUMMARY_2026-08-25_BASELINE_COMPLETE.md`。Flat baseline 与 50/50 flat/random-uniform rough 阶段均已完成；新会话应先读取本文件，再读取 `RESULTS_BASELINE.md`、`RESULTS_ROUGH.md` 和 `PROJECT_STATUS.md`。

## 1. 协作与安全边界

默认协作方式：助手提供一批明确命令，用户运行并返回完整输出；只有用户明确说“你直接运行/实现”时，助手才直接执行或修改。

禁止升级或重新安装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch；禁止覆盖、移动或删除现有 checkpoint、训练日志、GPU CSV、评估结果和视频。

## 2. 固定软件与代码版本

- Ubuntu 22.04.5，kernel 6.8.0-138-generic；
- NVIDIA Driver 580.173.02 Open；
- Conda env：`/home/kxj/miniforge3/envs/unitree_rl`；
- Python 3.11.16；
- PyTorch 2.7.0+cu128；
- Isaac Sim 5.1.0.0；
- Isaac Lab v2.3.2，兼容分支基于 `37ddf626871758333d6ed89cf64ad702aef127d0`；
- Unitree RL Lab upstream baseline：`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`；
- Rough task commit：`dea17cc`；
- Rough stratified evaluator commit：`61843d8`；
- RSL-RL 3.1.2。

## 3. 已完成的两个正式实验

### Flat baseline

- Task：`Unitree-Go2-Velocity`；
- 2048 env、seed 42、3000 iterations；
- 正式 checkpoint：`model_2999.pt`；
- SHA-256：`a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928`；
- Canonical flat evaluation：0.0756 ± 0.0021 m/s XY error，99.48% timeout。

### Rough 50/50

- Task：`Unitree-Go2-Velocity-Rough`；
- Terrain：50% flat、50% random-uniform rough，10 × 20 grid；
- 2048 env、seed 42、3000 iterations、147,456,000 transitions；
- Wall time：2:11:41；exit 0；non-finite 0；
- 正式 checkpoint：`model_2900.pt`；
- SHA-256：`9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889`。

## 4. Rough 分层评估

正式矩阵：flat `model_2999` control + rough `model_2400/2900/2999`，评估场景 seeds 7/42/123，每个 run 64 env × 2 episodes。共 12 runs、1536 episodes。

协议：`rough_stratified v1`，精确 50/50 flat/random_rough，两类均覆盖 level 0–9，terrain generator seed 等于场景 seed，运行期 terrain/command curriculum 更新关闭。

全部 12 runs exit 0；非有限值 0；base-contact termination 0。

正式 rough `model_2900` 的 384-episode 指标：

- Reward：31.8819 ± 0.1171；
- Overall XY error：0.0950 ± 0.0008 m/s；
- Random-rough XY error：0.1123 ± 0.0023 m/s；
- Overall timeout：381/384，99.22%；
- Base contact：0/384；
- Bad orientation：3/384。

## 5. 已固化的复现入口

- `scripts/project/train_go2_rough.sh`：rough smoke/pilot/formal 训练与 GPU/console 日志；
- `scripts/project/evaluate_go2.py`：单 checkpoint 分层评估；
- `scripts/project/evaluate_go2_rough_matrix.sh`：正式 12-run 矩阵；
- `scripts/project/audit_go2_rough_evaluation.py`：1536-episode 审计与聚合；
- `scripts/project/archive_go2_rough.py`：六个 rough repro 文件的 create/verify。

大型评估产物在 `evaluation/rough_stratified_v1_2026-08-25`，默认被 Git 忽略。可追踪聚合、inventory、official marker 和 SHA manifests 位于 `doc/repro/`。

## 6. 当前边界与下一阶段

已完成的 rough 只包含 flat/random-uniform rough，且只有一个训练 seed。评估 seeds 7/42/123 不是多训练 seed。尚未完成 stairs/boxes/slopes、height scan、消融、多训练 seed、rough 视频/导出、MuJoCo Sim2Sim 和 Sim2Real。

下一阶段优先级：

1. 补充至少 3 个训练 seed，量化训练方差；
2. 决定是否扩展 stairs/boxes/slopes，并为新增地形定义独立协议；
3. 做 terrain curriculum、扰动、观测与 reward 消融；
4. 完成 rough 正式视频/ONNX 导出与 MuJoCo Sim2Sim 回归；
5. 只有通过 Sim2Sim 和安全审查后再讨论实机。

## 7. 新会话首次检查

```bash
conda activate unitree_rl
cd /home/kxj/rl_ws/unitree_rl_lab

sed -n '1,320p' doc/SESSION_SUMMARY_2026-08-25_ROUGH_COMPLETE.md
git log --oneline -5
git status --short
git -C /home/kxj/rl_ws/IsaacLab log --oneline -3
git -C /home/kxj/rl_ws/IsaacLab status --short
scripts/project/archive_go2_rough.py --verify
```
