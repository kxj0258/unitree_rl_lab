# Go2 MultiTerrain 续训、对照与长视频阶段归档

## 1. 范围与可复现性边界

本阶段从 MultiTerrain training seed 42 的 `model_2995.pt` 分叉完成两组各 500 iterations 的续训：原 survival curriculum 基线，以及新增 tracking-aware curriculum。所有正式对比均使用冻结 curriculum 的 `multiterrain_stratified` v1 协议：七类 terrain、每类 level 0--9、70 个并行环境、每环境 2 episodes，共 140 episodes。

training seed 始终是 42；evaluation scenario seed 单独记录，绝不将其表述为训练随机种子。训练、评估和视频产物保留在 Git 忽略的 `logs/`、`evaluation/` 及 workspace 日志位置；Git 归档代码、结论和 SHA-256 清单。

## 2. 代码与 Git

- `7f10ca4`：新增独立 tracking-aware MultiTerrain curriculum 和 task variant；
- `506e1ed`：20 秒可审计 Go2 视频及逐步 CSV；
- `b089a3a`：两个 task variant 共享原 MultiTerrain experiment root；
- `2f7df96`：修复 RSL-RL `--experiment_name` 覆盖未生效；
- `8f2178c`：按实际 terrain origin 设置固定镜头；
- `1d3a154`：动态 command/actual vx、误差、距离和 termination 视频叠加；
- `929d435`：恢复适合作品集展示的 asset-follow 构图。

原 survival task 行为未被 tracking-aware variant 改写。Python compile、Shell `bash -n`、`git diff --check` 和旧 CSV 兼容读取均通过。

## 3. 训练产物

### Survival baseline

- run：`2026-08-26_04-32-39_multiterrain_v3_challenging_noheightscan_baseline_continue_from2995_seed42_env1024_addi500`；
- training seed：42；num envs：1024；起点：原 run 的 `model_2995.pt`；
- 候选：`model_3300.pt` SHA-256 `5756f08ca4df34d71bdd5d25fa3763b91c96c4af59f60c7598eec822322b33b3`；
- 最终：`model_3494.pt` SHA-256 `53d7caed903d9b7489468ae376a77b8ef70b6fc8352509aabfaa2807c3ec8fee`。

训练实际完成至 iteration 3494 并生成 TensorBoard 与全部 checkpoint。由于训练期间修改了仍在执行的 Shell 文件，训练进程结束后 wrapper 读到了错位命令并返回 127，且预期 console 文件只保留了该包装错误；因此该 run 的训练过程主要由终端输出、TensorBoard、checkpoint 和 GPU CSV 佐证，不能宣称 console log 完整。这不影响已落盘 checkpoint 的固定协议评估，但属于必须保留的审计限制。

### Tracking-aware branch

- run：`2026-08-26_04-58-48_multiterrain_v3_trackingaware_thr065_continue_from2995_seed42_env1024_addi500`；
- training seed：42；num envs：1024；起点同为原 `model_2995.pt`；
- curriculum：完整存活且线速度 tracking reward 达到最大加权奖励的 65% 才晋级；完整存活但跟踪不足或前半程失败则降级；
- `model_3300.pt` SHA-256 `25711e028fcb84bc9aa79af180c97b7bb5023da6eeb69131af8abafd66b85d12`；
- `model_3400.pt` SHA-256 `6b9f2efd8fa110ad1fffebb0fcd6a963a0eae233459b9783e88c22bd48e3e0d5`；
- `model_3494.pt` SHA-256 `e3ae8fdd2d8840f011541860f2645d613ed5ed60151c64a3b7dc43ccf9d2ba79`；
- 正式训练 exit code 0，console/TensorBoard/GPU CSV/checkpoint 完整，无数值型 NaN/Inf。

此前两次 tracking-aware 启动均在优化开始前因 experiment root 解析失败而安全退出，没有产生训练 checkpoint；失败日志保留用于诊断，随后由 `b089a3a` 和 `2f7df96` 修复。

## 4. Evaluation scenario seed 42 候选比较

| Branch/checkpoint | Command XY | XY error | Yaw error | Timeout | Base contact | Bad orientation |
|---|---:|---:|---:|---:|---:|---:|
| 原 `model_2995` | 0.536 | 0.292 | 0.165 | 77.1% | 0.0% | 22.9% |
| baseline `model_3300` | 0.515 | **0.270** | 0.178 | **83.6%** | 1.4% | 15.0% |
| baseline `model_3494` | 0.502 | 0.263 | **0.140** | 72.1% | 0.7% | 27.9% |
| tracking-aware `model_3300` | 0.533 | 0.271 | 0.169 | 70.7% | 0.0% | 29.3% |
| tracking-aware `model_3400` | 0.513 | 0.282 | 0.183 | 77.1% | 0.0% | 22.9% |
| tracking-aware `model_3494` | 0.529 | 0.283 | 0.175 | 72.9% | 0.0% | 27.1% |

baseline `model_3300` 是本阶段最佳折中：相对原 `model_2995`，XY error 下降约 7.5%，timeout 提升 6.4 个百分点。baseline `model_3494` 虽有更低瞬时跟踪误差，但 boxes 和 pyramid stairs 的姿态失败使整体存活显著下降。tracking-aware curriculum 没有超过 baseline `model_3300`，说明单独改变 terrain 晋级规则不足以解决跟踪与稳定性的冲突。

## 5. 入选 checkpoint 的多 evaluation-seed 复核

baseline `model_3300`、training seed 42：

| Evaluation scenario seed | Command XY | XY error | Yaw error | Timeout | Bad orientation |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.490 | 0.264 | 0.177 | 79.3% | 19.3% |
| 42 | 0.515 | 0.270 | 0.178 | 83.6% | 15.0% |
| 123 | 0.544 | 0.283 | 0.197 | 75.0% | 22.1% |
| mean | 0.516 | 0.272 | 0.184 | 79.3% | 18.8% |
| sample std | 0.027 | 0.010 | 0.011 | 4.3 pp | 3.6 pp |

跟踪误差跨 scenario seed 较稳定；稳定性仍存在约 4 个百分点的场景波动。这里的 7、42、123 全部是 evaluation scenario seeds，不是新增 training seeds。

## 6. 最终长视频 Demo

### MultiTerrain 七地形

- checkpoint：baseline `model_3300.pt`，training seed 42；
- evaluation scenario seed 42；level 5；固定 `vx=0.35 m/s`；
- 七地形各 1000 steps/20 秒，asset-follow 镜头；
- 动态成片：141.86 秒、1280x720、50 FPS、7093 帧全部可解码；
- SHA-256：`5b64a6722651b33977048453c57f20017c39754985c45ad13a41ccbbe9f93939`。

| Terrain | Post-warmup actual vx | XY error | Distance | Termination |
|---|---:|---:|---:|---|
| flat | 0.335 m/s | 0.060 m/s | 6.46 m | timeout |
| pyramid slope | 0.224 | 0.186 | 4.76 m | timeout |
| random rough | 0.152 | 0.220 | 3.40 m | timeout |
| pyramid stairs | 0.077 | 0.288 | 2.46 m | bad orientation |
| inverted pyramid slope | 0.064 | 0.294 | 2.43 m | timeout |
| boxes | 0.040 | 0.331 | 2.81 m | timeout |
| inverted pyramid stairs | 0.023 | 0.339 | 2.64 m | timeout |

该 Demo 清楚显示：flat 跟踪优秀，slope 尚可；rough、stairs、boxes 上策略偏向保守存活，真实速度明显低于命令。它适合展示多地形覆盖、评估体系和当前技术瓶颈，不应包装成“所有地形高速精确跟踪”。

### Rough 三 training-seed

- training seeds：42、7、123，各自 `model_2999.pt`；
- 共同 evaluation scenario seed：42；random rough level 5；固定 `vx=0.35 m/s`；
- 各 1000 steps/20 秒，三段均 timeout，无 base contact/bad orientation；
- 动态成片：61.94 秒、1280x720、50 FPS、3097 帧全部可解码；
- SHA-256：`3114114451694c63535542d79340dfb7e4328e89189622e018ab45ed51972b42`。

| Training seed | Actual vx | XY error | Distance |
|---:|---:|---:|---:|
| 42 | 0.249 m/s | 0.117 m/s | 5.03 m |
| 7 | 0.271 | 0.106 | 5.46 m |
| 123 | 0.257 | 0.107 | 5.15 m |

Rough 专项策略在相同固定场景下明显优于 MultiTerrain 策略的 random rough clip，且三 training seeds 一致性良好。简历主 Demo 可优先使用 Rough；MultiTerrain 作为覆盖能力与系统性实验的技术 Demo。

## 7. 清单与下一阶段

本阶段选定 checkpoint、summary JSON 和最终动态视频的路径/哈希见 `doc/repro/go2_multiterrain_stage2_final_artifacts_2026-08-26.sha256`。原始视频目录自带完整 `final_artifacts.sha256`，已逐项验证。

下一阶段不应继续无目标追加 iterations。优先方案：

1. 将 command curriculum 与 terrain curriculum 解耦，避免后段命令扩展与高 terrain level 同时发生；
2. 对 boxes/stairs 使用 terrain-aware 采样或失败回放，并调整 tracking/survival 奖励权衡；
3. 以 baseline `model_3300` 为主 checkpoint 做受控消融，仍用同一 `multiterrain_stratified` 协议选择；
4. 发布作品集时制作 30--45 秒 highlight，保留当前长版作为完整证据；
5. 之后再推进 MuJoCo Sim2Sim，不升级或重装 NVIDIA/Isaac/PyTorch 栈。
