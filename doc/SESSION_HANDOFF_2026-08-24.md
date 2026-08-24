# Unitree RL Lab 仿真复现：会话交接记录

更新时间：2026-08-25（Asia/Shanghai）

## 0. 2026-08-25 Baseline 收尾更新（优先读取）

本节替代下文第 8、10、11 节中的历史“正在训练/待评估”状态；下文旧记录保留用于还原过程。

### 正式训练已完成

- Task：`Unitree-Go2-Velocity`
- Environments：2048
- 训练 seed：42
- PPO iterations：3000
- Transitions：147,456,000
- 训练耗时：2:02:05
- 退出码：0
- 训练目录：`/home/kxj/rl_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000`

### 正式 checkpoint 已确定

```text
model_2999.pt
SHA-256: a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928
```

候选 `model_2400`、`model_2900`、`model_2999` 已分别在评估场景 seed 7/42/123 下完成 64 env × 2 episode 评估，共 1152 个 episode。正式模型 `model_2999` 覆盖 384 个 episode：

- reward：34.6861 ± 0.0358；
- XY velocity error：0.0756 ± 0.0021 m/s；
- yaw velocity error：0.0575 ± 0.0007 rad/s；
- 382/384 episode 达到 20 秒时限；
- base-contact termination：0/384；
- bad-orientation termination：2/384。

三个 seed 是评估场景 seed；训练仍只有 seed 42。

### 工程化收尾已完成

- `scripts/project/train_go2_baseline.sh`
- `scripts/project/analyze_tensorboard.py`
- `scripts/project/evaluate_go2.py`
- `scripts/project/record_go2_video.sh`
- `doc/PROJECT_STATUS.md`
- `doc/REPRODUCTION.md`
- `doc/RESULTS_BASELINE.md`
- `doc/RESUME_PROJECT.md`
- `doc/repro/OFFICIAL_BASELINE_CHECKPOINT.txt`
- `doc/repro/final_artifacts_2026-08-25.sha256`

正式 follow/grid 视频与 TorchScript/ONNX 已生成并人工验收。大文件保留在本地 `evaluation/`，不提交 Git。

### 当前下一阶段

Baseline 阶段已结束，不要重复训练、评估或重新录制。下一阶段应从复杂地形配置审计开始，然后设计 terrain curriculum 改进、多训练 seed 和消融对比。当前尚未完成复杂地形、MuJoCo Sim2Sim 或 Sim2Real。

## 1. 项目目标

- 面向机器人强化学习 / 具身智能算法岗秋招。
- 复现 Unitree RL Lab 的 Go2 locomotion，暂不做实机。
- 计划形成：官方基线、复杂地形/课程学习改进、自动评测、消融实验；时间允许再做 MuJoCo Sim2Sim。
- 当前采用官方 Go2 USD 资产。URDF 仅作为后续可选对照，不作为首轮基线。

## 2. 本机硬件与系统

- Ubuntu 22.04.5 LTS，内核 `6.8.0-138-generic`
- CPU：AMD Ryzen 9 8940HX，16 核 / 32 线程
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop，12227 MiB，compute capability 12.0
- 内存：约 31 GiB，Swap 约 61 GiB
- `/home` 分区约 164 GB，配置时尚余约 128 GB
- 训练时使用 `powerprofilesctl set performance`
- 网络代理为本机 `127.0.0.1:7897`，可访问 GitHub、PyPI、NVIDIA PyPI、Hugging Face

## 3. 已固定的软件版本

- NVIDIA Driver：`580.173.02` Open Kernel Module
- Miniforge：`/home/kxj/miniforge3`
- Conda 环境：`unitree_rl`
- Python：`3.11.16`
- PyTorch：`2.7.0+cu128`
- Torchvision：`0.22.0+cu128`
- Isaac Sim：`5.1.0.0`
- Isaac Lab：tag `v2.3.2`，commit `37ddf626871758333d6ed89cf64ad702aef127d0`
- Unitree RL Lab：commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`
- Unitree RL Lab Python package：`0.2.1`
- Unitree model：commit `323e350252b9c3aee9c40acbfdad84f6ce46a5ac`
- ONNX：`1.18.0`
- FastAPI：`0.115.7`
- Starlette：`0.45.3`
- psutil：`5.9.8`
- IPython：`8.37.0`

`python -m pip check` 已达到 `No broken requirements found.`

## 4. 关键路径

- 工作区：`/home/kxj/rl_ws`
- Isaac Lab：`/home/kxj/rl_ws/IsaacLab`
- Unitree RL Lab：`/home/kxj/rl_ws/unitree_rl_lab`
- Go2 USD 模型：`/home/kxj/rl_ws/unitree_model/Go2/usd/go2.usd`
- 当前正式训练日志：`/home/kxj/rl_ws/unitree_go2_baseline_seed42.log`
- 当前正式训练目录：`/home/kxj/rl_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000`
- 当前 GPU 监控：`/home/kxj/rl_ws/gpu_baseline_seed42.csv`

## 5. Conda 环境变量

激活环境：

```bash
conda activate unitree_rl
```

应有以下变量：

```text
ISAACLAB_PATH=/home/kxj/rl_ws/IsaacLab
UNITREE_RL_LAB_PATH=/home/kxj/rl_ws/unitree_rl_lab
UNITREE_MODEL_DIR=/home/kxj/rl_ws/unitree_model
LD_PRELOAD=/home/kxj/miniforge3/envs/unitree_rl/lib/libstdc++.so.6
```

检查命令：

```bash
echo "$ISAACLAB_PATH"
echo "$UNITREE_RL_LAB_PATH"
echo "$UNITREE_MODEL_DIR"
echo "$LD_PRELOAD"
python -m pip check
```

## 6. 已解决的重要问题

### 6.1 Isaac Sim 在 595.84 驱动下退出码 139

现象：RTX/Vulkan 初始化后在 Kit/RTX 插件中 segmentation fault。

原因：NVIDIA 595.x 与 Isaac Sim 5.1 的已知兼容问题。

处理：切换到 Ubuntu R580 Open 驱动 `580.173.02`。之后 Isaac Sim headless 正常到达 `app ready`。

不要重新升级到 595/610，除非同时升级整套 Isaac Sim/Isaac Lab/Unitree 版本并重新验证。

### 6.2 Conda C++ ABI 冲突

现象：`CXXABI_1.3.15 not found`，Kit 错误加载系统 `libstdc++.so.6`。

处理：只在 `unitree_rl` 环境设置：

```text
LD_PRELOAD=/home/kxj/miniforge3/envs/unitree_rl/lib/libstdc++.so.6
```

### 6.3 Isaac Lab 核心首次安装失败

`flatdict==4.0.1` 构建时缺少 `pkg_resources`。已使用：

```text
setuptools==80.9.0
wheel==0.45.1
flatdict==4.0.1（--no-build-isolation）
```

### 6.4 Isaac Sim 与 Isaac Lab Web 依赖冲突

Isaac Sim 5.1 固定 FastAPI 0.115.7 / Starlette <0.46，而 Isaac Lab 2.3.2 元数据固定 Starlette 0.49.1。

本项目不使用 livestream，已在 Isaac Lab 做最小兼容补丁，将 Starlette 固定为 0.45.3：

```text
/home/kxj/rl_ws/IsaacLab/source/isaaclab/setup.py
```

该补丁保留在 `git diff` 中。

### 6.5 Unitree 模型路径

已修改：

```text
/home/kxj/rl_ws/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
```

从硬编码占位符改为读取 `UNITREE_MODEL_DIR` / `UNITREE_ROS_DIR` 环境变量。该补丁保留在 `git diff` 中。

## 7. 已完成的验收

### Isaac Lab Cartpole

- 64 environments
- 2 PPO iterations
- 退出码 0
- 稳态约 10139 steps/s
- checkpoint、TensorBoard、参数文件均生成

### Unitree Go2 冒烟测试

- 16 env / 1 iteration：通过
- 256 env / 5 iterations：通过，稳态约 9400-9800 steps/s
- 1024 env / 10 iterations：通过，稳态约 18200 steps/s
- 2048 env / 10 iterations：通过，稳态约 21900 steps/s

2048 env 峰值：

```text
GPU utilization: 97%
GPU memory: 4906 MiB
GPU temperature: 53 C
GPU power: 66.11 W
```

正式训练选用 2048 env；不再测试 4096，因为 GPU 已接近饱和，2048 的吞吐/余量更合适。

## 8. 当前正在运行的正式基线

命令：

```bash
systemd-inhibit \
  --what=sleep \
  --why="Unitree Go2 baseline training seed 42" \
  ./unitree_rl_lab.sh --train \
  --task Unitree-Go2-Velocity \
  --num_envs 2048 \
  --seed 42 \
  --max_iterations 3000 \
  --run_name baseline_seed42_env2048_i3000 \
  --logger tensorboard \
  2>&1 | tee /home/kxj/rl_ws/unitree_go2_baseline_seed42.log
```

最近一次用户报告（iteration 430）：

```text
Computation: 21798 steps/s
Mean reward: 24.47
Mean episode length: 952.58 / ~1000
track_lin_vel_xy: 1.2502
track_ang_vel_z: 0.6430
error_vel_xy: 0.3224 m/s
error_vel_yaw: 0.2727 rad/s
time_out: 0.9399
bad_orientation: 0.0601
lin_vel_cmd_levels: 0.6000
```

状态判断：策略已经学会稳定存活，正在继续改善速度跟踪和课程难度。截至生成本文件时，磁盘中已经出现 `model_600.pt`，说明训练仍在继续。

预计 3000 iterations 总耗时约 1 小时 50 分至 2 小时。

## 9. 实时查看 TensorBoard

另开终端：

```bash
conda activate unitree_rl
tensorboard \
  --logdir /home/kxj/rl_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000 \
  --host 127.0.0.1 \
  --port 6006 \
  --reload_interval 5
```

浏览器打开：`http://127.0.0.1:6006`

重点曲线：mean reward、mean episode length、error_vel_xy、error_vel_yaw、bad_orientation、time_out、lin_vel_cmd_levels、terrain_levels、value loss、surrogate loss、mean noise std。

## 10. 训练结束后立即执行

在训练命令所在终端，紧接着执行（若 shell 中仍保留变量）：

```bash
baseline_status=${PIPESTATUS[0]}
kill "$gpu_monitor_pid" 2>/dev/null || true
wait "$gpu_monitor_pid" 2>/dev/null || true
echo "Baseline training exit code: $baseline_status"
```

提取指标：

```bash
grep -E "Learning iteration|Computation:|Mean reward|Mean episode length|error_vel_xy|error_vel_yaw|terrain_levels|lin_vel_cmd_levels|bad_orientation|base_contact|Training time" \
  /home/kxj/rl_ws/unitree_go2_baseline_seed42.log | tail -n 200
```

检查 checkpoint：

```bash
find /home/kxj/rl_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000 \
  -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 20
```

若训练终端已关闭，无法再通过 `PIPESTATUS` 获得退出码，可改为检查日志末尾、最终 checkpoint 和是否包含 traceback。

## 11. 下一阶段计划

1. 分析 3000 iterations 的收敛情况，选择最佳 checkpoint（不一定只选最后一个）。
2. 使用 play 脚本做 GUI 推理和视频录制。
3. 建立确定性评测脚本，输出速度误差、yaw 误差、跌倒率、episode length、动作平滑度、能耗等 CSV。
4. 若 3000 尚未收敛，从合适 checkpoint resume 到 5000。
5. 固定 baseline 后，实现一项复杂地形/课程学习改进。
6. 做 baseline / 改进版消融对比。
7. 时间允许时进行 ONNX 导出与 MuJoCo Sim2Sim。
8. 整理 README、环境锁定、训练曲线、视频、简历表述和面试问答。

## 12. 新会话接管规则

新助手应先完整读取本文件和以下两个仓库的当前 Git diff：

```bash
sed -n '1,320p' /home/kxj/rl_ws/unitree_rl_lab/doc/SESSION_HANDOFF_2026-08-24.md
git -C /home/kxj/rl_ws/IsaacLab diff
git -C /home/kxj/rl_ws/unitree_rl_lab diff
```

然后读取用户贴出的训练完成结果，分析收敛、稳定性和课程指标，再给下一批可直接运行的命令。继续采用“助手给一批命令 → 用户运行并反馈完整输出 → 助手分析后给下一步”的协作方式。

不要擅自升级 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch 或 Unitree commit，也不要覆盖已有 checkpoint/logs。
