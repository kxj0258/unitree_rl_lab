# 七教师冻结与统一验收（M1）

本阶段将七个独立教师接入同一注册表、具名观测适配器和评估入口。它不训练 Student，不宣称已经完成 Sim2Sim，也不把评估程序完成等同于运动性能达标。

## 归档复核补充（2026-11-16）

正式随机命令矩阵实际已在 `evaluation/teacher_bank_formal_20260910_104824/` 完成：7 地形 × scenario seeds 7/42/123 × 200 episodes，共 21 runs、4,200 episodes；训练 seed 均为 42，关闭观测噪声。原文下方“尚未执行”仅是较早快照，不能代表当前状态。正式矩阵 91 项产物哈希已复核；本轮不重跑仿真，不改旧 manifest。

| 地形 | Mean XY error (m/s) | 到达回合时限 / 600 |
|---|---:|---:|
| Flat | 0.0368 | 600 |
| Random Rough | 0.0489 | 600 |
| Pyramid Slope | 0.0356 | 600 |
| Inverted Slope | 0.0319 | 599 |
| Pyramid Stairs | 0.0372 | 599 |
| Inverted Stairs | 0.0364 | 599 |
| Boxes 6000 | 0.0654 | 505 |

Boxes 原始 episode CSV 整数 flag 加总为 base_contact 18、bad_orientation 78，标签可重叠。timeout 在这里表示到达回合时限，不是任务成功率。工程收集通过不等于运动质量全部达标。当前 `evaluate_go2.py` 与该历史正式矩阵记录源码 SHA 不同，原 provenance 保留；本轮提交不是历史实测版本。新加入的 ONNX 导出/MuJoCo 桥接代码按候选归档，不能据源码存在宣称 Sim2Sim 已验收。

## 历史 smoke 实测验收（2026-09-10）

- 七教师真实权重离线验收全部通过；6项边界回归测试通过；已有证据目录覆盖请求被正确拒绝。
- 七地形闭环smoke全部完成：scenario seed42，10环境/地形，level0–9，每环境1个episode，共70 episodes。
- 每地形比较1000个向量控制步，合计7000个向量控制步（70000个环境transition）；观测与动作最大绝对误差均为0。
- 本次70/70 episodes到达20秒时限，base-contact和bad-orientation终止均为0。
- 35个产物的SHA-256校验通过；验收时记录的三个实现文件哈希与最终运行实现一致。
- 完整正式随机命令矩阵已生成并通过离线审计，尚未执行：21 runs、4200 episodes。

| 地形 | Smoke mean XY error (m/s) | Timeout |
|---|---:|---:|
| flat | 0.0329 | 10/10 |
| random_rough | 0.0478 | 10/10 |
| pyramid_slope | 0.0364 | 10/10 |
| inverted_pyramid_slope | 0.0304 | 10/10 |
| pyramid_stairs | 0.0348 | 10/10 |
| inverted_pyramid_stairs | 0.0342 | 10/10 |
| boxes model6000 | 0.0816 | 10/10 |

这些是关闭观测噪声的单场景seed、小样本工程验收数据，不能替代独立场景性能确认，也不应与历史噪声开启协议直接比较。Boxes历史失败记录保留；本次10/10 timeout不证明全部工作域已经无失败。

机器可读摘要：[验收结果](repro/teacher_bank_m1_acceptance_2026-09-10.json)。原始结果目录：`evaluation/teacher_bank_m1_seven_smoke_2026-09-10_v1`，含顶层`result.json`、`baseline.csv`及每次运行的逐episode证据。

## 冻结选择

| 地形 | 任务版本 | checkpoint | Actor / Critic 输入 |
|---|---|---|---|
| flat | v2 | 4000 | 162 / 177 |
| random_rough | v2 | 4999 | 162 / 177 |
| pyramid_slope | v2 | 4999 | 162 / 177 |
| inverted_pyramid_slope | v2 | 4999 | 162 / 177 |
| pyramid_stairs | v3 | 4999 | 164 / 179 |
| inverted_pyramid_stairs | v3 | 4700 | 164 / 179 |
| boxes | v3 continuation | 6000 | 164 / 179 |

Rough 的历史 model4800 和倒坡的历史 model4700 在本地对应训练目录中缺失；用户明确选择各自 model4999，重新验收。Boxes model6000 由用户指定，用户已认可步态；仍完整报告运动指标与失败样本。

`config/seven_teacher_selection_v1.json` 保存选择来源；`config/seven_teacher_registry_v1.json` 保存各 checkpoint、env/agent/deploy YAML 的 SHA-256、真实网络配置、训练 seed、逐项观测与控制/命令/扫描合同。注册表里的 pending 状态描述冻结时的状态，验收结果追加保存在独立结果目录，不回写已冻结文件。

## 观测适配器

实现位于 `source/unitree_rl_lab/unitree_rl_lab/utils/teacher_bank.py`，可脱离 Isaac Sim 导入。

输入为具名、未缩放、未裁剪、未归一化的二维张量。关节顺序采用训练时 articulation 顺序；它不同于 SDK 电机顺序，映射保存在 control.joint_ids_map。适配器按每个教师的合同排序，先 clip 后 scale；网络内部使用该教师自己的已训练 normalizer。禁止重复归一化、补零缺失输入或共享不同教师的 GRU 状态。

普通教师 actor 项：base_ang_vel、projected_gravity、velocity_commands、joint_pos_rel、joint_vel_rel、last_action、height_scan。离散地形教师额外使用 gait_clock。Critic 还含 base_lin_vel、joint_effort。运行时 adapter 支持独立构造 policy/critic；本阶段为与原生评估严格比较，两组都构造。

gait_clock 是当前前进速度调制周期的 sin/cos；公式与训练代码一致，episode reset 时步数归零。它不能替换成任意自由运行的振荡器。height_scan 输入指训练函数计算后的原始高度差，不是射线距离或绝对地面高度。

Isaac bridge 使用当前 observation manager 的已解析函数获得原始观测项，再由冻结适配器预处理。每个仿真步都比较适配观测与原生观测，以及两个独立 GRU 实例的动作；实际由适配动作推进环境。任一比较失败立即停止，不静默退回原生策略。桥接层仅依赖 Isaac Lab 2.3.2 的一个内部已解析 term 列表，版本升级需要重新验证。

这构成具名观测接口与 Isaac 实现，不是 MuJoCo 观测构建器。后者是下一阶段工作。

## 运行命令

使用项目已有环境，无需安装新依赖。以下命令均从任意目录可运行；输出目录必须不存在。时间戳确保不会覆盖历史证据。

离线验收（真实 checkpoint、哈希、模型加载、输入错误拒绝、GRU reset 隔离及复现）：

```bash
/home/kxj/miniforge3/envs/unitree_rl/bin/python \
  /home/kxj/rl_ws/unitree_rl_lab/scripts/project/accept_go2_teacher_bank.py \
  --mode audit \
  --output "/home/kxj/rl_ws/unitree_rl_lab/evaluation/teacher_bank_audit_$(date +%Y%m%d_%H%M%S)"
```

七教师闭环 smoke（七任务依次运行，场景 seed42，每地形 level0–9 各一个环境、一个完整 episode，共70 episodes）：

```bash
/home/kxj/miniforge3/envs/unitree_rl/bin/python \
  /home/kxj/rl_ws/unitree_rl_lab/scripts/project/accept_go2_teacher_bank.py \
  --mode smoke --execute \
  --output "/home/kxj/rl_ws/unitree_rl_lab/evaluation/teacher_bank_smoke_$(date +%Y%m%d_%H%M%S)"
```

正式随机命令基线（3个场景 seed × 7地形 × 100环境 × 2episodes = 4200 episodes）：

```bash
/home/kxj/miniforge3/envs/unitree_rl/bin/python \
  /home/kxj/rl_ws/unitree_rl_lab/scripts/project/accept_go2_teacher_bank.py \
  --mode formal --execute \
  --output "/home/kxj/rl_ws/unitree_rl_lab/evaluation/teacher_bank_formal_$(date +%Y%m%d_%H%M%S)"
```

固定命令矩阵：在 formal 命令追加 `--profiles forward_mid turn_left turn_right`。它使用 vx=0.35 m/s、relative heading=0/±45°、初始 yaw=0，其他控制限幅遵守原教师合同。该矩阵为12600 episodes，不应误标为全部速度范围测试。可用 `--seeds`、`--num-envs`、`--episodes-per-env`、`--terrains` 缩小诊断范围。

省略 `--execute` 会完成离线检查并生成/打印仿真计划，不启动仿真，result.json 状态为 planned。默认模型资产取项目相邻的 unitree_model；可通过 `--model-dir` 指定，运行器会设置 UNITREE_MODEL_DIR。plan.json 中的子命令若单独复制运行，也须设置该环境变量。

## 验收产物与判定

- audit.json：七教师文件/输入/推理检查。
- provenance.json：注册表哈希、工作树状态、代码哈希、资产路径。
- registry.json：本次注册表副本。
- plan.json：精确子命令与期望 episode 数。
- 每个 run 的 metadata.json、episodes.csv、summary.json 及 console 日志。
- 顶层 episodes.csv：带 teacher/profile/training seed/scenario seed 的合并记录。
- baseline.csv：每地形/命令及 level 的结果，可由 episode CSV 重算；均值为 episode 等权。
- result.json：M1 工程验收结论；不设运动通过率阈值，不把 timeout 写作 success。
- artifacts.sha256.json：产物校验索引，不包含自身。
- failure.json：运行失败时保留堆栈；失败目录不能复用。

Smoke 的 pass 要求七教师离线检查通过、每步原生对齐通过、完整episode数与有限数值检查通过。姿态/接触终止保留在结果中，不能因为某episode失败就删除它。只有正式随机命令与固定命令矩阵完成后，才应依据工作域质量要求决定是否将替代候选作为最终教师。

已有 training seed 都是42；评估的7/42/123不构成多训练seed。七个教师分别在各自任务评估，不证明跨地形切换或统一Student。

## 重新冻结

先创建新版本 selection 文件，明确指定 checkpoint，再运行：

```bash
/home/kxj/miniforge3/envs/unitree_rl/bin/python \
  /home/kxj/rl_ws/unitree_rl_lab/scripts/project/freeze_go2_teacher_registry.py \
  --selection /absolute/path/in/project/new_selection.json \
  --output /absolute/path/in/project/new_registry.json
```

工具不会自动选择最优checkpoint，拒绝覆盖注册表。验收用 `--registry` 指向新版本。历史proxy注册表和历史checkpoint保持原样。
