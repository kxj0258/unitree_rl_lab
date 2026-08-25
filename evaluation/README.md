# Evaluation Artifacts

本目录保存 Go2 flat baseline 与 rough 扩展的本地评估和展示产物。大型二进制、逐 episode CSV 和运行日志默认不提交 Git；可追踪的结论与校验清单位于 `doc/` 和 `doc/repro/`。

## 正式 checkpoint

```text
../logs/rsl_rl/unitree_go2_velocity/2026-08-24_23-25-37_baseline_seed42_env2048_i3000/model_2999.pt
SHA-256: a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928
```

## 统一评估

三个候选 checkpoint 分别在评估场景 seed 7、42、123 下运行；每组 64 个并行环境、每环境 2 个 episode：

```text
eval_model{2400,2900,2999}_env64_seed{7,42,123}_ep2/
```

每个目录包含：

- `metadata.json`：任务、checkpoint hash、seed 和协议；
- `episodes.csv`：逐 episode 指标；
- `summary.json`：单次运行汇总。

正式 checkpoint `model_2999.pt` 在 384 个 episode 上达到 0.0756 ± 0.0021 m/s XY 平均速度误差、99.48% timeout rate、0% base-contact termination。完整表格见 `doc/RESULTS_BASELINE.md`。

## 正式展示视频

### 单环境 follow

```text
video_model2999_seed42_follow_1000/videos/play/rl-video-step-0.mp4
```

用于展示步态、身体姿态、转向与足端运动细节。

### 9 环境 grid

```text
video_model2999_seed42_grid9_1000/videos/play/rl-video-step-0.mp4
```

用于展示向量化并行仿真。不同机器人属于相互独立的克隆环境，跨环境碰撞被隔离；视觉轨迹交叠不是多机器人避碰实验。

## 正式导出

以 follow 运行产生的导出作为归档版本：

```text
video_model2999_seed42_follow_1000/exported/policy.pt
video_model2999_seed42_follow_1000/exported/policy.onnx
```

Follow 与 grid 导出的 ONNX 文件 byte-for-byte 相同。TorchScript archive 的权重张量一致，但自动生成的类名修饰和 serialization ID 会导致两次导出的文件 hash 不同，因此只指定 follow 版本作为正式归档。

## Rough 分层评估

Rough 正式评估目录：

```text
rough_stratified_v1_2026-08-25/
```

矩阵包含 flat `model_2999` control 与 rough `model_2400/2900/2999`，评估场景 seeds 7/42/123。每个 run 为 64 environments × 2 episodes，共 12 runs、1536 episodes。

每个 run 目录包含 `metadata.json`、`episodes.csv` 和 `summary.json`；对应的完整控制台日志保存在评估根目录。协议精确分配 50% flat / 50% random_rough，两类地形均覆盖 level 0–9，并输出 overall、terrain type、level 与 type-level 聚合。

正式 rough checkpoint：

```text
../logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000/model_2900.pt
SHA-256: 9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889
```

完整结果见 `doc/RESULTS_ROUGH.md`，聚合数据见 `doc/repro/rough_eval_multiseed_2026-08-25.csv`。

## Rough 多训练 Seed 评估

新增两个独立 training-seed 评估根目录：

```text
rough_stratified_v1_trainseed7_2026-08-25/
rough_stratified_v1_trainseed123_2026-08-25/
```

它们与原 training seed 42 的 rough runs 合计构成：三个 training seeds × 三个 checkpoint iterations × 三个 evaluation scenario seeds，共 27 rough runs、3456 episodes。目录名称中的 `trainseed` 与 `evalseed` 分别表示训练 seed 和评估场景 seed，不得混用。

Per-seed selected official 为 seed 42 `model_2900`、seed 7 `model_2999`、seed 123 `model_2999`；固定 iteration 的跨 training-seed 主结果推荐 `model_2999`。完整结果见 `doc/RESULTS_ROUGH_MULTITRAINING.md`。

## 校验与重新生成

```bash
sha256sum --check doc/repro/final_artifacts_2026-08-25.sha256
sha256sum --check doc/repro/rough_final_artifacts_2026-08-25.sha256
sha256sum --check doc/repro/rough_multitraining_evaluation_sha256_2026-08-25.txt
sha256sum --check doc/repro/rough_multitraining_final_artifacts_2026-08-25.sha256
```

重新生成方法见 `doc/REPRODUCTION.md` 以及 `scripts/project/` 中的 baseline/rough 训练、评估、审计、归档和视频脚本。
