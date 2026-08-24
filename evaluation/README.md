# Evaluation Artifacts

本目录保存 Go2 baseline 的本地评估与展示产物。大型二进制、逐 episode CSV 和运行日志默认不提交 Git；可追踪的结论与校验清单位于 `doc/` 和 `doc/repro/`。

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

## 校验与重新生成

```bash
sha256sum --check doc/repro/final_artifacts_2026-08-25.sha256
```

重新生成方法见 `doc/REPRODUCTION.md`、`scripts/project/evaluate_go2.py` 和 `scripts/project/record_go2_video.sh`。
