# Go2 项目命令与脚本保存审计

## 结论

关键底层训练、评估、视频录制、动态合成、TensorBoard 分析和归档能力均已保存到 Git。早期通过终端直接拼接的 MultiTerrain Stage-2 环境变量命令此前没有单独的 `.sh` 文件，但其参数已保存在接管/结果文档；现已进一步固化为默认 dry-run 的统一入口：

```text
scripts/project/reproduce_go2_multiterrain_stage2.sh
```

因此后续复现不依赖聊天记录或 Shell history。大型 checkpoint、TensorBoard、console/GPU CSV、evaluation CSV/JSON 和视频仍位于 Git 忽略的本地目录，通过 `doc/repro/*.sha256` 固化完整性。

## 已脚本化映射

| 工作 | 保存的入口 | 状态 |
|---|---|---|
| Flat baseline 训练 | `scripts/project/train_go2_baseline.sh` | 已保存 |
| Rough 训练 | `scripts/project/train_go2_rough.sh` | 已保存，含 console/GPU CSV |
| MultiTerrain survival/tracking-aware 训练 | `scripts/project/train_go2_multiterrain.sh` | 已保存，含 resume 检查、console/GPU CSV |
| Flat/rough/MultiTerrain 单 checkpoint 评估 | `scripts/project/evaluate_go2.py` | 已保存 |
| Rough 12-run 矩阵 | `scripts/project/evaluate_go2_rough_matrix.sh` | 已保存，支持 dry-run |
| Rough 多 training-seed 9-run 矩阵 | `scripts/project/evaluate_go2_rough_training_seed_matrix.sh` | 已保存，明确区分两类 seed |
| MultiTerrain Stage-2 精确续训/评估编排 | `scripts/project/reproduce_go2_multiterrain_stage2.sh` | 本次补齐，默认只打印 |
| Flat follow/grid 视频 | `scripts/project/record_go2_video.sh` | 已保存 |
| 七地形固定协议视频 | `scripts/project/record_go2_multiterrain_demo.py`、`.sh` | 已保存 |
| Rough 三 training-seed 视频 | `scripts/project/record_go2_rough_multitraining_demo.sh` | 已保存 |
| 动态遥测成片 | `scripts/project/compose_go2_demo_videos.py` | 已保存 |
| TensorBoard 分析 | `scripts/project/analyze_tensorboard.py` | 已保存 |
| Rough 审计/归档 | `audit_go2_rough*.py`、`archive_go2_rough*.py` | 已保存 |
| Stage-2 精选产物校验 | `reproduce_go2_multiterrain_stage2.sh verify-archive` | 已保存 |

## 本轮曾经仅在终端拼接、现已固化的命令

1. 从原 `model_2995.pt` 续训 survival baseline 500 iterations；
2. 从相同 checkpoint 续训 tracking-aware 500 iterations；
3. baseline `model_3300` × evaluation scenario seeds 7/42/123；
4. tracking-aware `model_3300/3400/3494` × scenario seed 42；
5. 入选 baseline 的七地形 level-5、20 秒 asset-follow 视频；
6. Rough training seeds 42/7/123 的同协议视频；
7. Stage-2 checkpoint、summary JSON 和最终动态视频 SHA-256 复核。

预览这些精确命令：

```bash
scripts/project/reproduce_go2_multiterrain_stage2.sh --help
OUTPUT_ROOT=evaluation/new_eval scripts/project/reproduce_go2_multiterrain_stage2.sh baseline-eval
```

只有显式设置 `EXECUTE=1` 才实际运行。训练命令还要求 `REPRO_TAG`，评估/视频命令要求不存在的 `OUTPUT_ROOT`，降低误覆盖风险。

## 有意不脚本化的内容

- 日常 `git status`、`find`、`sha256sum`、CSV/JSON 临时只读检查：这些是诊断命令，不属于实验定义；关键检查已进入 audit/archive 或 Stage-2 verify 入口。
- NVIDIA/Isaac/PyTorch 安装升级：明确禁止，未写自动安装脚本。
- MuJoCo/unitree_sdk2 安装和克隆：尚未授权执行，当前只保留上游说明，不把未验证步骤包装成“可复现成功脚本”。
- GUI 人工观看与作品集观感确认：必须由人完成，不能由 exit code 替代。

## 已知审计限制

survival baseline continuation 实际完成至 iteration 3494，checkpoint/TensorBoard/GPU CSV 和固定评估有效；但由于运行期间修改了正在执行的 Shell 文件，wrapper 最终返回 127，预期 console 文件不完整。该事实已写入 `RESULTS_MULTITERRAIN_CONTINUATION_2026-08-26.md`。tracking-aware 正式 run 则 exit 0 且日志完整。

## 文档入口

- 项目总览：`README.md`；
- 当前状态：`doc/PROJECT_STATUS.md`；
- 复现命令：`doc/REPRODUCTION.md`；
- 最新 MultiTerrain 结果：`doc/RESULTS_MULTITERRAIN_CONTINUATION_2026-08-26.md`；
- 简历与面试表述：`doc/RESUME_PROJECT.md`；
- 会话接管：`doc/SESSION_HANDOFF_2026-08-25_MULTITERRAIN_PROTOTYPE.md`；
- 精选产物清单：`doc/repro/go2_multiterrain_stage2_final_artifacts_2026-08-26.sha256`。
