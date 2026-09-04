# V5 Boxes Proxy-Teacher 结果（2026-08-29）

## 1. 结论

当前 boxes proxy-teacher 首选为 safety-remediation namespace 的 `model_900.pt`：

`logs/rsl_rl/unitree_go2_velocity_v5_boxes_safety_remediation/2026-08-28_20-41-58_v5_boxes_safety_remediation_v2_formal_from_mixed1999_trainseed42_env1024_i1000/model_900.pt`

SHA-256：`79f98b9780be4936f5fbd199147bf5865d5567a9184c4a75f82bb245098ea3cb`。

该模型是受限 operating-envelope 的 proxy teacher，不是最终七地形主路线中的 from-scratch
正式 teacher，也不能称为完整命令域安全策略。性能备选/消融基线为 mixed-command namespace 的
`model_1999.pt`，SHA-256 为
`cce1f636b085409124b3526f63a785131a40e12cb9d73a63a655f4e265f8016b`。

## 2. 来源链

`foundation model_3200 -> boxes specialist model_3999 -> mixed-command model_1999 -> safety-remediation model_900`

所有阶段 training seed 均为 42。每次目标或数据分布改变时均在独立 experiment namespace 中迁移
policy、GRU 与 normalizer，并重置 optimizer 和 iteration。`model_900` 不是从随机初始化训练的。

## 3. Safety-remediation 正式训练

- task：`Unitree-Go2-Velocity-V5-Boxes-Safety-Remediation`；
- 1024 envs，1000 iterations，32,768,000 transitions；
- 学习率 `2.5e-5`；
- 90% boxes、5% flat、5% random rough；
- 固定均匀覆盖 terrain levels 0--7，不启用 runtime terrain curriculum；
- 命令范围：`vx=[-0.40,0.40]`、`vy=[-0.25,0.20]`、`yaw=[-0.40,0.35]`；
- actor/critic/action schema：162/177/12，GRU hidden size 256；
- exit code 0；11 个 checkpoint 连续；optimizer state 完整；TensorBoard/checkpoint 无非有限值；
- GPU 平均利用率 76.89%，最高 63 C，峰值显存 5238 MiB。

## 4. 三种子六方向固定分层评估

evaluation scenario seeds 为 7、42、123，与 training seed 含义严格分开。每个模型覆盖六方向、
boxes levels 0--7、288 episodes。

| 指标 | mixed `model_1999` | safety `model_900` |
|---|---:|---:|
| mean XY error | 0.0916 m/s | 0.0882 m/s |
| mean yaw error | 0.1168 rad/s | 0.1183 rad/s |
| command-aligned progress | 0.2679 m/s | 0.2661 m/s |
| stall fraction | 4.53% | 3.98% |
| timeout rate | 95.8% | 96.2% |
| base-contact/bad-orientation failures | 12/288 | 11/288 |
| mechanical power | 67.3 W | 67.8 W |

Hard levels 6--7：

| 指标 | mixed `model_1999` | safety `model_900` |
|---|---:|---:|
| mean XY error | 0.1312 m/s | 0.1308 m/s |
| command-aligned progress | 0.2375 m/s | 0.2409 m/s |
| stall fraction | 11.31% | 9.62% |
| timeout rate | 87.5% | 90.3% |
| safety failures | 9/72 | 7/72 |

`model_900` 的 hard-level progress、stall、timeout 和安全失败均有跨种子净改善，因此选为当前
proxy checkpoint。总体仍有 11/288（3.82%）安全失败；timeout 不能单独解释为有效运动或成功。

## 5. Level-7 视频证据

录制根目录：

`evaluation/video_v5_boxes_safety_model900_trainseed42_evalseed42_level7_sixaxis_evidence_v1`

三段视频均为 H.264、1280x720、50 fps、999 帧、19.98 秒；checkpoint、CSV、视频和 metadata
SHA-256 均已校验。

| Clip | 命令 | 结果 |
|---|---|---|
| forward | `vx=0.35` | timeout，无安全终止；累计路径 6.561 m，前向位移 5.911 m，post-warmup stall 0% |
| lateral-right | `vy=-0.25` | timeout，无安全终止；累计路径 4.653 m，post-warmup stall 0% |
| reverse-yaw | `vx=-0.25, yaw=-0.30` | 无碰撞/翻倒但任务未有效完成；实际 x 速度约 -0.112 m/s，stall 74.78% |

reverse-yaw 片段明确保留为失败边界证据，不能用其 timeout 宣称成功。

## 6. 项目定位与下一步

该 checkpoint 可用于 proxy-teacher bank、multi-teacher dispatcher/数据流 smoke 和 transfer-vs-from-scratch
消融。不得把它登记为最终高质量 boxes teacher，也不应无筛选地蒸馏失败轨迹。最终主路线仍需在独立
namespace 从随机初始化训练 boxes teacher，并用相同三种子、六方向、全 levels 协议与本模型比较。

证据清单：`doc/repro/v5_boxes_proxy_teacher_model900_evidence_2026-08-29.sha256`，共 142 项；manifest
自身 SHA-256 为 `6946ef375e285278fa6e89dc89b46743eea41f706990856eea66f56837ec2eeb`。
