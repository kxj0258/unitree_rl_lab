# Unitree Go2 v5 Stairs Ascending Remediation 结果（2026-08-28）

## 1. 结论

本阶段解决了 v5 stairs 正式模型在 `inverted_pyramid_stairs` 中“路径很长但在坑底绕圈、
未真正上楼”的定向失败。补救训练产生两个重要候选：

- `model_900.pt`：当前最佳 ascending specialist 与综合折中候选；
- `model_1199.pt`：固定航向更强，但随机命令下过拟合和 hard-stairs 安全退化，不采用。

原 stairs `model_4999.pt` 仍是综合随机命令正式模型。当前采用双模型结论，不把 remediation
checkpoint 错写为已完成融合的单一通用模型。

## 2. 固定身份

所有训练均只有一个 training seed：`42`。评估使用的 `7/42/123` 是 evaluation scenario
seeds，不是额外训练 seed。

| 角色 | checkpoint | SHA-256 |
|---|---|---|
| v5 foundation 正式模型 | `model_3200.pt` | `c23c592eb9401c0173e470672cfb67aa1bf12a2074cb8fbe93a438f2c730ed08` |
| 综合 stairs 正式模型 | `model_4999.pt` | `47b42309c03450c88f27ceaf66ed393abfcc5a488e7a508a369055ed4cfcee50` |
| ascending specialist 候选 | `model_900.pt` | `57c66a27fb3be566ad28edec6e567bfc1f8eddea8db3115e0ed8009c9f38d1a3` |
| 过拟合、拒绝作为综合模型 | `model_1199.pt` | `df7d0e769056f0dec1bb9f6fb0d5c127e2ebb66ffebda603d0ebfe8f3e003e7e` |

`model_900.pt` 路径：

`logs/rsl_rl/unitree_go2_velocity_v5_stairs_ascending_remediation/2026-08-27_23-01-17_v5_stairs_ascending_remediation_formal_v1_from4999_trainseed42_env1024_i1200/model_900.pt`

## 3. 问题诊断

综合 stairs `model_4999.pt` 在固定 `vx=0.35 m/s, vy=0, yaw_rate=0` 的 inverted-stairs
诊断中会持续转圈。加入固定 world heading 与固定 initial yaw 后，heading controller 已产生
明显纠偏命令，但策略仍不能执行：

- inverted-stairs 路径长度约 `7.251 m`；
- 净位移仅约 `1.251 m`；
- yaw error 约 `0.852 rad/s`；
- 平均绝对 yaw correction command 约 `0.695 rad/s`；
- timeout 100%，base contact 0，bad orientation 0。

这排除了“只缺少 heading command”的解释。根因是原 curriculum 只检查 linear tracking 与
body-frame command progress；策略可通过沿自身朝向移动和绕圈获得 progress，而不保证世界方向
净位移或真实爬升。

## 4. Remediation 设计

新增 task：`Unitree-Go2-Velocity-V5-Stairs-Ascending-Remediation`。

- 从正式 stairs `model_4999.pt` 迁移 policy、GRU 与 observation normalization；
- 因 objective/terrain/command 分布改变，重置 optimizer 与 iteration；
- PPO learning rate `5e-5`；
- 1024 env，32 steps/env/iteration，1200 iterations；
- 总 transitions：`39,321,600`；
- terrain：70% inverted stairs、20% pyramid stairs、5% flat、5% random rough；
- command：`vx=0.25--0.45`、`vy=0`、fixed heading 0、initial yaw 0；
- 加强 `track_ang_vel_z`；
- curriculum 晋级同时要求 survival、linear tracking、command progress、yaw tracking。

正式 run：

`logs/rsl_rl/unitree_go2_velocity_v5_stairs_ascending_remediation/2026-08-27_23-01-17_v5_stairs_ascending_remediation_formal_v1_from4999_trainseed42_env1024_i1200`

训练 exit code 0，1200 TensorBoard iterations，无 NaN/Inf；checkpoint 从 `model_0.pt` 连续到
`model_1199.pt`，每个检查点均保存 optimizer state。

## 5. Heading-locked checkpoint 选择

固定条件：training seed 42、evaluation scenario seed 42、`vx=0.35`、`vy=0`、world heading 0、
initial yaw 0、4 terrains × 8 levels × 2 episodes。

| checkpoint | inverted 净位移 | inverted yaw error | 结论 |
|---|---:|---:|---|
| 原 model4999 | 1.251 m | 0.852 | 坑底绕圈 |
| remediation model600 | 6.985 m | 0.107 | levels 0--6 强，level 7 仍弱 |
| remediation model900 | 7.274 m | 0.0527 | level 7 成功，安全 |
| remediation model1199 | 7.351 m | 0.0518 | 固定航向最好 |

model900 inverted level 7：净位移 `7.167 m`、yaw error `0.0557 rad/s`、base contact 0、
bad orientation 0。

## 6. 原始随机命令回归：三 evaluation scenario seeds

对 model900、model1199 使用原 `v5_stairs_stratified` 随机 command 协议与 evaluation scenario
seeds `7/42/123`。每个模型共 192 episodes。

三-seed 总体平均：

| 指标 | 正式 model4999 | remediation model900 | remediation model1199 |
|---|---:|---:|---:|
| XY error m/s | 0.0498 | 0.0612 | 0.0648 |
| yaw error rad/s | 0.1558 | 0.0856 | 0.0788 |
| command-aligned distance m | 8.260 | 8.079 | 8.097 |
| integrated displacement m | 3.668 | 4.134 | 3.737 |
| stall fraction | 0.608% | 1.106% | 1.057% |
| power W | 44.61 | 48.79 | 50.58 |
| timeout | 98.96% | 98.44% | 96.35% |
| bad orientation | 1.04% | 1.56% | 3.65% |

model900 相比 model4999 显著改善 yaw 与 inverted displacement，但 XY tracking、stall 与功耗退化。
model1199 在 inverted levels 6--7 出现约 16.67% bad orientation，拒绝作为综合模型。

## 7. Level-7 上楼视频证据

视频目录：

`evaluation/video_v5_stairs_headinglocked_ascendingremediation_model900_trainseed42_evalseed42_inverted_level7_vx035_heading0_inityaw0_20s_2026-08-28_v1`

身份：model900、training seed 42、evaluation scenario seed 42、inverted-pyramid stairs level 7、
`vx=0.35`、heading 0、initial yaw 0。

量化与视觉检查：

- 起始 base z `-0.4859 m`；最终 base z `+0.3217 m`；高度增益约 `0.808 m`；
- final forward displacement `7.2157 m`；path length `7.2881 m`；
- post-warmup stall `0`；
- post-warmup yaw error `0.0500 rad/s`；
- timeout 1、base contact 0、bad orientation 0；
- 视频 1280×720、50 FPS、999 frames、19.98 s；CSV 1000 rows；
- 抽帧确认机器人从凹坑底部连续爬出台阶并到达上层平面。

视频 SHA-256：

`ff4192b91068c8a1725017078718647ae412ba64dcbe3db4577a47dcdbceb94e`

tracking CSV SHA-256：

`7fe1b2c0228b6375e495678654c5643e849bb48e7cbf8bd91b2e92af992087c0`

## 8. 当前决策与禁止事项

- 不继续当前窄 ascending-remediation 分布到 3000；目标已达成，继续会增加过拟合风险；
- 不把 timeout 单独描述为成功；
- 不把 model900 描述成已经统一解决七类混合地形；
- 不删除、覆盖或移动 model4999/model900/model1199、日志、TB、评估和视频；
- 当前可保留双模型：model4999 综合 stairs，model900 heading-locked ascending specialist；
- 若必须单一 policy，下一阶段应是 mixed-command consolidation 或 mixture/distillation，而不是
  继续窄分布盲目追加 iterations。

## 9. 未完成工作

1. model900 heading-locked evaluation scenario seeds 7/123；
2. mixed-command consolidation 的正式设计、smoke、训练和双协议回归；
3. boxes specialist；
4. 七类地形的统一最终模型或 mixture/router；
5. 最终 before/after 作品集成片；
6. policy export、domain randomization 与 sim-to-real 安全部署；
7. 将本结果纳入后续不可覆盖的新 handoff 与 manifest。
