# Go2 阶段验证 Demo 结果

## 定位

本阶段为秋招项目展示补充两套可审计视频：MultiTerrain 七地形原型，以及 rough 三个独立 training seeds 的固定 checkpoint 对比。视频用于直观验证，不替代既有 `multiterrain_stratified` 或 `rough_stratified v1` 正式指标。

所有片段使用 deterministic policy、evaluation scenario seed 42、terrain level 5、固定前进命令 `vx=0.35 m/s`、`vy=0`、`yaw=0`。每段为 300 simulation steps（约 6 秒 simulated time），1280×720、50 FPS、299 frames。运行期 curriculum 关闭，并检查 terrain column/level 不变化。

## MultiTerrain Demo

- Task：`Unitree-Go2-Velocity-MultiTerrain`；
- training seed：42；
- checkpoint：`model_2995.pt`；
- checkpoint SHA-256：`67da571685f42e56df412730cd6c13c335a63aa85a07beaf7f97b35cfb3dd2da`；
- 七类 terrain 共 7 个原始片段；
- 七段均无 timeout、base-contact 或 bad-orientation termination；
- 成片时长 43.86 秒，SHA-256 `d45508df3e9a92c254dae07c306ca86613fd329cd2fa247189dbc7bb58f324a7`。

| Terrain | Clip mean XY error | Clip mean yaw error | Termination |
|---|---:|---:|---:|
| flat | 0.1584 m/s | 0.0642 rad/s | 0 |
| random rough | 0.3135 m/s | 0.0702 rad/s | 0 |
| pyramid slope | 0.1588 m/s | 0.0650 rad/s | 0 |
| inverted pyramid slope | 0.1685 m/s | 0.0643 rad/s | 0 |
| pyramid stairs | 0.1948 m/s | 0.0834 rad/s | 0 |
| inverted pyramid stairs | 0.2165 m/s | 0.0613 rad/s | 0 |
| boxes | 0.2351 m/s | 0.0899 rad/s | 0 |

这些是单条固定命令短片的时间平均值，不能与 140-episode 分层评估直接比较，也不能据此宣称 MultiTerrain 多-seed 泛化。

## Rough 多训练 Seed Demo

- Task：`Unitree-Go2-Velocity-Rough`；
- training seeds：42、7、123；
- 每个 training seed 均使用固定 `model_2999.pt`，与跨 seed 固定 iteration 主结果一致；
- terrain：`random_rough level 5`；
- evaluation scenario seed：42；
- 三段均无 timeout、base-contact 或 bad-orientation termination；
- 成片时长 19.94 秒，SHA-256 `1ae78d815839376f43fb284a44a854896418c48bd78d0b34c9a9b492678bc50e`。

| Training seed | Checkpoint SHA-256 | Clip mean XY error | Clip mean yaw error |
|---:|---|---:|---:|
| 42 | `c3d8f8fd5b24aa78b870f1c90c74ca50271d68b13c41b29123261afc0ed95a38` | 0.1273 m/s | 0.0681 rad/s |
| 7 | `353ea7a9d1cfa3e35fdc8678901ede952c54e7f4fb54782ad86fd2f02352f4af` | 0.1129 m/s | 0.0666 rad/s |
| 123 | `d83a237bbf9918b3f81d269b1ce4684659756ee30c902b384f92d9dfced0945c` | 0.1107 m/s | 0.0764 rad/s |

training seed 42 的历史 per-seed official 仍是 `model_2900.pt`；本视频使用 `model_2999.pt` 是因为它展示跨 training-seed 的固定 iteration 主比较。evaluation scenario seed 42 不作为独立训练样本。

## 审计

- 10/10 原始 MP4 可解码；
- 每段 1280×720、50 FPS、299 frames、约 5.98 秒；
- 10/10 metadata 的 task、training seed、evaluation scenario seed、checkpoint、terrain type/level 和 command 已核对；
- checkpoint/video SHA-256 全部匹配；
- console 未发现 Traceback、fatal、CUDA error、OOM 或非有限值；
- 原始视频、console、metadata 和成片保留在 Git 忽略的 `evaluation/` 目录；Git 只归档代码、inventory 和 SHA-256。

## 后续

当前成片适合作为内部阶段验证和作品集素材。发布前仍建议人工剪辑封面、项目架构图和正式实验指标页；下一项高价值工作是 MuJoCo Sim2Sim，完成后制作 Isaac Sim/MuJoCo 并排 Demo。
