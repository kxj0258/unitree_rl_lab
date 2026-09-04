# Unitree Go2 项目 V8 主接管：Proxy Student 消融与待执行完整验证（含全部历史索引）

日期：2026-08-29（Asia/Shanghai）  
项目根目录：`/home/kxj/rl_ws/unitree_rl_lab`  
IsaacLab：`/home/kxj/rl_ws/IsaacLab`

## 1. 接管规则与永久约束

新助手必须从本文件第一行读到末尾，不能只读摘要。随后严格按第 12 节读取历史文件和证据。

- 禁止升级或重装 NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch。
- 禁止覆盖、移动或删除 checkpoint、训练日志、GPU CSV、TensorBoard event、评估结果或视频。
- 训练、评估、录制、视频合成只能由助手给出完整命令，由用户在终端执行并回传输出；助手不得自行启动。
- 安装、克隆、联网下载或扩大范围前必须得到用户明确授权。
- 不清理 dirty worktree；不修改或删除 `env_backup_unitree_rl.yml`。
- 不重复已完成的训练、评估、录制或同名视频合成。
- 永远严格区分 training seed 与 evaluation scenario seed；timeout 不等于运动有效或任务成功。
- 所有新 artifact 使用独立 namespace，并在运行前拒绝覆盖已有路径。

## 2. 仓库与运行状态（2026-08-29 接管时）

- Unitree RL Lab HEAD：`247f48bdc6ab6648ac08981fd2d076fab81e90a3`。
- IsaacLab HEAD：`39b494e6d8ae3f7adbb30ae1feaf45d21bdc7d19`。
- Unitree worktree 有大量有意保留的 tracked/untracked 修改；禁止清理。
- IsaacLab worktree只读检查未显示修改。
- 最后检查时没有残留训练、评估、录制或 Isaac Sim/Kit 进程（`pgrep` 仅匹配检查命令自身）。

## 3. 不可丢失的历史结论

1. HeightScan `model_10000` 与历史 34 秒 Demo 已完成，不得重复。
2. v5 foundation `model_3200` 已完成，并用 evaluation scenario seeds 7/42/123 验证；training seed 与 evaluation seed 必须分开。
3. stairs training seed 42：`model_2200` 是带 optimizer/iteration 的精确恢复起点；追加训练后最终为 `model_4999`，不是从零编号的新独立模型。
4. ascending remediation：`model_900` 是 ascending specialist；`model_1199` 因过拟合/安全退化被拒绝，不能替代 `model_4999`。
5. 当前 47 秒 Demo v1 已完成；它是 multi-checkpoint showcase，不是统一七地形策略。overlay 有小瑕疵但审计版不得覆盖。
6. 项目最终科学目标仍是：七个 terrain-specific teachers 分别从零训练并严格验证，然后统一 recurrent perceptive student 也从零训练。
7. 本轮建立的四 checkpoint/七地形 dispatcher 与 student 都是 **proxy-teacher ablation**，不能声称已有七个正式高质量 teachers，也不能声称最终统一 student 已完成。

## 4. Boxes 分支完整结论

路线：foundation3200 → boxes specialist model3999 → mixed-command remediation model1999 → safety-remediation model900。

当前 boxes proxy checkpoint：

`/home/kxj/rl_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity_v5_boxes_safety_remediation/2026-08-28_20-41-58_v5_boxes_safety_remediation_v2_formal_from_mixed1999_trainseed42_env1024_i1000/model_900.pt`

SHA-256：`79f98b9780be4936f5fbd199147bf5865d5567a9184c4a75f82bb245098ea3cb`。

- 跨 seeds 六方向评估：model900 为 11/288 safety failures，hard levels 为 7/72；来源 model1999 为 12/288 和 9/72。
- 因此 model900 仅被选为 bounded boxes proxy teacher。
- level7 视频中 forward、lateral-right 有效；reverse-yaw timeout 且 stall 74.78%，属于失败，不能以 timeout 当成功。
- 结果文档：`doc/RESULTS_V5_BOXES_PROXY_TEACHER_2026-08-29.md`。
- 证据 manifest：`doc/repro/v5_boxes_proxy_teacher_model900_evidence_2026-08-29.sha256`。

## 5. Proxy teacher registry

Registry：`config/proxy_teacher_registry_v1.json`，SHA-256：
`9f840578c28024c5345c388ce0df94f71317cc127aac6486a90257b3140130f8`。

映射：

- flat、random_rough、pyramid_slope、inverted_pyramid_slope → foundation3200；
- pyramid_stairs → stairs4999；
- inverted_pyramid_stairs → ascending900；
- boxes → boxes safety model900。

共七个 terrain dispatch entries，但只有四个 unique policies。共享 schema：actor 162、critic 177、action 12、action scale 0.25、height scan 117、GRU 256；terrain ID 不进入 student observation。

Offline audit：`evaluation/proxy_teacher_registry_v1_offline_audit_2026-08-29.json`，status pass；验证了 checkpoint SHA、iteration、33个 state entries/shape、有限推理和共享 foundation hidden reset slot isolation。

## 6. Online dispatcher smoke

脚本：

- `scripts/project/smoke_go2_proxy_teacher_dispatcher_online.py`
- `scripts/project/smoke_go2_proxy_teacher_dispatcher_online.sh`

第一次 v1 因通用 `isaaclab_tasks.utils.parse_env_cfg`/诊断不足提前退出，没有 JSON，不算有效运行。修正为项目 parser 并增加 traceback/console/exit validation 后，v2 通过：

- task：`Unitree-Go2-Velocity-MultiTerrain-HeightScan`；
- evaluation scenario seed：20260829；training seed：无；
- 70 env，七地形各10个并覆盖 level 0–9；64 steps；
- 4个 unique teachers；每地形640 labels；nonfinite=0；
- student actions 驱动该 smoke 的环境；当时 student 随机初始化，MSE 不是 teacher 性能排名；
- 64 steps 仅约1.28秒，zero termination 不能证明安全或任务成功。

证据：

- `evaluation/proxy_teacher_dispatcher_online_smoke_seed20260829_steps64_v2.json`，SHA `59edd59524d3fd1824bbd8b772d460b6e53f42668cab091e6855cfd304ad8ae7`；
- 对应 console SHA `8386c0850beeb30b772bbb8cbd531cb21ad46a8a7f8338e1150a21e3cd8d0847`。

## 7. Recurrent student imitation smoke

实现：

- `scripts/project/smoke_go2_proxy_teacher_student_imitation.py`
- `scripts/project/smoke_go2_proxy_teacher_student_imitation.sh`

语义：student 从零初始化；4个 proxy checkpoints 冻结；teacher actions 驱动环境以保持行为克隆数据分布；student 不接收 terrain ID；每步一次 optimizer update；GRU hidden state 按 done reset并在每步截断反传；固定校准批次用于前后 MSE 对照。

第一次 v1 因 target tensor 错建在 CPU，而 teacher 输出位于 cuda:0，明确失败；没有有效输出目录/checkpoint。随后修复为使用 `obs["policy"].device`，并增强 shell：只有 JSON/checkpoint 非空且 `status=pass` 才报告 exit 0。

有效 v2：128 updates，scenario/init seed 20260829（旧 smoke 命名）；结果：

- initial fixed MSE 0.520816；final 0.386458；ratio 0.742024；
- online first-window MSE 1.40055；last-window 0.205974；
- max/mean preclip grad norm 2.72398/0.624754；terminations 5；nonfinite 0；
- checkpoint SHA `161ba97c86bf32602d757e36ea1995546e11811dc21c76b32b0a213005d66d6a`。

它只是工程 smoke，不是正式 student。

## 8. Proxy student 10000-update 消融训练

用户执行了从零 student 的10000 optimizer updates；没有从 smoke model128 恢复。

- task：`Unitree-Go2-Velocity-MultiTerrain-HeightScan`；
- variant：proxy-teacher recurrent student behavior-cloning ablation；
- run：`proxy_teacher_student_fromscratch_trainseed42_updates10000_v1`；
- student training seed：42；evaluation scenario seed：无；
- source student checkpoint：无；current checkpoint：`student_model_10000.pt`；
- 70 env，七地形各10个；teacher actions 驱动环境；700,000 labelled transitions；
- 4 unique proxy teachers，terrain ID 不进入 student observation。

绝对目录：

`/home/kxj/rl_ws/unitree_rl_lab/logs/student_proxy_ablation/proxy_teacher_student_fromscratch_trainseed42_updates10000_v1`

验证：exit 0；checkpoint keys 为 model/optimizer/iter/infos；iteration=10000；optimizer 1 param group、12 states；nonfinite=0；terminations=908。

训练结果：

- initial fixed MSE 0.535064；final 0.165785；ratio 0.309841（下降约69%）；
- online first-window MSE 1.416928；last-window 0.021659；
- max/mean preclip grad norm 3.201061/0.242330。

证据 SHA：

- `result.json`：`c6a8d653aa91ce97b188a3ee27394621869a441bcbca120d6605312ed5b11d0f`；
- `student_model_10000.pt`：`f70ce64aa44a54ced84660e51c114d7b105f04f1fdfc876d93fb54eb9912cc5b`；
- console：`dd4a1d0d5aa9504a140c5a7c8fda40230a58cc1b1f00fd416a9f3ac870ca6b29`。

重要：训练 result 内早期 purpose 字段仍写有 `smoke_only`，但实际命令是10000-update proxy ablation；科学定位仍是消融，不是最终 student。训练 loss 下降只证明 imitation 学习，不能证明 closed-loop locomotion。

## 9. 当前 evaluator 改动与待执行验证

为测试 GRU student，新建 evaluation-only task：
`Unitree-Go2-Velocity-ProxyStudent-HeightScan`。它复用七地形 HeightScan env/play cfg，但绑定 `V5FoundationPPORunnerCfg` recurrent GRU schema；不改变历史 HeightScan task 的 MLP runner 语义。

`scripts/project/evaluate_go2.py` 已允许该 task 使用 `multiterrain_stratified`，并在 evaluation 中调用 `runner.load(..., load_optimizer=False)`，因为评估无需加载 PPO optimizer，且 student optimizer 不是 PPO optimizer。

完整 closed-loop 验证命令已给用户，但截至本 handoff **尚未执行**。计划 output root：

`/home/kxj/rl_ws/unitree_rl_lab/evaluation/proxy_teacher_student_model10000_trainseed42_full_validation_v1`

矩阵共21次：

1. evaluation scenario seeds 42/7/123 的 mixed-command multiterrain stratified；
2. 每个 seed 再跑 forward、reverse、lateral-left/right、yaw-left/right；
3. 每次 70 env、七地形、level 0–9、episodes_per_env=2、max_steps=3000；
4. 最后创建 `all_evaluation_evidence.sha256`。

新助手首先应检查该 root 是否已经出现。如果用户回传运行结果，逐项核对 exit、结果文件、terrain/level/command coverage、tracking、progress、stall、净位移、path length、timeout、base contact、bad orientation、跨 seed 稳定性和 SHA。yaw 原地旋转不能以净位移判断。若 root 不存在，不得假装验证已完成。

## 10. 对最终目标的准确定位

当前成果回答的是：“multi-checkpoint dispatcher 与从零 recurrent student 是否能形成可学习的工程闭环？”答案是 imitation/offline-calibration 层面已通过。

尚未回答：

- model10000 是否能在 closed-loop 七地形上稳定运动；
- 4个 proxy policies 是否足以替代7个高质量 teachers；答案原则上是否定的；
- 七个 terrain-specific from-scratch teachers 是否均完成；尚未完成；
- 最终 unified student 是否完成；尚未完成；
- 多 training seeds 的科学最优性；尚未建立。

正式路线仍是冻结共享 schema和独立 namespaces，七 teachers 各自从零训练/多 seed 评估，再训练真正统一 student；当前 proxy route只作为架构与方法消融。

## 11. 收到新输出时的固定分析协议

每次先识别：task、task variant、run name、training seed、evaluation scenario seed、source checkpoint、current checkpoint、绝对目录和时间。

训练：exit status；checkpoint连续性/SHA；iteration/optimizer；TensorBoard；console/GPU log；tracking/progress/stall；curriculum/termination；吞吐/GPU；NaN/Inf。

评估：terrain/level/command覆盖；episodes；tracking error；command-aligned progress；净位移；path length；stall；timeout；base contact；bad orientation；跨 evaluation seeds 稳定性。视频另查 metadata、CSV、帧率、时长、任务视觉完成度和SHA。timeout绝不等于成功。

## 12. 新会话强制读取顺序

1. 本 V8 文件，从第一行完整读到末尾；
2. `doc/repro/session_handoff_v8_proxy_student_ablation_2026-08-29.sha256`；
3. `doc/SESSION_HANDOFF_2026-08-28_V7_CURRENT_DEMO_AND_MASTER_ROUTE_ALL_HISTORY.md` 及 V7 manifest；
4. `doc/PROJECT_MASTER_GOAL_7TERRAIN_TEACHER_STUDENT_2026-08-28.md`；
5. V6、V5 stairs、HeightScan handoff 及各自 manifest；
6. 当前 Demo v1 结果与 manifest；
7. boxes proxy 结果与 manifest；
8. proxy registry、offline audit、online dispatcher v2 证据；
9. student smoke v2 与10000-update result/checkpoint metadata/optimizer/console；
10. 当前 evaluator、task registration、dispatcher/student scripts；
11. 若完整验证 root 已存在，读取其中全部 summary/episode CSV/console/manifest；
12. 开始工作前只读检查 Unitree/IsaacLab HEAD与dirty状态、V8及manifest SHA、关键checkpoint和残留进程。

## 13. 新会话首次回复必须明确

- HeightScan model10000与34秒Demo；foundation3200与eval seeds 7/42/123；stairs model2200恢复语义和model4999；ascending model900接受/model1199拒绝；47秒Demo v1是multi-checkpoint showcase。
- boxes model900仅是bounded proxy；七 terrain dispatch entries实际只有4 unique policies。
- dispatcher online v2与student imitation smoke v2通过。
- proxy student model10000来自training seed42、从零、10000 updates，SHA为 `f70ce64a...cc5b`；训练loss下降但closed-loop完整验证尚未确认。
- 最终目标仍是七个from-scratch teachers与真正from-scratch unified recurrent perceptive student。
- 不自行启动任何训练、评估、录制或合成；等待用户回传完整验证输出并基于本地证据分析。

## 14. 已完成事项，禁止重复

V7 第10节列出的全部历史训练/评估/录制/合成；boxes specialist/mixed/safety训练与既有评估/视频；registry offline audit；dispatcher online v2；student imitation smoke v2；proxy student 10000-update训练。失败的v1不算成功，但其同名证据不得覆盖（若存在）。不要重新训练同名 model10000，也不要创建同名 validation root。
