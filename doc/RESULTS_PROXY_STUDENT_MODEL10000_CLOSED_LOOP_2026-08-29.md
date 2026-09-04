# Proxy Student model10000：完整闭环验证与诊断视频结论

日期：2026-08-29（Asia/Shanghai）

## 科学定位

该模型是四个 unique proxy policies、七个 terrain dispatch entries 的工程消融，不是最终七教师统一
student。Student 从随机初始化开始，以 training seed 42 进行了 10000 次 imitation optimizer update；
训练环境由 teacher actions 驱动。

Checkpoint：

`logs/student_proxy_ablation/proxy_teacher_student_fromscratch_trainseed42_updates10000_v1/student_model_10000.pt`

SHA-256：`f70ce64aa44a54ced84660e51c114d7b105f04f1fdfc876d93fb54eb9912cc5b`。

Checkpoint iteration 为 10000；包含 33 个 model state entries、1 个 optimizer param group、12 个
optimizer states，模型与 optimizer tensor 均为有限值。

## Imitation 结果

- fixed calibration MSE：0.535064 → 0.165785，ratio 0.309841；
- online first-window MSE：1.416928；
- online last-window MSE：0.021659；
- nonfinite：0。

这些结果证明 imitation learning 和 dispatcher/training 工程链路成立，不证明 closed-loop locomotion。

## 完整 closed-loop validation

Task：`Unitree-Go2-Velocity-ProxyStudent-HeightScan`。

- training seed：42；
- evaluation scenario seeds：7、42、123；
- 21/21 runs；
- 2940 episodes；
- 七地形、levels 0–9；
- mixed commands、forward、reverse、lateral-left/right、yaw-left/right；
- output：`evaluation/proxy_teacher_student_model10000_trainseed42_full_validation_v1`。

全矩阵 episode 聚合：

- bad-orientation rate：91.9048%；
- timeout rate：8.0612%；
- base-contact rate：0.0340%；
- mean command-aligned progress：0.131204 m/s；
- mean stall fraction：17.8515%。

方向诊断：

- forward progress 约 0.2151 m/s，但 bad orientation 93.57%；
- reverse progress 约 0.0835 m/s，stall 42.98%；
- lateral-right progress 约 0.1071 m/s，stall 35.71%；
- yaw-left/right 的 mean yaw error 约 0.2739/0.1689 rad/s。

三个 evaluation scenario seeds 结果相近，表明失败可稳定复现。Evaluator exit 0 只表示评估完成；
timeout 不代表运动或任务成功。

完整 evaluation manifest 自身 SHA-256 为
`b46ce39bc4f10ecbb55c770a0e1c814fa6e04c8684f747eab34c5d6da0eb659a`。该 manifest 错误地包含
自身条目，因此 `sha256sum -c` 仅对 manifest 自身报告不匹配；其中列出的全部 JSON、CSV 和 console
均通过校验。不得覆盖该历史 manifest。

## 六场景诊断视频

Output：`evaluation/video_proxy_student_model10000_trainseed42_diagnostics_v1`。

- flat level0 forward：10 s 无终止，前向位移 1.796 m，stall 0；
- flat level0 reverse：step 252 bad orientation，位移 -0.320 m，stall 47.56%；
- flat level0 lateral-right：10 s 无终止，但跟踪弱且有非目标方向漂移；
- flat level0 yaw-left：10 s 无终止，mean yaw error 0.2645 rad/s；
- inverted-pyramid-stairs level7 forward：steps 125、456 两次 bad orientation，最终净前进 0.103 m；
- boxes level7 reverse：step 99 bad orientation，stall 85.56%，实际反向速度很低。

诊断 manifest SHA-256：
`7ec73d146a92f543c369efe6fe8a10cbeb0908cc229cdcfe90d627ac2e7af353`；其中全部 artifacts 校验通过。

## 固化结论

Proxy model10000 是有价值的负结果基线：teacher-driven、逐步 detach 的 recurrent behavior cloning 即使
取得低 imitation loss，也不足以产生可靠 closed-loop locomotion。不得把该模型作为成功统一策略或
成果 Demo；不得重复同名训练、完整验证或诊断录制。

项目主路线重置为：七个 terrain-specific teachers 在各自明确的可行速度/转向命令域内分别从零训练和
验证，再从零训练真正统一的 recurrent perceptive student。极端地形不再强制与平地使用完全相同的
全向命令盒。
