# Unitree Go2 当前成果多 Checkpoint Demo v1（2026-08-28）

## 1. 定位与结论

本成片是当前已验证结果的审计型作品集，不是单一七地形统一策略。它组合三个职责不同的策略：

- 历史 HeightScan `model_10000`：展示 flat、random rough、pyramid slope、
  inverted pyramid slope；
- v5 stairs specialist `model_4999`：展示 pyramid stairs 能力及 inverted stairs 已诊断短板；
- ascending-remediation `model_900`：展示更难 inverted stairs level 7 的专项修复结果。

所有策略的 training seed 均为 `42`；成片所用 evaluation scenario seed 均为 `42`。不得把这两类
seed 混为一谈，也不得把多个 checkpoint 描述成一个已经融合的策略。

## 2. 正式成片

路径：

`evaluation/portfolio_go2_current_results_multicheckpoint_demo_v1_2026-08-28.mp4`

属性：

- 1280 x 720；
- 50 FPS；
- 2350 frames；
- 47.0 s；
- 24,972,117 bytes；
- SHA-256：`3199ab908f2e7b1bd8091ccdb9c7281154b3eeb483d78325ff2851c7bcaa217f`。

sidecar 校验在 `evaluation/` 目录运行：

```bash
sha256sum -c portfolio_go2_current_results_multicheckpoint_demo_v1_2026-08-28.sha256
```

结果为 `成功`。

## 3. 时间线

| 成片时间 | 内容 | checkpoint/结论 |
|---|---|---|
| 0--3 s | 标题与声明 | 明确 multi-checkpoint，非统一策略 |
| 3--15 s | 四类连续地形 | HeightScan model10000，每类 3 s |
| 15--21 s | pyramid stairs level 6 | model4999，净前进 6.41 m |
| 21--23 s | failure-driven remediation 说明 | 解释专项修复依据 |
| 23--31 s | inverted stairs level 6 失败诊断 | model4999，净前进 0.89 m、stall 42.2% |
| 31--43 s | inverted stairs level 7 修复结果 | model900，净前进 7.22 m、stall 0% |
| 43--47 s | 长期路线 | 七 teachers、多教师蒸馏、PPO consolidation |

model4999 与 model900 两段都来自固定 `vx=0.35 m/s` 的原始录制证据。timeout 不被单独当作成功；
失败段明确使用净位移和 stall，成功段同时要求有效前进、无 base contact、无 bad orientation。

## 4. 可复现与不可覆盖

合成脚本：

- `scripts/project/compose_go2_current_results_demo_v1.py`；
- `scripts/project/compose_go2_current_results_demo_v1.sh`。

脚本固定每个源视频的预期 SHA-256，并在合成前验证源文件、帧率、分辨率和裁剪范围；目标视频、
JSON 或 SHA sidecar 任一已存在时拒绝覆盖。输出 JSON 保存完整源 inventory、裁剪区间、成片属性与
最终 SHA-256。

本阶段证据清单：

`doc/repro/current_multicheckpoint_demo_v1_2026-08-28.sha256`

## 5. 视觉 QA 与已知限制

抽检开头、所有地形段、failure/remediation 转场、model4999/model900 对照和片尾：无黑帧、截断、
空文件或编码异常，文字均可读。

已知展示瑕疵：前四段使用已经带 overlay 的 34 秒 HeightScan portfolio，再叠加本片统一 banner，
顶部存在轻微旧文字重影。这不影响 checkpoint 身份、指标或证据有效性，因此当前文件固化为审计版
Demo v1；若制作简历投递精修版，应以新文件名生成 v1.1，禁止覆盖本片。

## 6. 后续路线

本片完成 Track A 的当前成果展示。Track B 仍按
`doc/PROJECT_MASTER_GOAL_7TERRAIN_TEACHER_STUDENT_2026-08-28.md` 执行：七种地形分别从零训练
teachers，再从零训练统一 recurrent perceptive student，采用平衡在线多教师蒸馏和 PPO
consolidation，最后验证同 episode mixed-terrain transitions。只有最终单一 student 通过七地形固定
分层评估后，才能制作并声称统一七地形 Demo。
