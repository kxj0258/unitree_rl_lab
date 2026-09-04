# Go2 Teacher v2/v3 训练与验证结果

更新时间：2026-09-04

## 命令契约

最终部署命令是非负前进速度 `vx` 与相对当前朝向 ±45° 的 heading target，`vy=0`。不要求后退、横移或原地旋转。Teacher v1 因训练命令域与该契约不完全一致而冻结为历史 baseline；Teacher v2/v3 使用 heading command 路线。

## Teacher v2

七个 Teacher v2 task 已注册。已接受候选：flat `model_4000`、random rough `model_4800`、pyramid slope `model_4999`、inverted pyramid slope `model_4700`。这些候选完成了随机命令筛选、固定命令矩阵和人工视频检查。

原始 Pyramid Stairs、首次 stairs remediation、Inverted Stairs 与 Boxes 暴露不同程度的前腿跳跃、后腿拖行或单腿长期少参与。继续训练旧 checkpoint 不保证脱离局部最优，因此保留旧产物并从零训练 V3 专项任务。

## Teacher v3 设计

三个专项任务均使用 HeightScan，并在原速度/heading 跟踪、安全、能耗和动作平滑项之外加入：

- 最弱腿参与约束与四足 duty-factor 对称；
- 最坏负载足滑动惩罚；
- terrain-relative swing peak 和受地形调制的 gait clock；
- 沿目标 heading 的进展、横向漂移和 stall gate；
- Inverted Stairs 的 elevation progress；
- Boxes 的 support-edge penalty 与低/中难度混合课程。

正式训练均为 training seed 42、从零初始化、2,048 environments、5,000 PPO updates、32 rollout steps、16 mini-batches。每个任务使用 327,680,000 transitions。

| Terrain | Run start | Final checkpoint | SHA-256 |
|---|---|---|---|
| Pyramid Stairs | 2026-09-02 10:57 | `model_4999.pt` | `a46577f3f52c3faa4afd69a8a9d238af4717f347f91946d3b4d1c530caabb196` |
| Inverted Pyramid Stairs | 2026-09-02 16:54 | `model_4999.pt` | `598cd3e28127e302f5e6eff2e0e2aa3fa253a0a57f14cca3af570618ab8ce908` |
| Boxes | 2026-09-02 21:52 | `model_4999.pt` | `db5148a5daa4c21d24af7513ba49578354dfbccfbfaddb6b4107474f5a3e48ac` |

训练 TensorBoard 各有 5,000 点，无 NaN/Inf；optimizer state 完整。最终 100-iteration mean XY error 分别约为 0.0569、0.0570 和 0.0535 m/s。

## Candidate screen

协议：evaluation scenario seed 123、100 environments、2 episodes/environment、level 0--9 均匀覆盖、随机训练命令、运行期 curriculum 冻结。九个候选共 1,800 episodes，无非有限值。证据 manifest SHA-256：

```text
bce0bf9fd69775e99d7f12accd5f01dce0af57568541826abdf2af3bddcc1919
```

筛选结果：

- Pyramid Stairs：`model_4999`，200/200 timeout，zero safety termination，XY error 0.0431 m/s。
- Inverted Pyramid Stairs：`model_4700`，200/200 timeout，zero safety termination，XY error 0.0447 m/s；checkpoint SHA-256 `de8250790ae1e04437d014803edce17771045cd80e73735e37b81e79326fc640`。
- Boxes：暂选 `model_4999`，但 timeout 仅 82.5%，bad-orientation 16%；高难度泛化不足。

## 双视角 gait gate

协议：evaluation scenario seed 42、terrain level 5、`vx=0.40 m/s`、`vy=0`、relative heading 0、20 s；每个策略同时保留 terrain-fixed 与 asset-follow 视频。证据 manifest SHA-256：

```text
5926ade0400d7695b0549b57538e29d0d0c86a30021d93bb788b19cdd5c09775
```

| Candidate | Signed progress | Path efficiency | Stall | Duty spread | Result |
|---|---:|---:|---:|---:|---|
| Pyramid Stairs `model_4999` | 7.398 m | 99.49% | 0% | 0.053 | Numeric pass; visual review pending |
| Inverted Stairs `model_4700` | 7.327 m | 99.21% | 0% | 0.057 | Numeric pass; visual review pending |
| Boxes `model_4999` | -0.043 m | -0.58% | 11.11% | 0.110 | Fail: bad orientation, no useful net progress |

Timeout 只表示 episode 到达时限，不等于任务成功；必须同时检查净位移、path length、stall、安全终止、足接触指标和视频。

## 下一步

1. 完成两种楼梯视频的人工步态验收。
2. 分析 Boxes level-5 失败轨迹并设计受控修复；当前 Boxes V3 不得作为最终 teacher。
3. 合格候选再执行 evaluation scenario seeds 7/42/123 与 fixed-command matrix。
4. 最终七个 teacher 全部按部署契约验收后，才从零训练真正统一的 recurrent perceptive student。
