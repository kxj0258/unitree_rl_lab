# Go2 MultiTerrain 原型结果

## 实验定位

本阶段完成 `Unitree-Go2-Velocity-MultiTerrain` 的原型闭环：七类地形统一训练、固定分层评估，以及 TorchScript/ONNX 导出。结果只来自一条 training seed 42 的连续 resume 训练链，不是独立多 training-seed 正式实验。

策略保持 45 维 observation，不使用 height scan。地形包括 flat、random rough、正/反 pyramid slope、正/反 pyramid stairs 和 boxes。

## 训练链

| 阶段 | Run 末 checkpoint | 新增 iterations | 主要结论 |
|---|---:|---:|---|
| smoke | `model_4.pt` | 5 | task、七类地形和训练链路正常 |
| pilot | `model_499.pt` | 500 | 原 distance curriculum 锁在 level 0 |
| v1 resume | `model_1998.pt` | 1500 | policy 改善，但 terrain level 仍为 0 |
| v2 command fix | `model_2197.pt` | 200 | 扩大 command range 仍不足以推进 terrain |
| v3 survival curriculum | `model_2396.pt` | 200 | terrain level 从约 0.01 提升至 4.25 |
| v3 prototype | `model_2995.pt` | 600 | 最终 terrain level 5.77；完成 checkpoint 候选比较 |

六个 run 均 exit 0，共保存 37 个 checkpoint 和 6 个 TensorBoard event。TensorBoard 与 GPU telemetry 检查未发现非有限值。所有阶段属于同一个 training seed 42 policy lineage。

## 固定分层评估

`multiterrain_stratified` smoke v0 使用 evaluation scenario seed 42：

- 70 environments × 2 episodes = 140 episodes/run；
- 七类 terrain 各 10 environments、20 episodes；
- 每类覆盖 level 0–9，每个 type×level cell 2 episodes；
- deterministic policy；
- runtime terrain/command curriculum 关闭；
- episode 期间 terrain column/level 固定。

| Checkpoint | Reward ↑ | XY error ↓ | Yaw error ↓ | Timeout ↑ | Bad orientation ↓ |
|---|---:|---:|---:|---:|---:|
| `model_2600.pt` | 13.7961 | 0.3200 | 0.2014 | 94/140 | 46/140 |
| `model_2900.pt` | 15.7300 | 0.3079 | 0.1812 | 105/140 | 35/140 |
| **`model_2995.pt`** | **16.1924** | **0.2918** | **0.1646** | **108/140** | **32/140** |

三组均为 140 episodes、base-contact termination 0、non-finite 0。CSV/JSON 的 checkpoint hash、environment/episode identity、terrain type/level coverage、termination 和总体聚合已逐字段审计。

`model_2995.pt` 是当前 provisional Demo/export checkpoint。其 per-terrain timeout 为 flat 100%、inverted slope 85%、slope 80%、inverted stairs 100%、random rough 75%、boxes 65%、pyramid stairs 35%。正向上楼梯是当前无 height-scan policy 的主要短板。

## 导出

| Artifact | Size | SHA-256 |
|---|---:|---|
| `model_2995.pt` | 4,639,669 bytes | `67da571685f42e56df412730cd6c13c335a63aa85a07beaf7f97b35cfb3dd2da` |
| `exported/policy.pt` | 769,426 bytes | `2acbed11b1ff6648d8755d576bd9bc42bffc35d283d2cb59ae3dcee852dfb304` |
| `exported/policy.onnx` | 758,549 bytes | `0e13dd581d4606d52bb8642eeaa68ce5d2e81504bf287e2609aa8485682a4f4a` |

ONNX checker 通过，接口为 `obs [batch,45]` 到 `actions [batch,12]`。8 组随机 observation 的 TorchScript/ONNX 最大绝对差为 `2.3841858e-6`，`allclose(rtol=1e-5, atol=1e-5)` 通过。

## 结论与边界

该原型证明了无 height-scan 的统一 policy 能在七类固定难度分层环境中形成初步泛化，并完成可部署格式导出。但结果只有一个 training seed 和一个 evaluation scenario seed，不能描述为正式多-seed 结论，也不能与既有 `rough_stratified v1` 指标直接合并。

尚未完成：正式视频、多 evaluation scenario seeds、多 independent training seeds、受控消融、MuJoCo Sim2Sim、Sim2Real 或实机部署。

归档清单位于 `doc/repro/`，只记录现有 checkpoint、evaluation 和 export 的路径、大小与 SHA-256；实际大文件继续保留在被 Git 忽略的原始位置。
