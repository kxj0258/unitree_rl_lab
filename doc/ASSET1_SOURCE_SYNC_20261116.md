# ASSET1 代码归档审查

日期：2026-11-16，Asia/Shanghai。用户授权本阶段提交、推送现有 origin/main；后续资产备份、恢复核验与保守清理分别办理。

基线 `896f60823fe26904d77f14ef1dc8b9798b91185d` 与实时远端一致。审查原有 17 个小文件：教师 registry/selection、具名观测与 GRU 状态适配、验收/导出/MuJoCo 候选入口、说明及小型证据。排除 `env_backup_unitree_rl.yml`，不提交 logs/evaluation/checkpoint，不变更历史资产。

验证：8 个 Python 文件 AST 解析通过；6 项 `scripts/project/test_teacher_bank.py` 回归通过；`accept_go2_teacher_bank.py --mode audit` 七教师真实权重 SHA、配置 SHA、strict load、有限推理、reset 隔离及 replay 均通过，CPU 执行。常见密钥/带密码 URL/二进制扫描无命中。未执行 Isaac/MuJoCo rollout 或训练；ONNX/MuJoCo 功能仍是待专门验证候选，不能标 verified。复核已有正式矩阵并更正 M1 文档的滞后状态，不改原始指标。

源码参考沿用本地固定 Isaac Lab 2.3.2、RSL-RL 和 MuJoCo 接口，本轮以现有代码保存与审查为目的，不迁移依赖版本。推送结果与跨仓 SHA 由 zte 父仓台账记录。
