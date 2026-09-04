# V10 Session Handoff: Teacher v2/v3 Training and Gait Validation

Date: 2026-09-04

## Canonical objective

Train seven terrain-specific teachers from scratch under the deployable command contract (non-negative forward `vx`, `vy=0`, relative heading ±45°), validate them with stratified metrics and videos, and then train one recurrent perceptive student from scratch. No final unified student exists yet.

## Historical facts that must remain explicit

- HeightScan `model_10000` and the historical 34-second demo are preserved.
- Foundation `model_3200` was evaluated with scenario seeds 7/42/123.
- Stairs training seed 42 used `model_2200` resume semantics and ended at `model_4999`.
- Ascending `model_900` was accepted; `model_1199` was rejected for regression.
- The 47-second Demo v1 is a multi-checkpoint showcase, not one unified policy.
- Boxes safety `model_900` is only a bounded proxy teacher.
- Seven old dispatch entries represented only four unique proxy policies.
- Dispatcher online v2 and student imitation smoke v2 passed.
- Proxy student `model_10000` was trained from scratch for 10,000 updates with training seed 42; SHA-256 `f70ce64aa44a54ced84660e51c114d7b105f04f1fdfc876d93fb54eb9912cc5b`. Its falling imitation loss did not imply closed-loop success; full validation was poor.
- Seven Teacher v1 policies were independently trained from scratch for 5,000 updates and fully screened. Their broader command domain does not match the final interface, so they are frozen baselines.
- Teacher v1 Demo composed_v2 contains seven independent full videos and one overview, keeps the KNOWN ISSUE, uses bilingual titles and English numeric HUD.

## Current accepted/pending policies

Teacher v2 accepted: flat `model_4000`, random rough `model_4800`, pyramid slope `model_4999`, inverted pyramid slope `model_4700`.

Teacher v3 candidate-screen selections: pyramid stairs `model_4999`, inverted pyramid stairs `model_4700`, boxes `model_4999` provisional. Training seed is 42; evaluation scenario seeds are separately recorded.

The two stairs candidates pass the level-5 numeric gait gate, but visual review remains pending. Boxes fails the level-5 trajectory and is not accepted. Do not proceed to a final unified student until all seven teachers satisfy the final command contract and visual/quantitative gates.

## Evidence

- Formal Teacher v3 runs: `logs/rsl_rl/unitree_go2_velocity_teacher_v3_*` dated 2026-09-02.
- Candidate screen: `evaluation/teacher_v3_candidate_screen_trainseed42_evalseed123_randomcommand_2026-09-03`.
- Candidate-screen manifest SHA-256: `bce0bf9fd69775e99d7f12accd5f01dce0af57568541826abdf2af3bddcc1919`.
- Dual-view gait evidence: `evaluation/teacher_v3_*_gait_gate_*_2026-09-03`.
- Dual-view manifest SHA-256: `5926ade0400d7695b0549b57538e29d0d0c86a30021d93bb788b19cdd5c09775`.
- Detailed results: `doc/RESULTS_TEACHER_V2_V3_2026-09-04.md`.

## Operating constraints

Do not upgrade NVIDIA, Isaac Sim, Isaac Lab, or PyTorch. Do not overwrite, move, or delete checkpoints, logs, GPU CSV, TensorBoard events, evaluations, or videos. Do not clean the dirty worktree or modify/delete `env_backup_unitree_rl.yml`. The assistant must provide complete runtime commands for the user to execute. Never confuse training seed with evaluation scenario seed, and never treat timeout alone as success. Future interactive commands must not end with `exit`, because that closes the user's terminal window.
