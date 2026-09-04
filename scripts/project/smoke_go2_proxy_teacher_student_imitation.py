#!/usr/bin/env python3
"""From-scratch recurrent student imitation smoke using the proxy-teacher dispatcher."""

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

RSL_RL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))
import cli_args  # noqa: E402,F401

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--registry", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--training-seed", type=int, default=42, help="Student training, environment, and initialization seed.")
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0:
    parser.error("--steps must be positive")
if args.learning_rate <= 0.0:
    parser.error("--learning-rate must be positive")
output_dir = Path(args.output_dir).expanduser().resolve()
if output_dir.exists():
    parser.error(f"refusing to overwrite output directory: {output_dir}")
args.headless = True
app = AppLauncher(args).app

print("[IMITATION_SMOKE] Isaac Sim application launched", flush=True)
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from rsl_rl.modules import ActorCriticRecurrent  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
print("[IMITATION_SMOKE] Runtime imports and Unitree task registration complete", flush=True)

TASK = "Unitree-Go2-Velocity-MultiTerrain-HeightScan"
TERRAINS = [
    "flat", "random_rough", "pyramid_slope", "inverted_pyramid_slope",
    "pyramid_stairs", "inverted_pyramid_stairs", "boxes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def terrain_columns(cfg) -> list[str]:
    names = list(cfg.sub_terrains)
    proportions = [float(item.proportion) for item in cfg.sub_terrains.values()]
    total = sum(proportions)
    cumulative, running = [], 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)
    result = []
    for column in range(int(cfg.num_cols)):
        sample = column / int(cfg.num_cols) + 0.001
        result.append(names[next(index for index, upper in enumerate(cumulative) if sample < upper)])
    return result


def make_policy(device):
    return ActorCriticRecurrent(
        obs=TensorDict(
            {"policy": torch.zeros(1, 162, device=device), "critic": torch.zeros(1, 177, device=device)},
            batch_size=[1],
        ),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=12,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.8,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    ).to(device)


def dispatch(teachers, teacher_env_ids, obs):
    # A TensorDict can report ``device=None`` even when all leaf tensors are on CUDA.
    # Allocate beside the actual policy observation leaf, which is authoritative.
    targets = torch.empty(obs.batch_size[0], 12, device=obs["policy"].device)
    with torch.no_grad():
        for teacher_id, teacher in teachers.items():
            ids = teacher_env_ids[teacher_id]
            targets[ids] = teacher.act_inference(obs[ids])
    return targets


def stateless_student_actions(student, obs):
    student.reset()
    with torch.no_grad():
        result = student.act_inference(obs).detach().clone()
    student.reset()
    return result


def main():
    registry_path = Path(args.registry).expanduser().resolve()
    registry = json.loads(registry_path.read_text())
    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)

    print(f"[IMITATION_SMOKE] Parsing task configuration: {TASK}", flush=True)
    env_cfg = parse_env_cfg(
        TASK,
        device=args.device,
        num_envs=70,
        use_fabric=not args.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args.training_seed
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None
    generator = env_cfg.scene.terrain.terrain_generator
    generator.seed = args.training_seed
    generator.curriculum = True
    generator.num_rows = 10
    generator.num_cols = 20
    env_cfg.scene.terrain.max_init_terrain_level = 9

    print("[IMITATION_SMOKE] Creating 70-environment seven-terrain matrix", flush=True)
    base = gym.make(TASK, cfg=env_cfg)
    raw = base.unwrapped
    column_names = terrain_columns(generator)
    terrain_types, terrain_levels = [], []
    for terrain in TERRAINS:
        columns = [index for index, name in enumerate(column_names) if name == terrain]
        if not columns:
            raise RuntimeError(f"terrain has no generated column: {terrain}")
        terrain_types.extend(columns[index % len(columns)] for index in range(10))
        terrain_levels.extend(range(10))
    raw.scene.terrain.terrain_types[:] = torch.tensor(terrain_types, device=raw.device)
    raw.scene.terrain.terrain_levels[:] = torch.tensor(terrain_levels, device=raw.device)
    raw.scene.terrain.env_origins[:] = raw.scene.terrain.terrain_origins[
        raw.scene.terrain.terrain_levels, raw.scene.terrain.terrain_types
    ]
    base.reset()
    if isinstance(base.unwrapped, DirectMARLEnv):
        base = multi_agent_to_single_agent(base)
    env = RslRlVecEnvWrapper(base, clip_actions=100.0)
    obs = env.get_observations()
    if obs["policy"].shape != (70, 162) or obs["critic"].shape != (70, 177):
        raise RuntimeError(f"unexpected observation schema: {obs.batch_size}")
    print("[IMITATION_SMOKE] Environment reset complete", flush=True)

    teachers, teacher_env_ids, source_checkpoints = {}, {}, {}
    for teacher_id, teacher_cfg in registry["teachers"].items():
        checkpoint = Path(teacher_cfg["checkpoint"])
        actual_sha = sha256(checkpoint)
        if actual_sha != teacher_cfg["sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch: {teacher_id}")
        teacher = make_policy(raw.device)
        payload = torch.load(checkpoint, map_location=raw.device, weights_only=False)
        teacher.load_state_dict(payload["model_state_dict"], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teachers[teacher_id] = teacher
        teacher_env_ids[teacher_id] = torch.tensor(
            [
                terrain_index * 10 + level
                for terrain_index, terrain in enumerate(TERRAINS)
                for level in range(10)
                if registry["terrain_dispatch"][terrain] == teacher_id
            ],
            device=raw.device,
        )
        source_checkpoints[teacher_id] = {
            "path": str(checkpoint.resolve()),
            "sha256": actual_sha,
            "checkpoint_iteration": teacher_cfg["checkpoint_iteration"],
            "training_seed": registry["training_seed"],
        }
    print(f"[IMITATION_SMOKE] Loaded and froze {len(teachers)} unique proxy checkpoints", flush=True)

    student = make_policy(raw.device)
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    calibration_obs = obs.clone().detach()
    calibration_targets = dispatch(teachers, teacher_env_ids, calibration_obs).detach().clone()
    for teacher in teachers.values():
        teacher.reset()
    initial_actions = stateless_student_actions(student, calibration_obs)
    initial_calibration_loss = float(torch.mean((initial_actions - calibration_targets) ** 2))

    losses, grad_norms, terminations, nonfinite_count = [], [], 0, 0
    for step in range(args.steps):
        targets = dispatch(teachers, teacher_env_ids, obs)
        student_actions = student.act_inference(obs)
        loss = torch.mean((student_actions - targets) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at update {step}")
        optimizer.step()
        student.memory_a.detach_hidden_state()
        losses.append(float(loss.detach()))
        grad_norms.append(float(grad_norm.detach()))
        nonfinite_count += int((~torch.isfinite(targets)).sum()) + int((~torch.isfinite(student_actions)).sum())

        obs, _, dones, _ = env.step(targets.detach())
        terminations += int(dones.sum())
        student.reset(dones)
        for teacher_id, teacher in teachers.items():
            teacher.reset(dones[teacher_env_ids[teacher_id]])
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == args.steps:
            print(
                f"[IMITATION_SMOKE] update {step + 1}/{args.steps} "
                f"loss={losses[-1]:.6f} grad_norm={grad_norms[-1]:.6f}",
                flush=True,
            )

    final_actions = stateless_student_actions(student, calibration_obs)
    final_calibration_loss = float(torch.mean((final_actions - calibration_targets) ** 2))
    if nonfinite_count:
        raise RuntimeError(f"non-finite action values: {nonfinite_count}")
    if not math.isfinite(final_calibration_loss):
        raise RuntimeError("non-finite final calibration loss")

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / f"student_model_{args.steps}.pt"
    result_path = output_dir / "result.json"
    checkpoint = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": args.steps,
        "infos": {
            "purpose": "proxy_teacher_recurrent_student_imitation_smoke_only",
            "task": TASK,
            "training_seed": args.training_seed,
            "evaluation_scenario_seed": None,
            "from_scratch_student": True,
            "teacher_actions_drive_environment": True,
            "terrain_id_in_student_observation": False,
            "registry_path": str(registry_path),
            "registry_sha256": sha256(registry_path),
            "source_checkpoints": source_checkpoints,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    window = min(16, len(losses))
    result = {
        "status": "pass" if final_calibration_loss < initial_calibration_loss else "fail_loss_not_reduced",
        "purpose": "proxy_teacher_recurrent_student_imitation_smoke_only",
        "task": TASK,
        "training_seed": args.training_seed,
        "evaluation_scenario_seed": None,
        "steps_and_optimizer_updates": args.steps,
        "num_envs": 70,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "from_scratch_student": True,
        "teacher_actions_drive_environment": True,
        "terrain_id_in_student_observation": False,
        "unique_teacher_count": len(teachers),
        "registry_path": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "source_checkpoints": source_checkpoints,
        "initial_fixed_batch_mse": initial_calibration_loss,
        "final_fixed_batch_mse": final_calibration_loss,
        "fixed_batch_mse_ratio": final_calibration_loss / initial_calibration_loss,
        "first_window_online_mse": sum(losses[:window]) / window,
        "last_window_online_mse": sum(losses[-window:]) / window,
        "max_preclip_grad_norm": max(grad_norms),
        "mean_preclip_grad_norm": sum(grad_norms) / len(grad_norms),
        "terminations": terminations,
        "nonfinite_count": nonfinite_count,
        "checkpoint": str(checkpoint_path),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    result["checkpoint_sha256"] = sha256(checkpoint_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    if result["status"] != "pass":
        raise RuntimeError("fixed calibration loss did not decrease")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        print("[IMITATION_SMOKE] FAILED with traceback:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        app.close()
    raise SystemExit(exit_code)
