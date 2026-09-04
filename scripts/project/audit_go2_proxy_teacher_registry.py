#!/usr/bin/env python3
"""Audit and inference-smoke the seven-terrain proxy-teacher registry without Isaac Sim."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCriticRecurrent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def make_policy(actor_dim: int, critic_dim: int, action_dim: int, hidden_dim: int) -> ActorCriticRecurrent:
    policy = ActorCriticRecurrent(
        obs=TensorDict(
            {"policy": torch.zeros(1, actor_dim), "critic": torch.zeros(1, critic_dim)},
            batch_size=[1],
        ),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=action_dim,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.8,
        rnn_type="gru",
        rnn_hidden_dim=hidden_dim,
        rnn_num_layers=1,
    )
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True, help="New JSON audit output; parent must exist.")
    args = parser.parse_args()
    registry_path = Path(args.registry).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite: {output_path}")
    registry = json.loads(registry_path.read_text())
    terrain_dispatch = registry["terrain_dispatch"]
    expected_terrains = {
        "flat", "random_rough", "pyramid_slope", "inverted_pyramid_slope",
        "pyramid_stairs", "inverted_pyramid_stairs", "boxes",
    }
    if set(terrain_dispatch) != expected_terrains:
        raise RuntimeError(f"terrain dispatch mismatch: {sorted(terrain_dispatch)}")
    if registry["student_observation_contains_terrain_id"] is not False:
        raise RuntimeError("terrain identity must not be part of the student observation")

    actor_dim = int(registry["actor_observation_dim"])
    critic_dim = int(registry["critic_observation_dim"])
    action_dim = int(registry["action_dim"])
    hidden_dim = int(registry["rnn"]["hidden_dim"])
    reports = {}
    policies = {}
    reference_shapes = None
    for teacher_id, cfg in registry["teachers"].items():
        checkpoint = Path(cfg["checkpoint"])
        actual_sha = sha256_file(checkpoint)
        if actual_sha != cfg["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {teacher_id}: {actual_sha}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload["iter"]) != int(cfg["checkpoint_iteration"]):
            raise RuntimeError(f"iteration mismatch for {teacher_id}: {payload['iter']}")
        if not finite_tree(payload):
            raise RuntimeError(f"non-finite checkpoint content: {teacher_id}")
        state = payload["model_state_dict"]
        shapes = {key: list(value.shape) for key, value in state.items()}
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise RuntimeError(f"model-state schema mismatch: {teacher_id}")
        policy = make_policy(actor_dim, critic_dim, action_dim, hidden_dim)
        policy.load_state_dict(state, strict=True)
        policies[teacher_id] = policy
        reports[teacher_id] = {
            "checkpoint": str(checkpoint), "sha256": actual_sha,
            "iteration": int(payload["iter"]), "model_state_entries": len(state),
            "optimizer_state_present": bool(payload.get("optimizer_state_dict")),
        }

    # One shared policy instance serves every environment routed to the same
    # teacher, while its recurrent state retains a separate batch slot per env.
    terrain_order = list(terrain_dispatch)
    teacher_env_ids = {
        teacher_id: [index for index, terrain in enumerate(terrain_order) if terrain_dispatch[terrain] == teacher_id]
        for teacher_id in policies
    }
    torch.manual_seed(20260829)
    actions_by_terrain = {}
    hidden_reset_checks = {}
    for teacher_id, policy in policies.items():
        env_ids = teacher_env_ids[teacher_id]
        batch = len(env_ids)
        obs = TensorDict(
            {"policy": torch.randn(batch, actor_dim), "critic": torch.randn(batch, critic_dim)},
            batch_size=[batch],
        )
        # Use no_grad rather than inference_mode because RSL-RL resets selected
        # recurrent batch slots in-place after inference.
        with torch.no_grad():
            first = policy.act_inference(obs)
            second = policy.act_inference(obs)
        if first.shape != (batch, action_dim) or not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise RuntimeError(f"invalid inference output: {teacher_id}")
        hidden_before = policy.memory_a.hidden_state.clone()
        dones = torch.zeros(batch, dtype=torch.bool)
        dones[0] = True
        policy.reset(dones)
        hidden_after = policy.memory_a.hidden_state
        reset_zero = bool(torch.count_nonzero(hidden_after[:, 0, :]) == 0)
        untouched = True if batch == 1 else bool(torch.equal(hidden_before[:, 1:, :], hidden_after[:, 1:, :]))
        if not reset_zero or not untouched:
            raise RuntimeError(f"hidden-state reset isolation failed: {teacher_id}")
        hidden_reset_checks[teacher_id] = {"batch_envs": batch, "reset_slot_zero": reset_zero, "other_slots_unchanged": untouched}
        for local_id, global_id in enumerate(env_ids):
            actions_by_terrain[terrain_order[global_id]] = {
                "teacher_id": teacher_id,
                "action_shape": list(first[local_id].shape),
                "action_mean": float(first[local_id].mean()),
                "action_std": float(first[local_id].std(unbiased=False)),
                "finite": bool(torch.isfinite(first[local_id]).all()),
            }

    result = {
        "status": "pass",
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "unique_teacher_count": len(policies),
        "terrain_entry_count": len(terrain_dispatch),
        "shared_foundation_dispatch_count": sum(value == "foundation3200" for value in terrain_dispatch.values()),
        "schema": {"actor_observation_dim": actor_dim, "critic_observation_dim": critic_dim,
                   "action_dim": action_dim, "rnn_hidden_dim": hidden_dim},
        "teachers": reports,
        "terrain_actions": actions_by_terrain,
        "hidden_reset_checks": hidden_reset_checks,
        "terrain_id_in_student_observation": False,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
