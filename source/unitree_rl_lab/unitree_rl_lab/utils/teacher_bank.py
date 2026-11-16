"""Frozen teacher contracts and deterministic named-observation adapters.

This module deliberately has no Isaac Sim imports. Raw terms are unnormalised,
unscaled tensors; clipping precedes scaling, and each policy owns its normalizer
and recurrent state. Terrain IDs are routing metadata, never policy inputs.
"""

import hashlib
import json
from pathlib import Path

import torch
import yaml
from tensordict import TensorDict


TERRAINS = ("flat", "random_rough", "pyramid_slope", "inverted_pyramid_slope",
            "pyramid_stairs", "inverted_pyramid_stairs", "boxes")
TERM_DIMS = {"base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3,
             "velocity_commands": 3, "joint_pos_rel": 12, "joint_vel_rel": 12,
             "joint_effort": 12, "last_action": 12, "height_scan": 117, "gait_clock": 2}


class SnapshotLoader(yaml.SafeLoader):
    """Read Isaac config dumps as data; never instantiate Python objects."""


def _tagged_data(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


SnapshotLoader.add_multi_constructor("tag:yaml.org,2002:python/", _tagged_data)


def read_snapshot(path):
    return yaml.load(Path(path).read_text(), Loader=SnapshotLoader)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path, project_root):
    registry = json.loads(Path(path).read_text())
    if registry.get("schema_version") != 1 or registry.get("kind") != "go2_seven_teacher_bank":
        raise ValueError("unsupported teacher registry")
    if set(registry["teachers"]) != set(TERRAINS):
        raise ValueError("registry must contain exactly seven terrain teachers")
    checkpoints = set()
    for terrain, entry in registry["teachers"].items():
        checkpoint = str((Path(project_root) / entry["checkpoint"]).resolve())
        if checkpoint in checkpoints:
            raise ValueError("each terrain must have an independent teacher checkpoint")
        checkpoints.add(checkpoint)
        for group in ("policy", "critic"):
            terms = entry["observations"][group]
            names = [term["name"] for term in terms]
            if len(names) != len(set(names)) or any(n not in TERM_DIMS for n in names):
                raise ValueError(f"invalid named observation contract: {terrain}/{group}")
            if sum(t["dim"] for t in terms) != entry[f"{group}_dim"]:
                raise ValueError(f"dimension mismatch: {terrain}/{group}")
        if entry["action_dim"] != 12:
            raise ValueError("Go2 requires twelve actions")
    return registry


def verify_artifacts(entry, project_root):
    for artifact in [{"path": entry["checkpoint"], "sha256": entry["checkpoint_sha256"]},
                     *entry["config_artifacts"].values()]:
        path = Path(project_root) / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {path}")


def gait_clock(episode_steps, step_dt, forward_speed, params):
    """Match the saved speed-adaptive clock, including phase changes on commands."""
    fraction = ((forward_speed.abs() - params["minimum_speed"]) /
                (params["maximum_speed"] - params["minimum_speed"])).clamp(0, 1)
    period = params["period"] + fraction * (params["minimum_period"] - params["period"])
    phase = torch.remainder(episode_steps * step_dt, period) / period * (2 * torch.pi)
    return torch.stack((phase.sin(), phase.cos()), dim=-1)


class ObservationAdapter:
    def __init__(self, entry):
        self.entry = entry

    def build(self, raw_terms, group="policy"):
        values = []
        batch = device = dtype = None
        for term in self.entry["observations"][group]:
            name = term["name"]
            if name not in raw_terms:
                raise ValueError(f"missing raw observation term: {name}")
            value = raw_terms[name]
            if value.ndim != 2 or value.shape[1] != term["dim"]:
                raise ValueError(f"invalid shape for {name}: {tuple(value.shape)}")
            if batch is None:
                batch, device, dtype = value.shape[0], value.device, value.dtype
            if (value.shape[0], value.device, value.dtype) != (batch, device, dtype):
                raise ValueError("raw observation terms must share batch/device/dtype")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"non-finite or non-floating observation: {name}")
            if term["clip"] is not None:
                value = value.clamp(*term["clip"])
            if term["scale"] is not None:
                value = value * torch.as_tensor(term["scale"], device=device, dtype=dtype)
            values.append(value)
        result = torch.cat(values, dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("observation preprocessing produced non-finite values")
        return result


def load_teacher(entry, project_root, device="cpu"):
    from rsl_rl.modules import ActorCriticRecurrent

    verify_artifacts(entry, project_root)
    payload = torch.load(Path(project_root) / entry["checkpoint"], map_location=device, weights_only=True)
    if int(payload["iter"]) != entry["checkpoint_iteration"]:
        raise ValueError("checkpoint iteration mismatch")
    state = payload["model_state_dict"]
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("non-finite model state")
    cfg = dict(entry["network"])
    if cfg.pop("class_name") != "ActorCriticRecurrent" or cfg.get("rnn_type") != "gru":
        raise ValueError("only the audited GRU teacher architecture is supported")
    obs = TensorDict({g: torch.zeros(1, entry[f"{g}_dim"], device=device)
                      for g in ("policy", "critic")}, batch_size=[1])
    policy = ActorCriticRecurrent(obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]},
                                 num_actions=entry["action_dim"], **cfg).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


class IsaacObservationBridge:
    """Read raw terms using the active Isaac manager, then frozen preprocessing.

    The caller must compare against native observations before stepping. This
    bridge is deterministic and intentionally rejects corruption/history.
    """

    def __init__(self, env, entry):
        self.env, self.entry = env, entry
        self.adapter = ObservationAdapter(entry)
        self.configs = {}
        manager = env.observation_manager
        for group in ("policy", "critic"):
            frozen = entry["observations"][group]
            if list(manager.active_terms[group]) != [t["name"] for t in frozen]:
                raise ValueError(f"runtime observation order drift: {group}")
            group_cfg = getattr(env.cfg.observations, group)
            if group_cfg.enable_corruption or group_cfg.history_length not in (None, 0):
                raise ValueError("acceptance requires clean, non-stacked observations")
            # Isaac Lab 2.3.2 exposes active names publicly but resolved term
            # configs only through this list (also used by its deploy exporter).
            for term, cfg in zip(frozen, manager._group_obs_term_cfgs[group], strict=True):
                function = f"{cfg.func.__module__}:{cfg.func.__name__}"
                if function != term["func"] or cfg.history_length != 0 or cfg.modifiers:
                    raise ValueError(f"runtime observation function/history drift: {term['name']}")
                if (list(cfg.clip) if cfg.clip is not None else None) != term["clip"]:
                    raise ValueError(f"runtime observation clipping drift: {term['name']}")
                expected_scale = 1.0 if term["scale"] is None else term["scale"]
                actual_scale = 1.0 if cfg.scale is None else cfg.scale
                actual_scale = torch.as_tensor(actual_scale, device=env.device, dtype=torch.float32)
                torch.testing.assert_close(actual_scale, actual_scale.new_tensor(expected_scale), atol=0, rtol=0)
                if term["name"] == "gait_clock" and cfg.params != term["params"]:
                    raise ValueError("runtime gait clock parameters drift")
                self.configs[(group, term["name"])] = cfg
        if abs(env.step_dt - entry["control"]["step_dt"]) > 1e-9:
            raise ValueError("control period mismatch")
        robot = env.scene["robot"]
        deploy = entry["control"]
        for name, expected in (("default_joint_pos", deploy["default_joint_pos"]),):
            actual = getattr(robot.data, name)[0]
            torch.testing.assert_close(actual, actual.new_tensor(expected), atol=1e-6, rtol=0)
        from isaaclab.utils.string import resolve_matching_names
        joint_ids, _ = resolve_matching_names(robot.data.joint_names, env.cfg.scene.robot.joint_sdk_names,
                                              preserve_order=True)
        if joint_ids != deploy["joint_ids_map"]:
            raise ValueError("joint order differs from frozen deploy snapshot")
        for name, frozen_action in deploy["actions"].items():
            action = env.action_manager.get_term(name)
            for attr, field in (("_scale", "scale"), ("_offset", "offset"), ("_clip", "clip")):
                actual = torch.as_tensor(getattr(action, attr), device=env.device)
                expected = actual.new_tensor(frozen_action[field])
                if actual.ndim > expected.ndim:
                    actual = actual[0]
                actual = actual.expand_as(expected)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        command_cfg = env.command_manager.get_term("base_velocity").cfg
        for name in ("heading_control_stiffness", "max_yaw_rate", "minimum_turn_radius",
                     "maximum_lateral_acceleration", "minimum_turning_speed"):
            if getattr(command_cfg, name) != entry["command"][name]:
                raise ValueError(f"heading command contract drift: {name}")
        scan = env.scene["height_scanner"].cfg
        for name in ("pos", "rot"):
            if list(getattr(scan.offset, name)) != entry["height_scanner"]["offset"][name]:
                raise ValueError(f"height scan offset drift: {name}")
        if scan.update_period != entry["height_scanner"]["update_period"]:
            raise ValueError("height scan update period drift")
        if scan.ray_alignment != entry["height_scanner"]["ray_alignment"]:
            raise ValueError("height scan alignment drift")
        for name in ("resolution", "size", "direction", "ordering"):
            actual = getattr(scan.pattern_cfg, name)
            if isinstance(actual, tuple):
                actual = list(actual)
            if actual != entry["height_scanner"]["pattern_cfg"][name]:
                raise ValueError(f"height scan pattern drift: {name}")

    def build(self):
        groups = {}
        for group in ("policy", "critic"):
            raw = {}
            for term in self.entry["observations"][group]:
                cfg = self.configs[(group, term["name"])]
                raw[term["name"]] = cfg.func(self.env, **cfg.params)
            groups[group] = self.adapter.build(raw, group)
        return TensorDict(groups, batch_size=[self.env.num_envs])
