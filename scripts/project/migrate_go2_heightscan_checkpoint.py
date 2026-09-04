#!/usr/bin/env python3
"""Create a non-overwriting HeightScan bootstrap checkpoint from a 45-D Go2 policy."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True, help="Existing no-height-scan RSL-RL checkpoint.")
parser.add_argument("--output", required=True, help="New checkpoint path; parent must not contain this file.")
parser.add_argument("--height_scan_dim", type=int, default=117)
parser.add_argument("--optimizer_learning_rate", type=float)
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    help="Zero all Adam moments/steps for a clean optimizer on the new observation space.",
)
args = parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad_input_columns(tensor: torch.Tensor, added_columns: int) -> torch.Tensor:
    padding = torch.zeros(
        (tensor.shape[0], added_columns), dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat((tensor, padding), dim=1)


def main() -> None:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if output.exists():
        parser.error(f"refusing to overwrite output checkpoint: {output}")
    if args.height_scan_dim <= 0:
        parser.error("--height_scan_dim must be positive")
    if args.optimizer_learning_rate is not None and args.optimizer_learning_rate <= 0:
        parser.error("--optimizer_learning_rate must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(output.suffix + ".migration.json")
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    if metadata_path.exists() or checksum_path.exists():
        parser.error(f"refusing existing sidecar for output: {output}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = checkpoint.get("model_state_dict")
    optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(model, dict) or not isinstance(optimizer, dict):
        raise RuntimeError("expected model_state_dict and optimizer_state_dict")

    expected = {"actor.0.weight": (512, 45), "critic.0.weight": (512, 60)}
    for name, shape in expected.items():
        actual = tuple(model[name].shape)
        if actual != shape:
            raise RuntimeError(f"unexpected {name} shape: expected {shape}, got {actual}")

    model["actor.0.weight"] = pad_input_columns(model["actor.0.weight"], args.height_scan_dim)
    model["critic.0.weight"] = pad_input_columns(model["critic.0.weight"], args.height_scan_dim)

    optimizer_expansions = {}
    target_shapes = {(512, 45): "actor.0.weight", (512, 60): "critic.0.weight"}
    for parameter_id, state in optimizer.get("state", {}).items():
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
            continue
        source_shape = tuple(exp_avg.shape)
        if source_shape not in target_shapes:
            continue
        if tuple(exp_avg_sq.shape) != source_shape:
            raise RuntimeError(f"optimizer moment shape mismatch for parameter {parameter_id}")
        state["exp_avg"] = pad_input_columns(exp_avg, args.height_scan_dim)
        state["exp_avg_sq"] = pad_input_columns(exp_avg_sq, args.height_scan_dim)
        optimizer_expansions[str(parameter_id)] = target_shapes[source_shape]
    if set(optimizer_expansions.values()) != set(expected):
        raise RuntimeError(f"could not identify both optimizer input layers: {optimizer_expansions}")

    if args.reset_optimizer:
        for state in optimizer.get("state", {}).values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    value.zero_()
    if args.optimizer_learning_rate is not None:
        for parameter_group in optimizer.get("param_groups", []):
            parameter_group["lr"] = args.optimizer_learning_rate

    torch.save(checkpoint, output)
    output_hash = sha256_file(output)
    metadata = {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "output_checkpoint": str(output),
        "output_checkpoint_sha256": output_hash,
        "height_scan_dim": args.height_scan_dim,
        "actor_input_dim_before": 45,
        "actor_input_dim_after": 45 + args.height_scan_dim,
        "critic_input_dim_before": 60,
        "critic_input_dim_after": 60 + args.height_scan_dim,
        "initialization": "copied original input columns; appended height-scan columns initialized to zero",
        "optimizer_moments": (
            "all moments and steps reset to zero"
            if args.reset_optimizer
            else "copied original moments; appended columns initialized to zero"
        ),
        "optimizer_learning_rate": args.optimizer_learning_rate,
        "source_iteration": checkpoint.get("iter"),
        "optimizer_parameter_expansions": optimizer_expansions,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    checksum_path.write_text(f"{output_hash}  {output.name}\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
