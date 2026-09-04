#!/usr/bin/env python3
"""Summarize enhanced fixed-terrain Go2 diagnostic recordings without simulation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input_root", required=True, help="Diagnostic root containing per-clip metadata.json files.")
args = parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def first_true_time(rows: list[dict[str, str]], field: str) -> float | None:
    for row in rows:
        if int(row[field]):
            return float(row["time_seconds"])
    return None


def classify(metadata: dict, rows: list[dict[str, str]], steady: list[dict[str, str]]) -> str:
    terminations = metadata["terminations"]
    if terminations.get("base_contact", 0):
        return "base_contact"
    if terminations.get("bad_orientation", 0):
        return "bad_orientation"
    stalled_fraction = mean([float(row["stalled"]) for row in steady])
    if stalled_fraction >= 0.5:
        return "stalled"
    command = abs(float(metadata["command"]["lin_vel_x"]))
    actual = abs(mean([float(row["actual_x_mps"]) for row in steady]))
    if command >= 0.1 and actual < command * 0.5:
        return "slow_progress"
    if terminations.get("time_out", 0):
        return "effective_timeout"
    return "completed_clip"


def main() -> None:
    root = Path(args.input_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"input root is not a directory: {root}")
    metadata_paths = sorted(root.glob("*/metadata.json"))
    if not metadata_paths:
        parser.error(f"no per-clip metadata.json files found under {root}")

    summaries = []
    checkpoint_hashes = set()
    scenario_seeds = set()
    training_seeds = set()
    nonfinite_fields = []
    required_columns = {
        "base_pos_x_m",
        "base_pos_y_m",
        "base_pos_z_m",
        "base_roll_rad",
        "base_pitch_rad",
        "forward_displacement_m",
        "command_signed_progress_m",
        "base_contact_force_n",
        "stalled",
    }

    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        csv_path = Path(metadata["tracking_timeseries"])
        if not csv_path.is_absolute():
            csv_path = metadata_path.parent / csv_path
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        if sha256_file(csv_path) != metadata["tracking_timeseries_sha256"]:
            raise RuntimeError(f"tracking CSV hash mismatch: {csv_path}")
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        missing = required_columns - set(rows[0])
        if missing:
            raise RuntimeError(f"legacy/incomplete diagnostic CSV {csv_path}; missing {sorted(missing)}")
        if len(rows) != int(metadata["video_steps"]):
            raise RuntimeError(f"row count mismatch for {csv_path}: {len(rows)}")

        warmup = int(metadata["warmup_steps"])
        steady = rows[warmup:]
        numeric_fields = [
            "actual_x_mps",
            "xy_error_mps",
            "base_roll_rad",
            "base_pitch_rad",
            "forward_displacement_m",
            "command_signed_progress_m",
            "base_contact_force_n",
        ]
        for index, row in enumerate(rows):
            for field in numeric_fields:
                if not math.isfinite(float(row[field])):
                    nonfinite_fields.append(f"{csv_path}:{index + 2}:{field}")

        checkpoint_hashes.add(metadata["checkpoint_sha256"])
        scenario_seeds.add(int(metadata["evaluation_scenario_seed"]))
        training_seeds.add(int(metadata["training_seed"]))
        command_x = float(metadata["command"]["lin_vel_x"])
        direction = "positive" if command_x >= 0 else "negative"
        first_done_index = next((index for index, row in enumerate(rows) if int(row["done"])), len(rows) - 1)
        first_episode_rows = rows[: first_done_index + 1]
        forward_displacement = float(rows[-1]["forward_displacement_m"]) - float(
            rows[0]["forward_displacement_m"]
        )
        command_signed_progress = forward_displacement * (1.0 if command_x >= 0 else -1.0)
        first_episode_progress = (
            float(first_episode_rows[-1]["forward_displacement_m"])
            - float(first_episode_rows[0]["forward_displacement_m"])
        ) * (1.0 if command_x >= 0 else -1.0)
        first_termination = "none"
        for name in ("bad_orientation", "base_contact", "time_out"):
            if int(first_episode_rows[-1][f"termination_{name}"]):
                first_termination = name
                break
        summary = {
            "terrain_type": metadata["terrain_type"],
            "terrain_level": int(metadata["terrain_level"]),
            "direction": direction,
            "command_x_mps": command_x,
            "post_warmup_actual_x_mps": mean([float(row["actual_x_mps"]) for row in steady]),
            "post_warmup_xy_error_mps": mean([float(row["xy_error_mps"]) for row in steady]),
            "final_forward_displacement_m": forward_displacement,
            "final_command_signed_progress_m": command_signed_progress,
            "first_episode_command_signed_progress_m": first_episode_progress,
            "first_episode_end_time_s": float(first_episode_rows[-1]["time_seconds"]),
            "first_episode_termination": first_termination,
            "stalled_fraction": mean([float(row["stalled"]) for row in steady]),
            "first_stalled_time_s": first_true_time(rows, "stalled"),
            "max_abs_roll_rad": max(abs(float(row["base_roll_rad"])) for row in rows),
            "max_abs_pitch_rad": max(abs(float(row["base_pitch_rad"])) for row in rows),
            "max_base_contact_force_n": max(float(row["base_contact_force_n"]) for row in rows),
            "termination_time_out": int(metadata["terminations"].get("time_out", 0)),
            "termination_base_contact": int(metadata["terminations"].get("base_contact", 0)),
            "termination_bad_orientation": int(metadata["terminations"].get("bad_orientation", 0)),
            "classification": classify(metadata, rows, steady),
            "clip_directory": str(metadata_path.parent),
        }
        summaries.append(summary)

    if nonfinite_fields:
        raise RuntimeError("non-finite diagnostic values: " + ", ".join(nonfinite_fields[:20]))
    if len(checkpoint_hashes) != 1 or len(scenario_seeds) != 1 or len(training_seeds) != 1:
        raise RuntimeError(
            "mixed diagnostic identity: "
            f"checkpoint_hashes={checkpoint_hashes}, training_seeds={training_seeds}, "
            f"evaluation_scenario_seeds={scenario_seeds}"
        )

    summaries.sort(key=lambda row: (row["terrain_type"], row["direction"]))
    csv_output = root / "diagnostic_summary.csv"
    json_output = root / "diagnostic_summary.json"
    with csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        "input_root": str(root),
        "training_seed": next(iter(training_seeds)),
        "evaluation_scenario_seed": next(iter(scenario_seeds)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "clip_count": len(summaries),
        "nonfinite_value_count": 0,
        "clips": summaries,
        "summary_csv": str(csv_output),
    }
    json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
