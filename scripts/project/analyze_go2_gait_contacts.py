#!/usr/bin/env python3
"""Analyze per-foot gait/contact evidence from an existing recorder CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FEET = ("fl_foot", "fr_foot", "rl_foot", "rr_foot")


def longest_run(values: list[bool], target: bool) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value is target else 0
        best = max(best, current)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracking_csv", type=Path)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Discard this many initial samples before analysis."
    )
    args = parser.parse_args()
    with args.tracking_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("tracking CSV must contain at least two rows")
    if not 0 <= args.warmup_steps < len(rows) - 1:
        raise ValueError("--warmup_steps must leave at least two samples")
    rows = rows[args.warmup_steps :]
    dt = float(rows[1]["time_seconds"]) - float(rows[0]["time_seconds"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"invalid step duration: {dt}")

    contacts: dict[str, list[bool]] = {}
    for foot in FEET:
        key = f"{foot}_contact"
        if key not in rows[0]:
            raise KeyError(f"missing {key}; available={sorted(rows[0])}")
        contacts[foot] = [bool(int(float(row[key]))) for row in rows]

    per_foot = {}
    for foot, values in contacts.items():
        per_foot[foot] = {
            "contact_duty_factor": sum(values) / len(values),
            "maximum_continuous_air_time_seconds": longest_run(values, False) * dt,
            "maximum_continuous_contact_time_seconds": longest_run(values, True) * dt,
        }
    contact_counts = [sum(contacts[foot][index] for foot in FEET) for index in range(len(rows))]
    diagonal_mismatch = []
    for index in range(len(rows)):
        fl, fr, rl, rr = (contacts[foot][index] for foot in FEET)
        diagonal_mismatch.append(int(fl != rr) + int(fr != rl) + int(fl == fr))
    numeric_fields = (
        "xy_error_mps",
        "yaw_error_radps",
        "base_pos_x_m",
        "base_pos_y_m",
        "base_pos_z_m",
        "command_signed_progress_m",
        "cumulative_xy_distance_m",
    )
    nonfinite_values = sum(
        not math.isfinite(float(row[field])) for row in rows for field in numeric_fields
    )
    progress_start = float(rows[0]["command_signed_progress_m"])
    progress_end = float(rows[-1]["command_signed_progress_m"])
    path_start = float(rows[0]["cumulative_xy_distance_m"])
    path_end = float(rows[-1]["cumulative_xy_distance_m"])
    net_progress = progress_end - progress_start
    path_length = max(0.0, path_end - path_start)
    z_values = [float(row["base_pos_z_m"]) for row in rows]
    result = {
        "tracking_csv": str(args.tracking_csv.resolve()),
        "warmup_steps_discarded": args.warmup_steps,
        "rows": len(rows),
        "step_dt_seconds": dt,
        "per_foot": per_foot,
        "contact_count_fractions": {
            str(count): sum(value == count for value in contact_counts) / len(contact_counts)
            for count in range(5)
        },
        "all_feet_airborne_fraction": sum(value == 0 for value in contact_counts) / len(contact_counts),
        "mean_diagonal_trot_contact_mismatch": sum(diagonal_mismatch) / len(diagonal_mismatch),
        "duty_factor_spread": max(v["contact_duty_factor"] for v in per_foot.values())
        - min(v["contact_duty_factor"] for v in per_foot.values()),
        "motion": {
            "mean_xy_velocity_error_mps": sum(float(row["xy_error_mps"]) for row in rows) / len(rows),
            "mean_yaw_velocity_error_radps": sum(float(row["yaw_error_radps"]) for row in rows)
            / len(rows),
            "net_command_signed_progress_m": net_progress,
            "planar_path_length_m": path_length,
            "command_progress_path_efficiency": net_progress / max(path_length, 1.0e-6),
            "net_base_elevation_change_m": z_values[-1] - z_values[0],
            "base_elevation_range_m": max(z_values) - min(z_values),
            "stall_fraction": sum(bool(int(float(row["stalled"]))) for row in rows) / len(rows),
        },
        "safety": {
            "base_contact_fraction": sum(bool(int(float(row["base_contact"]))) for row in rows)
            / len(rows),
            "done_count": sum(bool(int(float(row["done"]))) for row in rows),
            "time_out_count": sum(bool(int(float(row["termination_time_out"]))) for row in rows),
            "base_contact_termination_count": sum(
                bool(int(float(row["termination_base_contact"]))) for row in rows
            ),
            "bad_orientation_termination_count": sum(
                bool(int(float(row["termination_bad_orientation"]))) for row in rows
            ),
        },
        "nonfinite_numeric_value_count": nonfinite_values,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError(args.output_json)
        args.output_json.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
