#!/usr/bin/env python3

"""Audit and summarize the formal Go2 rough_stratified evaluation matrix."""

import argparse
import csv
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "evaluation/rough_stratified_v1_2026-08-25"
LABELS = (
    "flat_model2999",
    "rough_model2400",
    "rough_model2900",
    "rough_model2999",
)
EXPECTED_HASHES = {
    "flat_model2999": "a009d982c6889c694bef83e406265749a5dbc7f1b49e9ae7acf4ab56ab340928",
    "rough_model2400": "bef2d39df9222038be615f92877e3c594fd0deacb52848f6938edbeba3fa1701",
    "rough_model2900": "9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889",
    "rough_model2999": "c3d8f8fd5b24aa78b870f1c90c74ca50271d68b13c41b29123261afc0ed95a38",
}
NUMERIC_EPISODE_FIELDS = (
    "episode_steps",
    "episode_seconds",
    "reward_sum",
    "mean_xy_velocity_error_mps",
    "rmse_xy_velocity_error_mps",
    "mean_yaw_velocity_error_radps",
    "rmse_yaw_velocity_error_radps",
    "mean_command_xy_speed_mps",
    "mean_command_abs_yaw_rate_radps",
    "mean_action_delta_l2",
    "mean_mechanical_power_w",
    "mechanical_energy_j",
    "terrain_level",
    "terrain_difficulty_lower",
    "terrain_difficulty_upper",
    "time_out",
    "base_contact",
    "bad_orientation",
)
CSV_HEADER = [
    "checkpoint",
    "subset",
    "evaluation_seeds",
    "episodes",
    "reward_mean",
    "reward_std",
    "xy_error_mean_mps",
    "xy_error_std_mps",
    "xy_rmse_mean_mps",
    "xy_rmse_std_mps",
    "yaw_error_mean_radps",
    "yaw_error_std_radps",
    "action_delta_l2_mean",
    "action_delta_l2_std",
    "mechanical_power_proxy_mean_w",
    "mechanical_power_proxy_std_w",
    "timeout_rate",
    "base_contact_rate",
    "bad_orientation_rate",
]
METRIC_MAPPING = (
    ("reward_mean", "reward_std", "mean_reward_sum"),
    ("xy_error_mean_mps", "xy_error_std_mps", "mean_xy_velocity_error_mps"),
    ("xy_rmse_mean_mps", "xy_rmse_std_mps", "rmse_xy_velocity_error_mps"),
    ("yaw_error_mean_radps", "yaw_error_std_radps", "mean_yaw_velocity_error_radps"),
    ("action_delta_l2_mean", "action_delta_l2_std", "mean_action_delta_l2"),
    (
        "mechanical_power_proxy_mean_w",
        "mechanical_power_proxy_std_w",
        "mean_mechanical_power_w",
    ),
)
RATE_MAPPING = (
    ("timeout_rate", "time_out_rate"),
    ("base_contact_rate", "base_contact_rate"),
    ("bad_orientation_rate", "bad_orientation_rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Formal matrix output root.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[7, 42, 123], help="Evaluation scenario seeds."
    )
    parser.add_argument(
        "--verify-csv",
        type=Path,
        help="Compare the computed run-level aggregate with an existing CSV.",
    )
    parser.add_argument(
        "--write-csv",
        type=Path,
        help="Write the computed aggregate to a new CSV; refuses overwrite.",
    )
    return parser.parse_args()


def aggregate_episode_rows(rows: list[dict]) -> dict:
    def mean(field: str) -> float:
        return statistics.fmean(float(row[field]) for row in rows)

    return {
        "n": len(rows),
        "reward": mean("reward_sum"),
        "xy": mean("mean_xy_velocity_error_mps"),
        "xy_rmse": mean("rmse_xy_velocity_error_mps"),
        "yaw": mean("mean_yaw_velocity_error_radps"),
        "action": mean("mean_action_delta_l2"),
        "power": mean("mean_mechanical_power_w"),
        "timeout": mean("time_out"),
        "base": mean("base_contact"),
        "bad": mean("bad_orientation"),
    }


def build_aggregate_csv(summaries: dict[str, list[dict]], seeds: list[int]) -> str:
    rows = []
    for label in LABELS:
        for subset in ("overall", "flat", "rough"):
            run_values = []
            for summary in summaries[label]:
                if subset == "overall":
                    values = summary
                else:
                    terrain_name = "flat" if subset == "flat" else "random_rough"
                    values = summary["terrain_metrics"]["by_type"][terrain_name]
                run_values.append(values)

            row = {
                "checkpoint": label,
                "subset": subset,
                "evaluation_seeds": ";".join(str(seed) for seed in seeds),
                "episodes": len(seeds) * (128 if subset == "overall" else 64),
            }
            for mean_name, std_name, source_name in METRIC_MAPPING:
                metric_values = [float(run[source_name]) for run in run_values]
                row[mean_name] = f"{statistics.fmean(metric_values):.6f}"
                row[std_name] = f"{statistics.stdev(metric_values):.6f}"
            for output_name, source_name in RATE_MAPPING:
                row[output_name] = (
                    f"{statistics.fmean(float(run[source_name]) for run in run_values):.6f}"
                )
            rows.append(row)

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_HEADER)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    seeds = args.seeds
    if args.verify_csv and args.write_csv:
        raise ValueError("--verify-csv and --write-csv are mutually exclusive")
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for sample standard deviation")
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {root}")

    rows_by_label: dict[str, list[dict]] = defaultdict(list)
    summaries_by_label: dict[str, list[dict]] = defaultdict(list)
    failures = []

    print("=== PER-RUN ARTIFACT AUDIT ===")
    for label in LABELS:
        for seed in seeds:
            run_name = f"{label}_seed{seed}_env64_ep2"
            run_dir = root / run_name
            console_path = root / f"{run_name}.console.log"
            required = (
                run_dir / "metadata.json",
                run_dir / "summary.json",
                run_dir / "episodes.csv",
                console_path,
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                failures.append((run_name, "missing", missing))
                print(f"FAIL {run_name} missing={missing}")
                continue

            metadata = json.loads((run_dir / "metadata.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            with (run_dir / "episodes.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))

            problems = []
            if metadata["checkpoint_sha256"] != EXPECTED_HASHES[label]:
                problems.append("checkpoint_hash")
            if metadata["terrain_protocol"] != "rough_stratified":
                problems.append("terrain_protocol")
            if metadata["num_envs"] != 64 or metadata["episodes_per_env"] != 2:
                problems.append("evaluation_shape")
            if summary["completed_episodes"] != 128 or len(rows) != 128:
                problems.append("episode_count")

            type_counts = Counter(row["terrain_type"] for row in rows)
            if type_counts != {"flat": 64, "random_rough": 64}:
                problems.append(f"type_counts={dict(type_counts)}")
            for terrain_type in ("flat", "random_rough"):
                levels = {
                    int(row["terrain_level"])
                    for row in rows
                    if row["terrain_type"] == terrain_type
                }
                if levels != set(range(10)):
                    problems.append(f"{terrain_type}_levels={sorted(levels)}")

            episode_keys = {
                (int(row["env_id"]), int(row["episode_index"])) for row in rows
            }
            if len(episode_keys) != 128:
                problems.append("duplicate_episode_keys")
            env_assignments: dict[int, set[tuple]] = defaultdict(set)
            for row in rows:
                env_assignments[int(row["env_id"])].add(
                    (row["terrain_type"], int(row["terrain_column"]), int(row["terrain_level"]))
                )
            if len(env_assignments) != 64 or any(
                len(values) != 1 for values in env_assignments.values()
            ):
                problems.append("terrain_assignment_changed")

            nonfinite = []
            for row_number, row in enumerate(rows, start=2):
                for field in NUMERIC_EPISODE_FIELDS:
                    try:
                        value = float(row[field])
                    except (KeyError, TypeError, ValueError):
                        nonfinite.append((row_number, field, row.get(field)))
                        continue
                    if not math.isfinite(value):
                        nonfinite.append((row_number, field, row[field]))
            if nonfinite:
                problems.append(f"nonfinite={nonfinite[:5]}")

            console_text = console_path.read_text(errors="replace")
            suspicious = re.findall(
                r"(?i)traceback|exception|failed|"
                r"(?<![A-Za-z0-9_])(?:nan|inf)(?![A-Za-z0-9_])",
                console_text,
            )
            if suspicious:
                problems.append(f"console_suspicious={len(suspicious)}")

            terminations = {
                name: sum(int(row[name]) for row in rows)
                for name in ("time_out", "base_contact", "bad_orientation")
            }
            status = "PASS" if not problems else "FAIL"
            print(
                f"{status} {run_name} episodes={len(rows)}"
                f" flat={type_counts['flat']} rough={type_counts['random_rough']}"
                f" timeout={terminations['time_out']}"
                f" base_contact={terminations['base_contact']}"
                f" bad_orientation={terminations['bad_orientation']}"
                f" nonfinite={len(nonfinite)}"
            )
            if problems:
                failures.append((run_name, "validation", problems))

            rows_by_label[label].extend(rows)
            summaries_by_label[label].append(summary)

    print("\n=== EPISODE-WEIGHTED AGGREGATES ===")
    print(
        "model subset n reward xy xy_rmse yaw action_delta power "
        "timeout base_contact bad_orientation"
    )
    for label in LABELS:
        all_rows = rows_by_label[label]
        subsets = {
            "overall": all_rows,
            "flat": [row for row in all_rows if row["terrain_type"] == "flat"],
            "rough": [row for row in all_rows if row["terrain_type"] == "random_rough"],
        }
        for subset_name, subset_rows in subsets.items():
            value = aggregate_episode_rows(subset_rows)
            print(
                f"{label:17s} {subset_name:7s} {value['n']:3d}"
                f" {value['reward']:.4f} {value['xy']:.6f} {value['xy_rmse']:.6f}"
                f" {value['yaw']:.6f} {value['action']:.6f} {value['power']:.3f}"
                f" {value['timeout']:.6f} {value['base']:.6f} {value['bad']:.6f}"
            )

    print("\n=== RUN-LEVEL MEAN +/- SAMPLE SD ===")
    for label in LABELS:
        summaries = summaries_by_label[label]
        overall_xy = [summary["mean_xy_velocity_error_mps"] for summary in summaries]
        rough_xy = [
            summary["terrain_metrics"]["by_type"]["random_rough"][
                "mean_xy_velocity_error_mps"
            ]
            for summary in summaries
        ]
        timeout = [summary["time_out_rate"] for summary in summaries]
        rough_timeout = [
            summary["terrain_metrics"]["by_type"]["random_rough"]["time_out_rate"]
            for summary in summaries
        ]
        print(
            f"{label:17s}"
            f" overall_xy={statistics.fmean(overall_xy):.6f}+/-{statistics.stdev(overall_xy):.6f}"
            f" rough_xy={statistics.fmean(rough_xy):.6f}+/-{statistics.stdev(rough_xy):.6f}"
            f" timeout={statistics.fmean(timeout):.6f}+/-{statistics.stdev(timeout):.6f}"
            f" rough_timeout={statistics.fmean(rough_timeout):.6f}+/-{statistics.stdev(rough_timeout):.6f}"
        )

    print("\n=== RANDOM_ROUGH LEVEL BREAKDOWN ===")
    print("model level n xy timeout bad_orientation")
    for label in LABELS:
        rough_rows = [
            row for row in rows_by_label[label] if row["terrain_type"] == "random_rough"
        ]
        for level in range(10):
            level_rows = [row for row in rough_rows if int(row["terrain_level"]) == level]
            value = aggregate_episode_rows(level_rows)
            print(
                f"{label:17s} {level:2d} {value['n']:2d} {value['xy']:.6f}"
                f" {value['timeout']:.6f} {value['bad']:.6f}"
            )

    aggregate_csv = build_aggregate_csv(summaries_by_label, seeds)
    if args.verify_csv:
        verify_path = args.verify_csv.expanduser().resolve()
        if not verify_path.is_file():
            raise FileNotFoundError(f"Aggregate CSV not found: {verify_path}")
        if verify_path.read_text().splitlines() != aggregate_csv.splitlines():
            failures.append((str(verify_path), "aggregate_csv_mismatch", []))
        else:
            print(f"\nAGGREGATE_CSV=PASS {verify_path}")
    if args.write_csv:
        output_path = args.write_csv.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite aggregate CSV: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(aggregate_csv)
        print(f"\nAGGREGATE_CSV_WRITTEN={output_path}")

    if failures:
        print("\nAUDIT_RESULT=FAIL")
        for failure in failures:
            print(failure)
        return 1

    expected_runs = len(LABELS) * len(seeds)
    print("\nAUDIT_RESULT=PASS")
    print(f"run_count={expected_runs}")
    print(f"episode_count={expected_runs * 128}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
