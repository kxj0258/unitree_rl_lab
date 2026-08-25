#!/usr/bin/env python3

"""Audit and aggregate the formal multi-training-seed Go2 rough matrices."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SEEDS = (42, 7, 123)
EVALUATION_SEEDS = (7, 42, 123)
MODELS = (2400, 2900, 2999)
TRAINING_RUNS = {
    42: REPO_ROOT / "logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000",
    7: REPO_ROOT / "logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_13-53-47_roughmix50_curriculum_formal_seed7_env2048_i3000",
    123: REPO_ROOT / "logs/rsl_rl/unitree_go2_velocity_rough/2026-08-25_17-23-39_roughmix50_curriculum_formal_seed123_env2048_i3000",
}
EVALUATION_ROOTS = {
    42: REPO_ROOT / "evaluation/rough_stratified_v1_2026-08-25",
    7: REPO_ROOT / "evaluation/rough_stratified_v1_trainseed7_2026-08-25",
    123: REPO_ROOT / "evaluation/rough_stratified_v1_trainseed123_2026-08-25",
}
CHECKPOINT_HASHES = {
    42: {
        2400: "bef2d39df9222038be615f92877e3c594fd0deacb52848f6938edbeba3fa1701",
        2900: "9afefda4777569ed120419a83c40d2a487cdc4be862ac16c7bd9ab6fb45c2889",
        2999: "c3d8f8fd5b24aa78b870f1c90c74ca50271d68b13c41b29123261afc0ed95a38",
    },
    7: {
        2400: "3a04a27fd4ebd50643a6cf894ce5a61e384091f76559bf910290ac94a65517a2",
        2900: "d31467c4a0feee120336d3e52b77154e3d96de654891e780601be8ccde2a4c5a",
        2999: "353ea7a9d1cfa3e35fdc8678901ede952c54e7f4fb54782ad86fd2f02352f4af",
    },
    123: {
        2400: "fd6289ba4c35e0ebb8af6167a130f8bc2d4b1b8504865ddfdae8b2f5218d6c91",
        2900: "26649d0c4e7edf283e76740e22b4b0e9a7bfa9abf2d23b24a89333cf363dee1d",
        2999: "d83a237bbf9918b3f81d269b1ce4684659756ee30c902b384f92d9dfced0945c",
    },
}
OFFICIAL_MODELS = {42: 2900, 7: 2999, 123: 2999}
METRICS = {
    "reward": "reward_sum",
    "xy_error_mps": "mean_xy_velocity_error_mps",
    "yaw_error_radps": "mean_yaw_velocity_error_radps",
    "action_delta_l2": "mean_action_delta_l2",
    "power_proxy_w": "mean_mechanical_power_w",
}
FATAL_PATTERN = re.compile(
    r"(^|[^A-Za-z0-9_])(nan|inf)([^A-Za-z0-9_]|$)|traceback|runtimeerror|"
    r"exception|segmentation fault|cuda error|out of memory|assertionerror|failed",
    re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-selected-csv", type=Path)
    parser.add_argument("--verify-fixed-csv", type=Path)
    parser.add_argument("--write-selected-csv", type=Path)
    parser.add_argument("--write-fixed-csv", type=Path)
    return parser.parse_args()


def run_name(training_seed: int, model: int, evaluation_seed: int) -> str:
    if training_seed == 42:
        return f"rough_model{model}_seed{evaluation_seed}_env64_ep2"
    return (
        f"rough_trainseed{training_seed}_model{model}_"
        f"evalseed{evaluation_seed}_env64_ep2"
    )


def mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subsets(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "overall": rows,
        "flat": [row for row in rows if row["terrain_type"] == "flat"],
        "rough": [row for row in rows if row["terrain_type"] == "random_rough"],
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    result = {name: mean(rows, field) for name, field in METRICS.items()}
    result.update(
        timeout_rate=mean(rows, "time_out"),
        base_contact_rate=mean(rows, "base_contact"),
        bad_orientation_rate=mean(rows, "bad_orientation"),
    )
    return result


def collect() -> tuple[dict[tuple[int, int], list[dict]], list[str]]:
    rows_by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    failures: list[str] = []
    print("=== MULTI-TRAINING-SEED PER-RUN AUDIT ===")
    for training_seed in TRAINING_SEEDS:
        for model in MODELS:
            checkpoint = TRAINING_RUNS[training_seed] / f"model_{model}.pt"
            if not checkpoint.is_file():
                failures.append(f"missing checkpoint: {checkpoint}")
            elif sha256(checkpoint) != CHECKPOINT_HASHES[training_seed][model]:
                failures.append(f"checkpoint hash mismatch: {checkpoint}")
            for evaluation_seed in EVALUATION_SEEDS:
                name = run_name(training_seed, model, evaluation_seed)
                root = EVALUATION_ROOTS[training_seed]
                run_dir = root / name
                console_path = root / f"{name}.console.log"
                required = (
                    run_dir / "metadata.json",
                    run_dir / "summary.json",
                    run_dir / "episodes.csv",
                    console_path,
                )
                missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
                if missing:
                    failures.append(f"{name}: missing_or_empty={missing}")
                    print(f"FAIL {name} missing_or_empty={missing}")
                    continue

                metadata = json.loads((run_dir / "metadata.json").read_text())
                summary = json.loads((run_dir / "summary.json").read_text())
                with (run_dir / "episodes.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                problems = []
                if metadata["checkpoint_sha256"] != CHECKPOINT_HASHES[training_seed][model]:
                    problems.append("checkpoint_hash")
                if metadata["seed"] != evaluation_seed:
                    problems.append("evaluation_seed")
                if metadata["terrain_protocol"] != "rough_stratified":
                    problems.append("terrain_protocol")
                if metadata["num_envs"] != 64 or metadata["episodes_per_env"] != 2:
                    problems.append("evaluation_shape")
                if metadata["curriculum_updates_enabled"] is not False:
                    problems.append("curriculum_updates")
                if summary["completed_episodes"] != 128 or len(rows) != 128:
                    problems.append("episode_count")

                counts = Counter(row["terrain_type"] for row in rows)
                if counts != {"flat": 64, "random_rough": 64}:
                    problems.append(f"terrain_counts={dict(counts)}")
                for terrain_type in ("flat", "random_rough"):
                    levels = {int(row["terrain_level"]) for row in rows if row["terrain_type"] == terrain_type}
                    if levels != set(range(10)):
                        problems.append(f"{terrain_type}_levels={sorted(levels)}")

                keys = {(int(row["env_id"]), int(row["episode_index"])) for row in rows}
                if len(keys) != 128:
                    problems.append("episode_keys")
                assignments: dict[int, set[tuple]] = defaultdict(set)
                for row in rows:
                    env_id = int(row["env_id"])
                    assignments[env_id].add(
                        (row["terrain_type"], int(row["terrain_column"]), int(row["terrain_level"]))
                    )
                    expected_type = "flat" if env_id < 32 else "random_rough"
                    expected_level = ((env_id % 32) + evaluation_seed) % 10
                    if row["terrain_type"] != expected_type or int(row["terrain_level"]) != expected_level:
                        problems.append(f"assignment_env_{env_id}")
                    for field in (*METRICS.values(), "time_out", "base_contact", "bad_orientation"):
                        try:
                            value = float(row[field])
                        except (KeyError, TypeError, ValueError):
                            problems.append(f"invalid_{field}")
                            continue
                        if not math.isfinite(value):
                            problems.append(f"nonfinite_{field}")
                    if sum(int(row[field]) for field in ("time_out", "base_contact", "bad_orientation")) != 1:
                        problems.append("termination_sum")
                if len(assignments) != 64 or any(len(value) != 1 for value in assignments.values()):
                    problems.append("terrain_assignment_changed")
                if FATAL_PATTERN.search(console_path.read_text(errors="replace")):
                    problems.append("console_suspicious")

                if problems:
                    unique = sorted(set(problems))
                    failures.append(f"{name}: {unique}")
                    print(f"FAIL {name} problems={unique}")
                else:
                    print(f"PASS {name} episodes=128 flat=64 rough=64 levels=0-9 nonfinite=0")
                    for row in rows:
                        row["_evaluation_seed"] = evaluation_seed
                        rows_by_key[(training_seed, model)].append(row)
    return rows_by_key, failures


def csv_text(header: list[str], rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().replace("\r\n", "\n")


def build_selected_csv(rows_by_key: dict[tuple[int, int], list[dict]]) -> str:
    header = [
        "scope", "training_seed", "checkpoint", "subset", "training_seeds",
        "evaluation_seeds", "episodes",
        *[item for name in METRICS for item in (f"{name}_mean", f"{name}_sd")],
        "timeout_rate", "base_contact_rate", "bad_orientation_rate",
    ]
    output = []
    per_seed: dict[tuple[int, str], dict[str, float]] = {}
    for training_seed in TRAINING_SEEDS:
        model = OFFICIAL_MODELS[training_seed]
        for subset_name, subset_rows in subsets(rows_by_key[(training_seed, model)]).items():
            row = {
                "scope": "training_seed", "training_seed": training_seed,
                "checkpoint": f"model_{model}", "subset": subset_name,
                "training_seeds": str(training_seed),
                "evaluation_seeds": ";".join(map(str, EVALUATION_SEEDS)),
                "episodes": len(subset_rows),
            }
            value = aggregate(subset_rows)
            per_seed[(training_seed, subset_name)] = value
            for name, field in METRICS.items():
                seed_values = [
                    mean([item for item in subset_rows if item["_evaluation_seed"] == seed], field)
                    for seed in EVALUATION_SEEDS
                ]
                row[f"{name}_mean"] = f"{value[name]:.6f}"
                row[f"{name}_sd"] = f"{statistics.stdev(seed_values):.6f}"
            for rate in ("timeout_rate", "base_contact_rate", "bad_orientation_rate"):
                row[rate] = f"{value[rate]:.6f}"
            output.append(row)

    for subset_name in ("overall", "flat", "rough"):
        row = {
            "scope": "multitraining_mean", "training_seed": "",
            "checkpoint": "per_seed_selected", "subset": subset_name,
            "training_seeds": ";".join(map(str, TRAINING_SEEDS)),
            "evaluation_seeds": ";".join(map(str, EVALUATION_SEEDS)),
            "episodes": sum(len(subsets(rows_by_key[(seed, OFFICIAL_MODELS[seed])])[subset_name]) for seed in TRAINING_SEEDS),
        }
        for name in METRICS:
            values = [per_seed[(seed, subset_name)][name] for seed in TRAINING_SEEDS]
            row[f"{name}_mean"] = f"{statistics.fmean(values):.6f}"
            row[f"{name}_sd"] = f"{statistics.stdev(values):.6f}"
        for rate in ("timeout_rate", "base_contact_rate", "bad_orientation_rate"):
            values = [per_seed[(seed, subset_name)][rate] for seed in TRAINING_SEEDS]
            row[rate] = f"{statistics.fmean(values):.6f}"
        output.append(row)
    return csv_text(header, output)


def build_fixed_csv(rows_by_key: dict[tuple[int, int], list[dict]]) -> str:
    header = [
        "checkpoint", "subset", "training_seeds", "evaluation_seeds", "episodes",
        *[item for name in METRICS for item in (f"{name}_mean", f"{name}_training_seed_sd")],
        "timeout_rate", "timeout_training_seed_sd", "base_contact_rate",
        "base_contact_training_seed_sd", "bad_orientation_rate",
        "bad_orientation_training_seed_sd",
    ]
    output = []
    for model in MODELS:
        for subset_name in ("overall", "flat", "rough"):
            values_by_seed = [
                aggregate(subsets(rows_by_key[(seed, model)])[subset_name])
                for seed in TRAINING_SEEDS
            ]
            row = {
                "checkpoint": f"model_{model}", "subset": subset_name,
                "training_seeds": ";".join(map(str, TRAINING_SEEDS)),
                "evaluation_seeds": ";".join(map(str, EVALUATION_SEEDS)),
                "episodes": sum(len(subsets(rows_by_key[(seed, model)])[subset_name]) for seed in TRAINING_SEEDS),
            }
            for name in METRICS:
                values = [item[name] for item in values_by_seed]
                row[f"{name}_mean"] = f"{statistics.fmean(values):.6f}"
                row[f"{name}_training_seed_sd"] = f"{statistics.stdev(values):.6f}"
            for rate in ("timeout_rate", "base_contact_rate", "bad_orientation_rate"):
                values = [item[rate] for item in values_by_seed]
                row[rate] = f"{statistics.fmean(values):.6f}"
                row[f"{rate.removesuffix('_rate')}_training_seed_sd"] = f"{statistics.stdev(values):.6f}"
            output.append(row)
    return csv_text(header, output)


def verify_or_write(text: str, verify: Path | None, write: Path | None, label: str) -> None:
    if verify and write:
        raise ValueError(f"{label}: verify and write are mutually exclusive")
    if verify:
        path = verify.expanduser().resolve()
        if path.read_text().splitlines() != text.splitlines():
            raise RuntimeError(f"{label} CSV mismatch: {path}")
        print(f"{label.upper()}_CSV=PASS {path}")
    if write:
        path = write.expanduser().resolve()
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"{label.upper()}_CSV_WRITTEN={path}")


def main() -> int:
    args = parse_args()
    rows_by_key, failures = collect()
    selected_csv = build_selected_csv(rows_by_key) if not failures else ""
    fixed_csv = build_fixed_csv(rows_by_key) if not failures else ""
    if failures:
        print("AUDIT_RESULT=FAIL")
        for failure in failures:
            print(failure)
        return 1
    verify_or_write(selected_csv, args.verify_selected_csv, args.write_selected_csv, "selected")
    verify_or_write(fixed_csv, args.verify_fixed_csv, args.write_fixed_csv, "fixed")
    print("AUDIT_RESULT=PASS")
    print("run_count=27")
    print("episode_count=3456")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
