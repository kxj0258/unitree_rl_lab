#!/usr/bin/env python3

"""Create or verify the formal Go2 rough reproducibility manifests."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from audit_go2_rough_evaluation import EXPECTED_HASHES, LABELS, build_aggregate_csv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = REPO_ROOT / "evaluation/rough_stratified_v1_2026-08-25"
DEFAULT_TRAIN_ROOT = (
    REPO_ROOT
    / "logs/rsl_rl/unitree_go2_velocity_rough"
    / "2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "doc/repro"
SEEDS = (7, 42, 123)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true", help="Create all six files; refuse overwrite.")
    mode.add_argument("--verify", action="store_true", help="Verify all six existing files byte-for-byte.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def display_path(path: Path) -> str:
    try:
        return relative(path)
    except ValueError:
        return str(path)


def checkpoint_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def build_contents(eval_root: Path, train_root: Path, output_dir: Path) -> dict[Path, str]:
    official_checkpoint = train_root / "model_2900.pt"
    if not official_checkpoint.is_file():
        raise FileNotFoundError(f"Official checkpoint not found: {official_checkpoint}")
    if sha256(official_checkpoint) != EXPECTED_HASHES["rough_model2900"]:
        raise RuntimeError("Official rough checkpoint hash mismatch")

    checkpoints = sorted(train_root.glob("model_*.pt"), key=checkpoint_number)
    if len(checkpoints) != 31:
        raise RuntimeError(f"Expected 31 rough checkpoints, found {len(checkpoints)}")
    checkpoint_inventory = "".join(
        f"{path.name} {path.stat().st_size} bytes\n" for path in checkpoints
    )

    all_eval_files = sorted(path for path in eval_root.rglob("*") if path.is_file())
    summary_paths = sorted(eval_root.glob("*_env64_ep2/summary.json"))
    if len(all_eval_files) != 48:
        raise RuntimeError(f"Expected 48 evaluation files, found {len(all_eval_files)}")
    if len(summary_paths) != 12:
        raise RuntimeError(f"Expected 12 summary files, found {len(summary_paths)}")

    summaries: dict[str, list[dict]] = {label: [] for label in LABELS}
    episode_rows: dict[str, list[dict]] = {label: [] for label in LABELS}
    for label in LABELS:
        for seed in SEEDS:
            run_name = f"{label}_seed{seed}_env64_ep2"
            run_dir = eval_root / run_name
            metadata = json.loads((run_dir / "metadata.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            with (run_dir / "episodes.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))

            if metadata["checkpoint_sha256"] != EXPECTED_HASHES[label]:
                raise RuntimeError(f"Checkpoint hash mismatch in {run_name}")
            if metadata["terrain_protocol"] != "rough_stratified":
                raise RuntimeError(f"Terrain protocol mismatch in {run_name}")
            if summary["completed_episodes"] != 128 or len(rows) != 128:
                raise RuntimeError(f"Episode count mismatch in {run_name}")
            summaries[label].append(summary)
            episode_rows[label].extend(rows)

    aggregate_csv = build_aggregate_csv(summaries, list(SEEDS))

    official_rows = episode_rows["rough_model2900"]
    flat_rows = [row for row in official_rows if row["terrain_type"] == "flat"]
    rough_rows = [row for row in official_rows if row["terrain_type"] == "random_rough"]
    if len(official_rows) != 384 or len(flat_rows) != 192 or len(rough_rows) != 192:
        raise RuntimeError("Official checkpoint evaluation coverage mismatch")

    def episode_mean(field: str, rows: list[dict] = official_rows) -> float:
        return statistics.fmean(float(row[field]) for row in rows)

    termination_counts = {
        field: sum(int(row[field]) for row in official_rows)
        for field in ("time_out", "base_contact", "bad_orientation")
    }
    selection_protocol = (
        "flat model_2999 control and rough model_2400/model_2900/model_2999 evaluated "
        "with scenario seeds 7, 42, and 123; 64 environments and 2 episodes per "
        "environment per run."
    )
    terrain_protocol = (
        "rough_stratified v1; exact 50/50 flat/random_rough assignment; both terrain "
        "types cover levels 0-9; terrain and command curriculum updates disabled during "
        "evaluation."
    )
    selection_reason = (
        "model_2900 has the lowest overall and random-rough XY tracking error, the lowest "
        "action delta and mechanical-power proxy among the two equally reliable late "
        "checkpoints, and lower cross-seed XY variation than model_2999. Model_2999 remains "
        "a retained late-training alternative."
    )
    official_text = f"""Unitree Go2 rough velocity official checkpoint
Selected: 2026-08-25

Training task: Unitree-Go2-Velocity-Rough
Training environments: 2048
Training seed: 42
Training iterations: 3000
Training transitions: 147456000
Training wall time: 2:11:41
Training exit code: 0
Training non-finite values: 0

Path: {relative(official_checkpoint)}
Size: {official_checkpoint.stat().st_size} bytes
SHA-256: {sha256(official_checkpoint)}

Selection protocol: {selection_protocol}
Terrain protocol: {terrain_protocol}
Official-checkpoint coverage: 384 episodes (192 flat, 192 random_rough).
Mean reward: {episode_mean("reward_sum"):.6f}.
Mean XY velocity error: {episode_mean("mean_xy_velocity_error_mps"):.6f} m/s.
Flat XY velocity error: {episode_mean("mean_xy_velocity_error_mps", flat_rows):.6f} m/s.
Random-rough XY velocity error: {episode_mean("mean_xy_velocity_error_mps", rough_rows):.6f} m/s.
Mean yaw velocity error: {episode_mean("mean_yaw_velocity_error_radps"):.6f} rad/s.
Timeout rate: {episode_mean("time_out"):.6f} ({termination_counts["time_out"]}/384).
Base-contact termination rate: {episode_mean("base_contact"):.6f} ({termination_counts["base_contact"]}/384).
Bad-orientation termination rate: {episode_mean("bad_orientation"):.6f} ({termination_counts["bad_orientation"]}/384).

Selection reason: {selection_reason}

This marks an existing checkpoint. It does not replace, copy, or modify the model file.
"""

    inventory_text = "".join(
        f"{path.stat().st_mtime_ns} {path.stat().st_size} {relative(path)}\n"
        for path in all_eval_files
    )
    evaluation_sha = "".join(
        f"{sha256(path)}  {relative(path)}\n" for path in all_eval_files
    )
    formal_artifacts = [official_checkpoint, *summary_paths]
    final_sha = "".join(
        f"{sha256(path)}  {relative(path)}\n" for path in formal_artifacts
    )

    return {
        output_dir / "OFFICIAL_ROUGH_CHECKPOINT.txt": official_text,
        output_dir / "rough_checkpoints_2026-08-25.txt": checkpoint_inventory,
        output_dir / "rough_eval_multiseed_2026-08-25.csv": aggregate_csv,
        output_dir / "rough_evaluation_inventory_2026-08-25.txt": inventory_text,
        output_dir / "rough_evaluation_sha256_2026-08-25.txt": evaluation_sha,
        output_dir / "rough_final_artifacts_2026-08-25.sha256": final_sha,
    }


def main() -> int:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    train_root = args.train_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    contents = build_contents(eval_root, train_root, output_dir)

    if args.create:
        existing = [str(path) for path in contents if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing files: {existing}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, text in contents.items():
            path.write_text(text)
            print(f"CREATE PASS {display_path(path)}")
        print("ARCHIVE_RESULT=CREATED")
        return 0

    failures = []
    for path, expected_text in contents.items():
        if not path.is_file():
            failures.append(f"missing: {path}")
            print(f"VERIFY FAIL {display_path(path)} missing")
            continue
        if path.read_text().splitlines() != expected_text.splitlines():
            failures.append(f"content mismatch: {path}")
            print(f"VERIFY FAIL {display_path(path)} content_mismatch")
            continue
        print(f"VERIFY PASS {display_path(path)}")

    if failures:
        print("ARCHIVE_RESULT=FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("ARCHIVE_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
