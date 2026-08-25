#!/usr/bin/env python3

"""Create or verify the additive Go2 rough multi-training-seed archive."""

import argparse
import contextlib
import hashlib
import io
from pathlib import Path

from audit_go2_rough_multitraining import (
    CHECKPOINT_HASHES,
    EVALUATION_ROOTS,
    EVALUATION_SEEDS,
    MODELS,
    OFFICIAL_MODELS,
    TRAINING_RUNS,
    TRAINING_SEEDS,
    aggregate,
    build_fixed_csv,
    build_selected_csv,
    collect,
    run_name,
    subsets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "doc/repro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true", help="Create all six files; refuse overwrite.")
    mode.add_argument("--verify", action="store_true", help="Verify all six files byte-for-byte.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def checkpoint_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def build_official_text(rows_by_key: dict[tuple[int, int], list[dict]]) -> str:
    lines = [
        "Unitree Go2 rough velocity multi-training-seed official checkpoints",
        "Selected: 2026-08-25",
        "",
        "Protocol: rough_stratified v1; 64 environments x 2 episodes; evaluation scenario seeds 7, 42, 123.",
        "Training configuration: 2048 environments; 3000 PPO iterations; 147456000 transitions per training seed.",
        "",
    ]
    for training_seed in TRAINING_SEEDS:
        model = OFFICIAL_MODELS[training_seed]
        checkpoint = TRAINING_RUNS[training_seed] / f"model_{model}.pt"
        overall = aggregate(rows_by_key[(training_seed, model)])
        rough = aggregate(subsets(rows_by_key[(training_seed, model)])["rough"])
        terminations = {
            name: sum(int(row[name]) for row in rows_by_key[(training_seed, model)])
            for name in ("time_out", "base_contact", "bad_orientation")
        }
        lines.extend(
            [
                f"Training seed: {training_seed}",
                f"Checkpoint: model_{model}.pt",
                f"Path: {relative(checkpoint)}",
                f"Size: {checkpoint.stat().st_size} bytes",
                f"SHA-256: {sha256(checkpoint)}",
                "Coverage: 384 episodes (192 flat, 192 random_rough).",
                f"Mean reward: {overall['reward']:.6f}.",
                f"Mean XY velocity error: {overall['xy_error_mps']:.6f} m/s.",
                f"Random-rough XY velocity error: {rough['xy_error_mps']:.6f} m/s.",
                f"Mean yaw velocity error: {overall['yaw_error_radps']:.6f} rad/s.",
                f"Timeout: {terminations['time_out']}/384.",
                f"Base-contact termination: {terminations['base_contact']}/384.",
                f"Bad-orientation termination: {terminations['bad_orientation']}/384.",
                "",
            ]
        )
    lines.extend(
        [
            "Fixed-iteration multi-training-seed recommendation: model_2999.pt.",
            "The historical seed-42 official model_2900.pt remains unchanged; model_2999 is the primary fixed-iteration comparison across all three training seeds.",
            "This archive marks existing checkpoints. It does not replace, copy, or modify model files.",
            "",
        ]
    )
    return "\n".join(lines)


def build_contents(output_dir: Path) -> dict[Path, str]:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        rows_by_key, failures = collect()
    if failures:
        raise RuntimeError("Multi-training audit failed: " + "; ".join(failures))

    selected_csv = build_selected_csv(rows_by_key)
    fixed_csv = build_fixed_csv(rows_by_key)
    official_text = build_official_text(rows_by_key)

    evaluation_files = sorted(
        path
        for root in EVALUATION_ROOTS.values()
        for path in root.rglob("*")
        if path.is_file()
    )
    if len(evaluation_files) != 120:
        raise RuntimeError(f"Expected 120 evaluation files, found {len(evaluation_files)}")
    inventory = "".join(
        f"{path.stat().st_size} {relative(path)}\n" for path in evaluation_files
    )
    evaluation_sha = "".join(
        f"{sha256(path)}  {relative(path)}\n" for path in evaluation_files
    )

    checkpoint_lines = []
    for training_seed in TRAINING_SEEDS:
        checkpoints = sorted(
            TRAINING_RUNS[training_seed].glob("model_*.pt"), key=checkpoint_number
        )
        if len(checkpoints) != 31:
            raise RuntimeError(
                f"Training seed {training_seed}: expected 31 checkpoints, found {len(checkpoints)}"
            )
        for model in MODELS:
            checkpoint = TRAINING_RUNS[training_seed] / f"model_{model}.pt"
            if sha256(checkpoint) != CHECKPOINT_HASHES[training_seed][model]:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
        official = OFFICIAL_MODELS[training_seed]
        checkpoint_lines.append(
            f"training_seed={training_seed} official=model_{official}.pt "
            f"sha256={CHECKPOINT_HASHES[training_seed][official]}\n"
        )
    official_text += "Official selection inventory:\n" + "".join(checkpoint_lines)

    contents = {
        output_dir / "OFFICIAL_ROUGH_CHECKPOINTS_MULTITRAINING.txt": official_text,
        output_dir / "rough_multitraining_selected_official_2026-08-25.csv": selected_csv,
        output_dir / "rough_multitraining_fixed_checkpoints_2026-08-25.csv": fixed_csv,
        output_dir / "rough_multitraining_evaluation_inventory_2026-08-25.txt": inventory,
        output_dir / "rough_multitraining_evaluation_sha256_2026-08-25.txt": evaluation_sha,
    }

    external_artifacts = []
    for training_seed in TRAINING_SEEDS:
        model = OFFICIAL_MODELS[training_seed]
        external_artifacts.append(TRAINING_RUNS[training_seed] / f"model_{model}.pt")
        for candidate in MODELS:
            for evaluation_seed in EVALUATION_SEEDS:
                external_artifacts.append(
                    EVALUATION_ROOTS[training_seed]
                    / run_name(training_seed, candidate, evaluation_seed)
                    / "summary.json"
                )
    final_sha = "".join(
        f"{sha256(path)}  {relative(path)}\n" for path in external_artifacts
    )
    final_sha += "".join(
        f"{text_sha256(text)}  {relative(path)}\n"
        for path, text in contents.items()
    )
    contents[output_dir / "rough_multitraining_final_artifacts_2026-08-25.sha256"] = final_sha
    return contents


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    contents = build_contents(output_dir)
    if args.create:
        existing = [str(path) for path in contents if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing files: {existing}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, text in contents.items():
            path.write_text(text)
            print(f"CREATE PASS {relative(path)}")
        print("ARCHIVE_RESULT=CREATED")
        return 0

    failures = []
    for path, expected in contents.items():
        if not path.is_file():
            failures.append(f"missing: {path}")
            print(f"VERIFY FAIL {relative(path)} missing")
        elif path.read_text().splitlines() != expected.splitlines():
            failures.append(f"content mismatch: {path}")
            print(f"VERIFY FAIL {relative(path)} content_mismatch")
        else:
            print(f"VERIFY PASS {relative(path)}")
    if failures:
        print("ARCHIVE_RESULT=FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("ARCHIVE_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
