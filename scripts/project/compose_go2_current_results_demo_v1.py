#!/usr/bin/env python3

"""Compose the audited multi-checkpoint Go2 current-results Demo v1.

This is deliberately a presentation of several independently identified policies.  It
must never be described as a single unified seven-terrain policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVALUATION = REPO / "evaluation"


@dataclass(frozen=True)
class Clip:
    section: str
    label: str
    path: Path
    expected_sha256: str
    start_seconds: float
    duration_seconds: float
    lines: tuple[str, ...]


PORTFOLIO = EVALUATION / "portfolio_go2_heightscan_model10000_level5_vx035_assetfollow_selected4_34s_2026-08-26.mp4"
MODEL4999_ROOT = EVALUATION / "video_v5_stairs_directional_diagnostic_model4999_trainseed42_evalseed42_level6_vx035_20s_2026-08-27"
MODEL900_ROOT = EVALUATION / "video_v5_stairs_headinglocked_ascendingremediation_model900_trainseed42_evalseed42_inverted_level7_vx035_heading0_inityaw0_20s_2026-08-28_v1"

CLIPS = (
    Clip("foundation", "flat", PORTFOLIO, "90a4909bb527eea70034e699cd9f3a81a1f119e619772309b0a06845eb2de887", 4.0, 3.0,
         ("FOUNDATION POLICY | model_10000", "Flat | level 5 | vx=0.35 m/s", "training seed=42 | evaluation scenario seed=42")),
    Clip("foundation", "random_rough", PORTFOLIO, "90a4909bb527eea70034e699cd9f3a81a1f119e619772309b0a06845eb2de887", 12.0, 3.0,
         ("FOUNDATION POLICY | model_10000", "Random rough | level 5 | vx=0.35 m/s", "training seed=42 | evaluation scenario seed=42")),
    Clip("foundation", "pyramid_slope", PORTFOLIO, "90a4909bb527eea70034e699cd9f3a81a1f119e619772309b0a06845eb2de887", 20.0, 3.0,
         ("FOUNDATION POLICY | model_10000", "Pyramid slope | level 5 | vx=0.35 m/s", "training seed=42 | evaluation scenario seed=42")),
    Clip("foundation", "inverted_pyramid_slope", PORTFOLIO, "90a4909bb527eea70034e699cd9f3a81a1f119e619772309b0a06845eb2de887", 28.0, 3.0,
         ("FOUNDATION POLICY | model_10000", "Inverted pyramid slope | level 5 | vx=0.35 m/s", "training seed=42 | evaluation scenario seed=42")),
    Clip("stairs_baseline", "pyramid_stairs_level6", MODEL4999_ROOT / "pyramid_stairs_level6_vx035/video/pyramid_stairs_level6-step-0.mp4",
         "530620906a69b695e94e43971013dd4afd8c83d31e525a71e55d054ec82e231c", 1.0, 6.0,
         ("STAIRS SPECIALIST | model_4999", "Pyramid stairs | level 6 | net forward=6.41 m", "train seed=42 | eval scenario seed=42 | no base contact")),
    Clip("diagnosed_limit", "inverted_stairs_before", MODEL4999_ROOT / "inverted_pyramid_stairs_level6_vx035/video/inverted_pyramid_stairs_level6-step-0.mp4",
         "11f04356161c2f1d0c5a5fbfb5b48f59f397e615c15dae6d3048976e28751691", 4.0, 8.0,
         ("DIAGNOSED LIMIT | model_4999", "Inverted stairs | level 6 | net forward=0.89 m", "stalled=42.2% | timeout alone is NOT success")),
    Clip("remediation", "inverted_stairs_after", MODEL900_ROOT / "inverted_pyramid_stairs_level7_vx035_heading0_inityaw0/video/inverted_pyramid_stairs_level7-step-0.mp4",
         "ff4192b91068c8a1725017078718647ae412ba64dcbe3db4577a47dcdbceb94e", 0.0, 12.0,
         ("ASCENDING REMEDIATION | model_900", "Harder inverted stairs | level 7 | net forward=7.22 m", "stalled=0% | base contact=0 | bad orientation=0")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_banner(frame: np.ndarray, lines: tuple[str, ...]) -> np.ndarray:
    result = frame.copy()
    overlay = result.copy()
    height = 24 + 31 * len(lines)
    cv2.rectangle(overlay, (0, 0), (result.shape[1], height), (12, 15, 22), -1)
    cv2.addWeighted(overlay, 0.74, result, 0.26, 0.0, result)
    for index, line in enumerate(lines):
        cv2.putText(result, line, (24, 37 + 31 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72 if index == 0 else 0.56, (255, 255, 255), 2 if index == 0 else 1, cv2.LINE_AA)
    return result


def title_card(width: int, height: int, title: str, lines: tuple[str, ...]) -> np.ndarray:
    image = np.full((height, width, 3), (24, 27, 35), dtype=np.uint8)
    cv2.putText(image, title, (64, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.18, (255, 255, 255), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (66, 270 + 45 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.67,
                    (205, 218, 235), 1, cv2.LINE_AA)
    return image


def validate_sources() -> tuple[float, int, int, list[dict]]:
    inventory = []
    properties = None
    hash_cache: dict[Path, str] = {}
    for clip in CLIPS:
        path = clip.path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = hash_cache.setdefault(path, sha256_file(path))
        if actual_hash != clip.expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual_hash}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if properties is None:
            properties = (fps, width, height)
        if properties != (fps, width, height):
            raise RuntimeError(f"incompatible source properties for {path}: {(fps, width, height)}")
        if clip.start_seconds < 0 or clip.duration_seconds <= 0:
            raise RuntimeError(f"invalid trim for {clip.label}")
        if clip.start_seconds + clip.duration_seconds > frame_count / fps + 1.0 / fps:
            raise RuntimeError(f"trim exceeds source duration for {clip.label}")
        inventory.append({
            "section": clip.section, "label": clip.label, "path": str(path),
            "sha256": actual_hash, "start_seconds": clip.start_seconds,
            "duration_seconds": clip.duration_seconds, "source_frames": frame_count,
        })
    assert properties is not None
    return *properties, inventory


def write_repeated(writer: cv2.VideoWriter, frame: np.ndarray, count: int) -> int:
    for _ in range(count):
        writer.write(frame)
    return count


def compose(output: Path) -> dict:
    fps, width, height, inventory = validate_sources()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists() or output.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite output or sidecar: {output}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output writer: {output}")
    written = 0
    try:
        intro = title_card(width, height, "UNITREE GO2 | CURRENT RESULTS DEMO v1", (
            "Velocity tracking across mixed terrain, plus a documented stairs remediation.",
            "MULTI-CHECKPOINT SHOWCASE: this is not yet one unified seven-terrain policy.",
            "All policies: training seed 42. All shown scenarios: evaluation seed 42.",
        ))
        written += write_repeated(writer, intro, round(3.0 * fps))
        for index, clip in enumerate(CLIPS):
            if clip.section == "diagnosed_limit":
                card = title_card(width, height, "FAILURE-DRIVEN REMEDIATION", (
                    "The baseline tracks local velocity but fails to make useful uphill progress.",
                    "A yaw-gated ascending curriculum targets the observed failure mode.",
                ))
                written += write_repeated(writer, card, round(2.0 * fps))
            cap = cv2.VideoCapture(str(clip.path))
            start = round(clip.start_seconds * fps)
            count = round(clip.duration_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"unexpected EOF in {clip.label}")
                writer.write(draw_banner(frame, clip.lines))
                written += 1
            cap.release()
        outro = title_card(width, height, "NEXT: ONE UNIFIED SEVEN-TERRAIN STUDENT", (
            "7 from-scratch terrain teachers -> balanced online multi-teacher distillation",
            "-> balanced PPO consolidation -> composite mixed-terrain validation.",
            "Current demo remains an audited multi-checkpoint result.",
        ))
        written += write_repeated(writer, outro, round(4.0 * fps))
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"empty output: {output}")
    output_hash = sha256_file(output)
    return {
        "artifact": "Unitree Go2 current-results Demo v1",
        "claim_scope": "multi-checkpoint showcase; not a unified seven-terrain policy",
        "training_seed": 42,
        "evaluation_scenario_seed": 42,
        "source_inventory": inventory,
        "fps": fps,
        "resolution": [width, height],
        "frames": written,
        "duration_seconds": written / fps,
        "output": str(output.resolve()),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": output_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true", help="Validate every source, hash and trim without composing.")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.dry_run:
        fps, width, height, inventory = validate_sources()
        print(json.dumps({"status": "validated", "fps": fps, "resolution": [width, height],
                          "planned_duration_seconds": 3.0 + 2.0 + 4.0 + sum(c.duration_seconds for c in CLIPS),
                          "output": str(output), "source_inventory": inventory}, indent=2))
        return
    metadata = compose(output)
    metadata_path = output.with_suffix(".json")
    sha_path = output.with_suffix(".sha256")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    sha_path.write_text(f"{metadata['output_sha256']}  {output.name}\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
