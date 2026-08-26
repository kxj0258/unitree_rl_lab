#!/usr/bin/env python3

"""Compose audited Go2 terrain clips into labeled portfolio demo videos."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_lines(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    result = frame.copy()
    overlay = result.copy()
    banner_height = 20 + 32 * len(lines)
    cv2.rectangle(overlay, (0, 0), (result.shape[1], banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, result, 0.38, 0.0, result)
    for index, line in enumerate(lines):
        scale = 0.78 if index == 0 else 0.58
        thickness = 2 if index == 0 else 1
        cv2.putText(result, line, (24, 34 + index * 32), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return result


def compose(root: Path, output: Path, title: str) -> dict:
    metadata_paths = sorted(root.glob("*/metadata.json"))
    if not metadata_paths:
        raise RuntimeError(f"no metadata found under {root}")
    items = [json.loads(path.read_text()) for path in metadata_paths]
    terrain_order = {
        name: index
        for index, name in enumerate(
            ("flat", "random_rough", "pyramid_slope", "inverted_pyramid_slope", "pyramid_stairs", "inverted_pyramid_stairs", "boxes")
        )
    }
    training_seed_order = {42: 0, 7: 1, 123: 2}
    if all(item["task"] == "Unitree-Go2-Velocity-Rough" for item in items):
        items.sort(key=lambda item: training_seed_order[item["training_seed"]])
    else:
        items.sort(key=lambda item: terrain_order[item["terrain_type"]])
    first_video = cv2.VideoCapture(items[0]["video"])
    fps = first_video.get(cv2.CAP_PROP_FPS)
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")

    intro = np.full((height, width, 3), 28, dtype=np.uint8)
    command_x = items[0]["command"]["lin_vel_x"]
    scenario_seed = items[0]["evaluation_scenario_seed"]
    seconds_per_clip = items[0]["simulated_seconds"]
    camera_mode = items[0].get("camera", {}).get("mode", "legacy")
    intro_lines = [
        title,
        f"Fixed level 5 | command vx={command_x:.2f} m/s | {seconds_per_clip:.0f} s/clip | evaluation scenario seed={scenario_seed}",
        f"camera={camera_mode}; raw clips, time series, metadata and SHA-256 are preserved",
    ]
    intro = draw_lines(intro, intro_lines)
    for _ in range(round(fps * 2.0)):
        writer.write(intro)

    total_source_frames = 0
    for item in items:
        cap = cv2.VideoCapture(item["video"])
        if not cap.isOpened():
            raise RuntimeError(f"failed to open {item['video']}")
        label = item["terrain_type"].replace("_", " ")
        lines = [
            f"{label} | level {item['terrain_level']}",
            f"training seed={item['training_seed']} | evaluation scenario seed={item['evaluation_scenario_seed']}",
            f"post-warmup XY error={item['post_warmup_mean_xy_velocity_error_mps']:.3f} m/s | actual vx={item['post_warmup_mean_actual_x_velocity_mps']:.3f} m/s",
            f"distance={item['cumulative_xy_distance_m']:.2f} m | terminations={sum(item['terminations'].values())}",
        ]
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(draw_lines(frame, lines))
            total_source_frames += 1
        cap.release()
    writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"empty output: {output}")
    return {
        "title": title,
        "source_root": str(root.resolve()),
        "source_clips": len(items),
        "source_frames": total_source_frames,
        "fps": fps,
        "resolution": [width, height],
        "duration_seconds": (round(fps * 2.0) + total_source_frames) / fps,
        "output": str(output.resolve()),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite: {args.output}")
    metadata = compose(args.source_root.resolve(), args.output.resolve(), args.title)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    args.output.with_suffix(".sha256").write_text(f"{metadata['output_sha256']}  {args.output.name}\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
