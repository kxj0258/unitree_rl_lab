#!/usr/bin/env python3

"""Compose audited Go2 terrain clips into labeled portfolio demo videos."""

import argparse
import csv
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


def load_tracking_samples(item: dict) -> list[dict[str, float]]:
    path = Path(item["tracking_timeseries"])
    if not path.is_file():
        raise FileNotFoundError(f"tracking time series not found: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty tracking time series: {path}")
    samples = [{key: float(value) for key, value in row.items()} for row in rows]
    if "cumulative_xy_distance_m" not in samples[0]:
        distance = 0.0
        step_dt = float(item["step_dt"])
        for sample in samples:
            if not sample["done"]:
                distance += float(np.hypot(sample["actual_x_mps"], sample["actual_y_mps"])) * step_dt
            sample["cumulative_xy_distance_m"] = distance
    terminations = 0
    for sample in samples:
        terminations += int(sample["done"])
        sample["termination_count"] = float(terminations)
    return samples


def draw_dynamic_tracking(
    frame: np.ndarray,
    samples: list[dict[str, float]],
    sample_index: int,
    velocity_limit: float,
) -> np.ndarray:
    result = frame.copy()
    sample = samples[sample_index]
    height, width = result.shape[:2]
    panel_top = max(0, height - 142)
    overlay = result.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, result, 0.32, 0.0, result)

    live = (
        f"t={sample['time_seconds']:5.2f}s  cmd vx={sample['command_x_mps']:+.3f} m/s  "
        f"actual vx={sample['actual_x_mps']:+.3f} m/s  XY error={sample['xy_error_mps']:.3f} m/s"
    )
    cumulative = (
        f"distance={sample['cumulative_xy_distance_m']:.2f} m  "
        f"terminations={int(sample['termination_count'])}  "
        "cyan=command vx  green=actual vx"
    )
    cv2.putText(result, live, (18, panel_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, cumulative, (18, panel_top + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    graph_left, graph_right = 20, width - 20
    graph_top, graph_bottom = panel_top + 66, height - 14
    def graph_y(value: float) -> int:
        normalized = np.clip(value / velocity_limit, -1.0, 1.0)
        return int(graph_bottom - (normalized + 1.0) * 0.5 * (graph_bottom - graph_top))

    zero_y = graph_y(0.0)
    cv2.line(result, (graph_left, zero_y), (graph_right, zero_y), (125, 125, 125), 1, cv2.LINE_AA)
    if sample_index > 0:
        indices = np.arange(sample_index + 1)
        xs = graph_left + indices * (graph_right - graph_left) / max(1, len(samples) - 1)
        command_points = np.column_stack(
            (xs, [graph_y(row["command_x_mps"]) for row in samples[: sample_index + 1]])
        ).astype(np.int32)
        actual_points = np.column_stack(
            (xs, [graph_y(row["actual_x_mps"]) for row in samples[: sample_index + 1]])
        ).astype(np.int32)
        cv2.polylines(result, [command_points], False, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.polylines(result, [actual_points], False, (80, 255, 80), 2, cv2.LINE_AA)
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
        samples = load_tracking_samples(item)
        velocity_limit = max(
            0.5,
            max(abs(row["command_x_mps"]) for row in samples),
            max(abs(row["actual_x_mps"]) for row in samples),
        ) * 1.10
        frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        lines = [
            f"{label} | level {item['terrain_level']}",
            f"training seed={item['training_seed']} | evaluation scenario seed={item['evaluation_scenario_seed']}",
            f"post-warmup XY error={item['post_warmup_mean_xy_velocity_error_mps']:.3f} m/s | actual vx={item['post_warmup_mean_actual_x_velocity_mps']:.3f} m/s",
        ]
        clip_frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sample_index = min(
                len(samples) - 1,
                int(clip_frame_index * len(samples) / frame_count),
            )
            labeled = draw_lines(frame, lines)
            writer.write(draw_dynamic_tracking(labeled, samples, sample_index, velocity_limit))
            total_source_frames += 1
            clip_frame_index += 1
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
