#!/usr/bin/env python3

"""Compose seven audited Teacher v1 demos and one overview from recorded clips."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
TERRAINS = (
    "flat",
    "random_rough",
    "pyramid_slope",
    "inverted_pyramid_slope",
    "pyramid_stairs",
    "inverted_pyramid_stairs",
    "boxes",
)
SCENARIOS = (
    "straight_low",
    "straight_mid",
    "straight_high",
    "forward_left_gentle",
    "forward_right_gentle",
    "forward_left_strong",
    "forward_right_strong",
)
OVERVIEW_SECONDS_PER_CLIP = 3.2
OVERVIEW_SKIP_SECONDS = 0.8

TITLES = {
    "flat": ("平地", "FLAT"),
    "random_rough": ("随机粗糙地形", "RANDOM ROUGH"),
    "pyramid_slope": ("金字塔斜坡", "PYRAMID SLOPE"),
    "inverted_pyramid_slope": ("倒金字塔斜坡", "INVERTED PYRAMID SLOPE"),
    "pyramid_stairs": ("上行阶梯", "PYRAMID STAIRS"),
    "inverted_pyramid_stairs": ("下行阶梯", "INVERTED PYRAMID STAIRS"),
    "boxes": ("随机箱体", "BOXES"),
}
SCENARIO_TITLES = {
    "straight_low": ("低速直行", "LOW-SPEED FORWARD"),
    "straight_mid": ("中速直行", "MID-SPEED FORWARD"),
    "straight_high": ("高速直行", "HIGH-SPEED FORWARD"),
    "forward_left_gentle": ("前进缓慢左转", "FORWARD + GENTLE LEFT"),
    "forward_right_gentle": ("前进缓慢右转", "FORWARD + GENTLE RIGHT"),
    "forward_left_strong": ("前进较强左转", "FORWARD + STRONG LEFT"),
    "forward_right_strong": ("前进较强右转", "FORWARD + STRONG RIGHT"),
}
STATUS = {
    "flat": ("PASS", "Fixed-command matrix: no safety termination."),
    "random_rough": ("KNOWN ISSUE", "Forward command: 7/120 safety terminations in matrix."),
    "pyramid_slope": ("PASS / LIMIT", "Forward-only interface is stable; broader-v1 reverse had 1/120 failure."),
    "inverted_pyramid_slope": ("KNOWN ISSUE", "Combined command: 2/120 safety terminations in matrix."),
    "pyramid_stairs": ("KNOWN ISSUE", "Forward-high/right-correction had occasional early terminations."),
    "inverted_pyramid_stairs": ("KNOWN ISSUE", "Systematic yaw tracking error; do not use as final teacher yet."),
    "boxes": ("KNOWN ISSUE", "High-level combined steering safety remains below target."),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size, index=2)


def title_card(width: int, height: int, title_zh: str, title_en: str, lines: list[str]) -> np.ndarray:
    image = Image.new("RGB", (width, height), (22, 27, 36))
    draw = ImageDraw.Draw(image)
    draw.text((52, 120), title_zh, font=font(42), fill=(245, 248, 255))
    draw.text((54, 180), title_en, font=font(31), fill=(90, 210, 255))
    for index, line in enumerate(lines):
        draw.text((56, 252 + 38 * index), line, font=font(20), fill=(210, 220, 235))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def load_clip(root: Path, terrain: str, scenario: str) -> dict:
    directory = root / terrain / scenario
    metadata_path = directory / "metadata.json"
    samples_path = directory / "tracking_timeseries.csv"
    if not metadata_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError(f"missing metadata/time series: {directory}")
    metadata = json.loads(metadata_path.read_text())
    videos = sorted((directory / "video").glob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"expected one video in {directory}, got {videos}")
    if sha256_file(videos[0]) != metadata["video_sha256"]:
        raise RuntimeError(f"video SHA mismatch: {videos[0]}")
    if sha256_file(samples_path) != metadata["tracking_timeseries_sha256"]:
        raise RuntimeError(f"tracking SHA mismatch: {samples_path}")
    samples = []
    with samples_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            samples.append({key: float(value) for key, value in row.items()})
    if not samples:
        raise RuntimeError(f"empty tracking data: {samples_path}")
    # The recorder's historical ``stalled`` field projects displacement onto
    # the initial world heading and is therefore invalid once a commanded turn
    # bends the path.  Derive display-only forward progress and stall from the
    # body-frame vx trace, which matches the final vx+yaw command interface.
    step_dt = float(metadata["step_dt"])
    warmup_time = int(metadata["warmup_steps"]) * step_dt
    body_forward_distance = 0.0
    for sample in samples:
        body_forward_distance += sample["actual_x_mps"] * step_dt
        sample["display_body_forward_distance_m"] = body_forward_distance
        sample["display_stalled"] = float(
            sample["time_seconds"] >= warmup_time
            and sample["command_x_mps"] >= 0.1
            and sample["actual_x_mps"] < 0.05
        )
    return {"metadata": metadata, "samples": samples, "video": videos[0], "directory": directory}


def validate_inventory(root: Path) -> tuple[dict[tuple[str, str], dict], tuple[float, int, int]]:
    clips = {}
    properties = None
    for terrain in TERRAINS:
        for scenario in SCENARIOS:
            clip = load_clip(root, terrain, scenario)
            meta = clip["metadata"]
            if meta["terrain_type"] != terrain:
                raise RuntimeError(f"terrain mismatch: {clip['directory']}")
            if float(meta["command"]["lin_vel_x"]) < 0 or float(meta["command"]["lin_vel_y"]) != 0:
                raise RuntimeError(f"demo command violates forward-only interface: {clip['directory']}")
            cap = cv2.VideoCapture(str(clip["video"]))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {clip['video']}")
            current = (
                float(cap.get(cv2.CAP_PROP_FPS)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            cap.release()
            if properties is None:
                properties = current
            if current != properties:
                raise RuntimeError(f"incompatible video properties: {clip['video']} {current} != {properties}")
            clips[(terrain, scenario)] = clip
    assert properties is not None
    return clips, properties


def graph_y(value: float, limit: float, top: int, height: int) -> int:
    return int(top + height / 2 - np.clip(value / limit, -1.0, 1.0) * (height / 2 - 5))


def draw_panel(frame: np.ndarray, clip: dict, terrain: str, scenario: str, sample_index: int) -> np.ndarray:
    metadata = clip["metadata"]
    samples = clip["samples"]
    sample = samples[sample_index]
    height, width = frame.shape[:2]
    panel_h = min(220, max(180, height // 3))
    view_h = height - panel_h
    result = np.full_like(frame, (16, 20, 28))
    result[:view_h] = cv2.resize(frame, (width, view_h), interpolation=cv2.INTER_AREA)

    status, note = STATUS[terrain]
    color = (80, 230, 110) if status == "PASS" else (40, 185, 255)
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (width, 92), (8, 12, 18), -1)
    cv2.addWeighted(overlay, 0.76, result, 0.24, 0, result)
    cv2.putText(result, f"{TITLES[terrain][1]} | level {metadata['terrain_level']} | {SCENARIO_TITLES[scenario][1]}",
                (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    checkpoint_name = Path(metadata["checkpoint"]).name
    cv2.putText(result, f"{checkpoint_name} sha={metadata['checkpoint_sha256'][:12]} | train seed=42 | eval seed={metadata['evaluation_scenario_seed']}",
                (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 220, 238), 1, cv2.LINE_AA)
    cv2.putText(result, f"{status}: {note}", (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    panel_top = view_h
    cmd = (sample["command_x_mps"], sample["command_y_mps"], sample["command_yaw_radps"])
    actual = (sample["actual_x_mps"], sample["actual_y_mps"], sample["actual_yaw_radps"])
    cv2.putText(result, f"t={sample['time_seconds']:5.2f}s  CMD vx/vy/yaw={cmd[0]:+.2f}/{cmd[1]:+.2f}/{cmd[2]:+.2f}",
                (18, panel_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110, 225, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f"ACT vx/vy/yaw={actual[0]:+.2f}/{actual[1]:+.2f}/{actual[2]:+.2f}  XYerr={sample['xy_error_mps']:.3f}  YawErr={sample['yaw_error_radps']:.3f}",
                (18, panel_top + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (105, 255, 120), 1, cv2.LINE_AA)
    flags = (
        f"path={sample['cumulative_xy_distance_m']:.2f}m  "
        f"body-forward={sample['display_body_forward_distance_m']:.2f}m  "
        f"stall={int(sample['display_stalled'])}  done={int(sample['done'])}"
    )
    cv2.putText(result, flags, (18, panel_top + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 235), 1, cv2.LINE_AA)

    graph_top = panel_top + 82
    graph_h = panel_h - 92
    graph_w = (width - 48) // 3
    definitions = (
        ("vx m/s", "command_x_mps", "actual_x_mps"),
        ("vy m/s", "command_y_mps", "actual_y_mps"),
        ("yaw rad/s", "command_yaw_radps", "actual_yaw_radps"),
    )
    for index, (label, command_key, actual_key) in enumerate(definitions):
        left = 12 + index * (graph_w + 12)
        right = left + graph_w
        values = [abs(row[command_key]) for row in samples] + [abs(row[actual_key]) for row in samples]
        limit = max(0.15, max(values) * 1.10)
        cv2.rectangle(result, (left, graph_top), (right, graph_top + graph_h), (55, 62, 73), 1)
        zero = graph_y(0, limit, graph_top, graph_h)
        cv2.line(result, (left, zero), (right, zero), (85, 90, 100), 1)
        count = sample_index + 1
        if count > 1:
            xs = np.linspace(left + 2, right - 2, count).astype(np.int32)
            cp = np.column_stack((xs, [graph_y(row[command_key], limit, graph_top, graph_h) for row in samples[:count]])).astype(np.int32)
            ap = np.column_stack((xs, [graph_y(row[actual_key], limit, graph_top, graph_h) for row in samples[:count]])).astype(np.int32)
            cv2.polylines(result, [cp], False, (255, 220, 50), 1, cv2.LINE_AA)
            cv2.polylines(result, [ap], False, (90, 255, 105), 2, cv2.LINE_AA)
        cv2.putText(result, label, (left + 5, graph_top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 235, 245), 1, cv2.LINE_AA)
    return result


def write_card(writer, card: np.ndarray, fps: float, seconds: float) -> int:
    count = round(fps * seconds)
    for _ in range(count):
        writer.write(card)
    return count


def append_clip(writer, clip: dict, terrain: str, scenario: str, fps: float,
                start_seconds: float = 0.0, duration_seconds: float | None = None) -> int:
    cap = cv2.VideoCapture(str(clip["video"]))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = min(frame_count - 1, max(0, round(start_seconds * fps)))
    stop = frame_count if duration_seconds is None else min(frame_count, start + round(duration_seconds * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = 0
    for frame_index in range(start, stop):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"unexpected EOF: {clip['video']}")
        sample_index = min(len(clip["samples"]) - 1, int(frame_index * len(clip["samples"]) / frame_count))
        writer.write(draw_panel(frame, clip, terrain, scenario, sample_index))
        written += 1
    cap.release()
    return written


def open_writer(path: Path, fps: float, width: int, height: int):
    if path.exists():
        raise FileExistsError(path)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer: {path}")
    return writer


def compose(root: Path, output_dir: Path) -> dict:
    clips, (fps, width, height) = validate_inventory(root)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    artifacts = []
    for terrain in TERRAINS:
        path = output_dir / f"teacher_v1_{terrain}_full_demo.mp4"
        writer = open_writer(path, fps, width, height)
        frames = 0
        try:
            zh, en = TITLES[terrain]
            status, note = STATUS[terrain]
            frames += write_card(writer, title_card(width, height, f"{zh} Teacher v1", f"{en} TEACHER v1", [
                "最终展示接口 / Display interface: vx >= 0, vy = 0, forward steering only",
                f"{status}: {note}",
                "黄色=目标 / cyan=command; 绿色=实际 / green=actual",
            ]), fps, 3.0)
            for scenario in SCENARIOS:
                frames += append_clip(writer, clips[(terrain, scenario)], terrain, scenario, fps)
            frames += write_card(writer, title_card(width, height, "证据边界", "EVIDENCE BOUNDARY", [
                "This is one terrain-specific Teacher checkpoint, not a unified policy.",
                "Known failures are retained and labeled; timeout alone is not treated as success.",
            ]), fps, 2.0)
        finally:
            writer.release()
        artifacts.append({"kind": "terrain_full", "terrain": terrain, "path": str(path.resolve()),
                          "frames": frames, "duration_seconds": frames / fps,
                          "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    overview = output_dir / "teacher_v1_seven_terrain_overview_demo.mp4"
    writer = open_writer(overview, fps, width, height)
    frames = 0
    try:
        frames += write_card(writer, title_card(width, height, "七地形 Teacher v1 总览", "SEVEN TERRAIN TEACHER v1 OVERVIEW", [
            "7 independent checkpoints / 7个独立策略；不是统一策略 / NOT a unified policy",
            "Display interface: vx >= 0, vy = 0, forward steering only",
            "Training seed=42; evaluation scenario seed is shown in every clip.",
        ]), fps, 4.0)
        for terrain in TERRAINS:
            zh, en = TITLES[terrain]
            status, note = STATUS[terrain]
            frames += write_card(writer, title_card(width, height, zh, en, [f"{status}: {note}"]), fps, 1.2)
            for scenario in SCENARIOS:
                frames += append_clip(writer, clips[(terrain, scenario)], terrain, scenario, fps,
                                      OVERVIEW_SKIP_SECONDS, OVERVIEW_SECONDS_PER_CLIP)
        frames += write_card(writer, title_card(width, height, "下一步", "NEXT STEP", [
            "Freeze evidence, remediate failed teachers, then distill a true recurrent perceptive student.",
            "保留证据，修复未通过的 Teacher，再训练真正统一的循环感知 Student。",
        ]), fps, 4.0)
    finally:
        writer.release()
    artifacts.append({"kind": "overview", "path": str(overview.resolve()), "frames": frames,
                      "duration_seconds": frames / fps, "sha256": sha256_file(overview),
                      "size_bytes": overview.stat().st_size})

    metadata = {
        "artifact": "Unitree Go2 Terrain Teacher v1 forward-steering demos v2",
        "claim_scope": "seven independent terrain-specific checkpoints; not a unified policy",
        "command_interface": {"lin_vel_x": "non-negative and within each training range",
                              "lin_vel_y": 0.0, "yaw": "nonzero only while moving forward"},
        "training_seed": 42,
        "hud_stall_definition": (
            "After warmup, command vx >= 0.1 m/s and instantaneous body-frame actual vx < 0.05 m/s. "
            "Body-forward distance integrates actual body-frame vx; it is not a world-frame net displacement."
        ),
        "source_root": str(root.resolve()),
        "source_clips": 49,
        "fps": fps,
        "resolution": [width, height],
        "artifacts": artifacts,
    }
    (output_dir / "demo_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    with (output_dir / "all_composed_demos.sha256").open("w") as stream:
        for item in artifacts:
            stream.write(f"{item['sha256']}  {Path(item['path']).name}\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()
    clips, properties = validate_inventory(args.source_root.resolve())
    if args.validate_only:
        print(json.dumps({"status": "validated", "clips": len(clips), "video_properties": properties,
                          "planned_outputs": 8}, indent=2))
        return
    print(json.dumps(compose(args.source_root.resolve(), args.output_dir.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
