#!/usr/bin/env python3

import argparse
import math
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = {
    "reward": "Train/mean_reward",
    "episode_len": "Train/mean_episode_length",
    "vel_xy_err": "Metrics/base_velocity/error_vel_xy",
    "yaw_err": "Metrics/base_velocity/error_vel_yaw",
    "time_out": "Episode_Termination/time_out",
    "base_contact": "Episode_Termination/base_contact",
    "bad_orient": "Episode_Termination/bad_orientation",
    "terrain": "Curriculum/terrain_levels",
    "cmd_level": "Curriculum/lin_vel_cmd_levels",
    "lin_track": "Episode_Reward/track_lin_vel_xy",
    "yaw_track": "Episode_Reward/track_ang_vel_z",
    "noise": "Policy/mean_noise_std",
    "value_loss": "Loss/value_function",
    "surrogate": "Loss/surrogate",
    "fps": "Perf/total_fps",
}


def mean(values):
    return statistics.mean(values)


def slope_per_100(values):
    xs = list(range(len(values)))
    x_mean = mean(xs)
    y_mean = mean(values)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, values)
    ) / denominator
    return slope * 100.0


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a Unitree Go2 TensorBoard training run."
    )
    parser.add_argument("run_dir", help="TensorBoard run directory")
    parser.add_argument(
        "--stage-size",
        type=int,
        default=500,
        help="Iterations per coarse training stage",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Checkpoint/window interval",
    )
    parser.add_argument(
        "--tail-size",
        type=int,
        default=500,
        help="Number of final iterations used for trend analysis",
    )
    args = parser.parse_args()

    accumulator = EventAccumulator(
        args.run_dir,
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()

    available = set(accumulator.Tags().get("scalars", []))
    missing = [tag for tag in TAGS.values() if tag not in available]
    if missing:
        raise RuntimeError(f"Missing TensorBoard scalar tags: {missing}")

    data = {
        name: [float(event.value) for event in accumulator.Scalars(tag)]
        for name, tag in TAGS.items()
    }

    lengths = {name: len(values) for name, values in data.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Scalar lengths differ: {lengths}")

    total = len(data["reward"])
    print(f"run_dir: {args.run_dir}")
    print(f"iterations: {total}")
    print(f"scalar_tags: {len(TAGS)}")

    non_finite = [
        (name, index, value)
        for name, values in data.items()
        for index, value in enumerate(values)
        if not math.isfinite(value)
    ]
    print(f"non_finite_values: {len(non_finite)}")
    if non_finite:
        raise RuntimeError(f"Non-finite values found: {non_finite[:20]}")

    print("\n=== Stage means ===")
    print(
        "window      reward  ep_len  xy_err  yaw_err  timeout  "
        "bad_ori  terrain  cmd_lvl  noise   fps"
    )
    for start in range(0, total, args.stage_size):
        end = min(start + args.stage_size, total)
        values = {
            name: mean(data[name][start:end])
            for name in (
                "reward",
                "episode_len",
                "vel_xy_err",
                "yaw_err",
                "time_out",
                "bad_orient",
                "terrain",
                "cmd_level",
                "noise",
                "fps",
            )
        }
        print(
            f"{start:04d}-{end - 1:04d} "
            f"{values['reward']:7.3f} "
            f"{values['episode_len']:7.2f} "
            f"{values['vel_xy_err']:7.4f} "
            f"{values['yaw_err']:8.4f} "
            f"{values['time_out']:8.4f} "
            f"{values['bad_orient']:8.4f} "
            f"{values['terrain']:8.4f} "
            f"{values['cmd_level']:8.4f} "
            f"{values['noise']:7.4f} "
            f"{values['fps']:7.0f}"
        )

    print("\n=== Checkpoint-window summaries ===")
    print(
        "checkpoint  reward_mean reward_std ep_len  xy_err  yaw_err "
        "timeout bad_ori terrain noise"
    )
    windows = []
    interval = args.checkpoint_interval
    for end in range(interval, total + 1, interval):
        start = end - interval
        checkpoint = total - 1 if end == total else end
        result = {
            "checkpoint": checkpoint,
            "reward": mean(data["reward"][start:end]),
            "reward_std": statistics.pstdev(data["reward"][start:end]),
            "episode_len": mean(data["episode_len"][start:end]),
            "vel_xy_err": mean(data["vel_xy_err"][start:end]),
            "yaw_err": mean(data["yaw_err"][start:end]),
            "time_out": mean(data["time_out"][start:end]),
            "bad_orient": mean(data["bad_orient"][start:end]),
            "terrain": mean(data["terrain"][start:end]),
            "noise": mean(data["noise"][start:end]),
        }
        windows.append(result)
        print(
            f"model_{checkpoint:<4d} "
            f"{result['reward']:11.3f} "
            f"{result['reward_std']:10.3f} "
            f"{result['episode_len']:7.2f} "
            f"{result['vel_xy_err']:7.4f} "
            f"{result['yaw_err']:7.4f} "
            f"{result['time_out']:7.4f} "
            f"{result['bad_orient']:7.4f} "
            f"{result['terrain']:7.4f} "
            f"{result['noise']:7.4f}"
        )

    tail_start = max(0, total - args.tail_size)
    print(f"\n=== Final {total - tail_start} iteration trends ===")
    for name, values in data.items():
        tail = values[tail_start:]
        print(
            f"{name:13s} "
            f"mean={mean(tail):10.5f} "
            f"std={statistics.pstdev(tail):10.5f} "
            f"min={min(tail):10.5f} "
            f"max={max(tail):10.5f} "
            f"trend_per_100={slope_per_100(tail):+10.5f}"
        )

    print("\n=== Top checkpoint windows by mean reward ===")
    for result in sorted(
        windows,
        key=lambda item: item["reward"],
        reverse=True,
    )[:10]:
        print(
            f"model_{result['checkpoint']:<4d} "
            f"reward={result['reward']:7.3f} "
            f"xy_err={result['vel_xy_err']:7.4f} "
            f"yaw_err={result['yaw_err']:7.4f} "
            f"bad_ori={result['bad_orient']:7.4f} "
            f"terrain={result['terrain']:7.4f}"
        )


if __name__ == "__main__":
    main()
