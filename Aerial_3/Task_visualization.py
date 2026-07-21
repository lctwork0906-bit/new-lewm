# -*- coding: utf-8 -*-
"""Visualize an evaluated UAV trajectory with metric-correct geometry."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_DIR = PACKAGE_ROOT / "log_5_DownTown/success_DownTown.json/task_9"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize one Aerial task")
    parser.add_argument(
        "task_dir",
        nargs="?",
        default=str(DEFAULT_TASK_DIR),
        help="Task directory containing object_description.json and log/trajectory.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path; defaults to Aerial_3/image/<task_name>.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window after saving the image",
    )
    return parser.parse_args()


def load_task(task_dir):
    object_path = task_dir / "object_description.json"
    trajectory_path = task_dir / "log/trajectory.jsonl"
    if not object_path.is_file():
        raise FileNotFoundError("Missing object description: {}".format(object_path))
    if not trajectory_path.is_file():
        raise FileNotFoundError("Missing trajectory: {}".format(trajectory_path))

    with object_path.open("r", encoding="utf-8") as stream:
        description = json.load(stream)

    positions = []
    directions = []
    frames = []
    actions = []
    with trajectory_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            state = record["sensors"]["state"]
            positions.append(state["position"])
            # AirSim and scipy both use quaternion order [x, y, z, w].
            directions.append(R.from_quat(state["quaternionr"]).apply([1, 0, 0]))
            frames.append(record["frame"])
            actions.append(record.get("action"))

    if not positions:
        raise ValueError("Trajectory is empty: {}".format(trajectory_path))

    return (
        description,
        np.asarray(positions, dtype=np.float64),
        np.asarray(directions, dtype=np.float64),
        np.asarray(frames, dtype=np.int64),
        actions,
    )


def to_plot_coordinates(world_points, origin_xy):
    """Translate XY to the actual start and flip AirSim's downward-positive Z."""
    points = np.asarray(world_points, dtype=np.float64).copy()
    points[..., 0] -= origin_xy[0]
    points[..., 1] -= origin_xy[1]
    points[..., 2] *= -1.0
    return points


def set_metric_equal_3d(ax, points, padding=0.06):
    """Use the same rendered scale for one metre on every 3D axis."""
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    largest_span = max(float(np.max(maximum - minimum)), 1.0)
    absolute_padding = largest_span * padding
    lower = minimum - absolute_padding
    upper = maximum + absolute_padding
    span = np.maximum(upper - lower, 1.0)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    # 恢复最初版本：盒体三边与实际米制跨度成比例，三轴上的1米视觉长度一致。
    ax.set_box_aspect(tuple(span))


def annotate_point(ax, point, text, color, offset=(0.6, 0.6, 0.6)):
    ax.text(
        point[0] + offset[0],
        point[1] + offset[1],
        point[2] + offset[2],
        text,
        color=color,
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.18", fc="white", ec=color,
            lw=0.8, alpha=0.88,
        ),
    )


def visualize(task_dir, output_path, show=False):
    description, world_trajectory, world_directions, frames, _ = load_task(task_dir)
    target_world = np.asarray(description["pose"][0], dtype=np.float64)

    # 采用轨迹第0帧传感器返回的实际起点（actual start）作为绘图原点。
    # 它与数据集中的理论起点可能因AirSim复位后的物理稳定过程存在较小位置误差。
    # 图中仍统一标记为“Start”，避免引入两个容易混淆的起点概念。
    actual_start_world = world_trajectory[0]
    origin_xy = actual_start_world[:2]
    trajectory = to_plot_coordinates(world_trajectory, origin_xy)
    target = to_plot_coordinates(target_world, origin_xy)
    start = trajectory[0]
    final = trajectory[-1]

    directions = world_directions.copy()
    directions[:, 2] *= -1.0

    distances_3d = np.linalg.norm(world_trajectory - target_world, axis=1)
    distances_xy = np.linalg.norm(
        world_trajectory[:, :2] - target_world[:2], axis=1
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.fontsize": 8.5,
            "grid.color": "#B8B8B8",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.35,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0))
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_distance = fig.add_subplot(grid[1, 1])

    # Metric-correct 3D view. Orthographic projection avoids perspective shrinkage.
    ax_3d.set_proj_type("ortho")
    ax_3d.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color="#315B8A", linewidth=1.9, label="UAV trajectory",
    )
    ax_3d.scatter(*start, color="#2E7D32", s=42, label="Start", depthshade=False)
    ax_3d.scatter(*target, color="#C62828", s=46, label="Target", depthshade=False)
    ax_3d.scatter(*final, color="#6A3D9A", s=42, label="Final", depthshade=False)
    ax_3d.plot(
        [start[0], target[0]], [start[1], target[1]], [start[2], target[2]],
        color="#7A7A7A", linestyle="--", linewidth=1.0, label="Start-target distance",
    )
    ax_3d.plot(
        [final[0], target[0]], [final[1], target[1]], [final[2], target[2]],
        color="#A61C1C", linestyle=":", linewidth=1.7, label="Final-target distance",
    )
    # Sparse orientation arrows preserve readability and still show heading changes.
    arrow_stride = max(1, len(trajectory) // 14)
    arrow_indices = np.arange(0, len(trajectory), arrow_stride)
    if arrow_indices[-1] != len(trajectory) - 1:
        arrow_indices = np.append(arrow_indices, len(trajectory) - 1)
    ax_3d.quiver(
        trajectory[arrow_indices, 0],
        trajectory[arrow_indices, 1],
        trajectory[arrow_indices, 2],
        directions[arrow_indices, 0],
        directions[arrow_indices, 1],
        directions[arrow_indices, 2],
        color="#D4841C",
        length=1.8,
        normalize=True,
        linewidth=0.8,
        alpha=0.72,
    )

    annotate_point(ax_3d, start, "Start", "green")
    annotate_point(ax_3d, target, "Target", "red")
    annotate_point(ax_3d, final, "Final", "purple")
    plot_points = np.vstack((trajectory, target[None, :]))
    set_metric_equal_3d(ax_3d, plot_points)
    ax_3d.view_init(elev=24, azim=-58)
    ax_3d.set_xlabel("X from start (m)")
    ax_3d.set_ylabel("Y from start (m)")
    ax_3d.set_zlabel("Height (m)", labelpad=3)
    ax_3d.set_title("(a)  3D trajectory (orthographic, equal metric scale)", pad=8)
    ax_3d.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.94),
        frameon=False, handlelength=2.4,
    )
    for axis in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))

    # The XY view is the unambiguous horizontal navigation geometry.
    color_values = np.arange(len(trajectory))
    ax_xy.plot(trajectory[:, 0], trajectory[:, 1], color="#315B8A", linewidth=1.6)
    scatter = ax_xy.scatter(
        trajectory[:, 0], trajectory[:, 1], c=color_values,
        cmap="viridis", s=15, edgecolors="none", zorder=3,
    )
    ax_xy.scatter(start[0], start[1], color="#2E7D32", s=38, zorder=4, label="Start")
    ax_xy.scatter(target[0], target[1], color="#C62828", s=42, zorder=4, label="Target")
    ax_xy.scatter(final[0], final[1], color="#6A3D9A", s=38, zorder=4, label="Final")
    ax_xy.plot(
        [final[0], target[0]], [final[1], target[1]],
        color="#A61C1C", linestyle=":", linewidth=1.7,
    )
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.margins(0.12)
    ax_xy.grid(True)
    ax_xy.set_xlabel("X from start (m)")
    ax_xy.set_ylabel("Y from start (m)")
    ax_xy.set_title("(b)  Top-down trajectory (equal metric scale)", pad=7)
    ax_xy.legend(loc="best", frameon=False, ncol=3, handletextpad=0.35, columnspacing=0.8)
    colorbar = fig.colorbar(scatter, ax=ax_xy, pad=0.018, shrink=0.82, aspect=25)
    colorbar.set_label("Frame order", fontsize=9)
    colorbar.ax.tick_params(labelsize=8, direction="in")

    # A distance curve prevents perspective from being mistaken for progress.
    ax_distance.plot(frames, distances_3d, color="#A61C1C", linewidth=1.9, label="3D distance")
    ax_distance.plot(frames, distances_xy, color="#315B8A", linestyle="--", linewidth=1.5, label="XY distance")
    ax_distance.scatter(frames[0], distances_3d[0], color="#2E7D32", s=30, zorder=3)
    ax_distance.scatter(frames[-1], distances_3d[-1], color="#6A3D9A", s=30, zorder=3)
    ax_distance.annotate(
        "{:.2f} m".format(distances_3d[0]),
        (frames[0], distances_3d[0]), xytext=(7, 6), textcoords="offset points",
    )
    ax_distance.annotate(
        "{:.2f} m".format(distances_3d[-1]),
        (frames[-1], distances_3d[-1]), xytext=(-42, 8), textcoords="offset points",
    )
    ax_distance.set_xlabel("Frame")
    ax_distance.set_ylabel("Distance to target (m)")
    ax_distance.set_title("(c)  Target-distance progression", pad=7)
    ax_distance.grid(True)
    ax_distance.legend(loc="best", frameon=False)
    for axis in (ax_xy, ax_distance):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    episode = description.get("episode_id", task_dir.name.replace("task_", ""))
    fig.suptitle(
        "Task {}  |  3D target distance: {:.2f} m -> {:.2f} m  |  XY: {:.2f} m -> {:.2f} m".format(
            episode,
            distances_3d[0],
            distances_3d[-1],
            distances_xy[0],
            distances_xy[-1],
        ),
        fontsize=13,
        fontweight="semibold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
    print("Image saved to: {}".format(output_path))
    print("3D distance: {:.2f} m -> {:.2f} m".format(distances_3d[0], distances_3d[-1]))
    print("XY distance: {:.2f} m -> {:.2f} m".format(distances_xy[0], distances_xy[-1]))
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    task_dir = Path(args.task_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else PACKAGE_ROOT / "image/{}.png".format(task_dir.name)
    )
    visualize(task_dir, output_path, show=args.show)


if __name__ == "__main__":
    main()
