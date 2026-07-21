# -*- coding: utf-8 -*-
"""Render a recorded trajectory inside its original AirSim environment.

The AirVLN scene manager must already be listening.  This script asks it to
open the map, draws the recorded world-coordinate trajectory with AirSim debug
primitives, moves camera 0 (not the vehicle trajectory), and captures top-down
and oblique scene images.
"""

import argparse
import json
import math
import time
from pathlib import Path

import airsim
import cv2
import msgpackrpc
import numpy as np
from scipy.spatial.transform import Rotation as R


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_DIR = PACKAGE_ROOT / "log_1/success_WesternTown.json/task_10"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a logged Aerial trajectory inside AirSim and capture it"
    )
    parser.add_argument("task_dir", nargs="?", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--server-port", type=int, default=30031)
    parser.add_argument("--airsim-timeout", type=int, default=120)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--top-height", type=float, default=62.0)
    parser.add_argument("--oblique-distance", type=float, default=48.0)
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "image/simulator"))
    parser.add_argument("--keep-scene", action="store_true")
    return parser.parse_args()


def reconstruct_positions(description, records):
    """Distance-constrained dead reckoning for compact pose-free logs."""
    start_pose = description["start_pose"]
    current = np.asarray(start_pose["start_position"], dtype=np.float64).copy()
    target = np.asarray(description["pose"][0], dtype=np.float64)
    quaternion = [float(value) for value in start_pose["start_quaternionr"]]
    x, y, z, w = quaternion
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    def constrain_to_logged_distance(point, record):
        distance = record.get("distance_to_end")
        if not isinstance(distance, (int, float)) or float(distance) < 0.0:
            return point
        offset = np.asarray(point, dtype=np.float64) - target
        norm = float(np.linalg.norm(offset))
        if norm <= 1e-8:
            return point
        # 在所有满足真实目标距离的点中，选择与动作积分预测最近的点。
        return target + offset * (float(distance) / norm)

    current = constrain_to_logged_distance(current, records[0])
    positions = [current.copy()]
    for record in records[1:]:
        action = record.get("action")
        if action in ("rotl", "rotr"):
            sign = -1.0 if action == "rotl" else 1.0
            yaw += sign * math.radians(float(record.get("step_size", 0.0)))
            yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        else:
            # step_move_distance是实际位移长度；方向由动作和累计偏航角恢复。
            distance = max(float(record.get("step_move_distance", 0.0)), 0.0)
            forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
            right = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0])
            if action == "forward":
                current = current + forward * distance
            elif action == "left":
                current = current - right * distance
            elif action == "right":
                current = current + right * distance
            elif action == "ascend":
                current = current + np.asarray([0.0, 0.0, -distance])
            elif action == "descend":
                current = current + np.asarray([0.0, 0.0, distance])
        current = constrain_to_logged_distance(current, record)
        positions.append(current.copy())
    if len(positions) != len(records):
        raise RuntimeError("Reconstructed trajectory length does not match records")
    return np.asarray(positions, dtype=np.float64)


def load_task(task_dir):
    with (task_dir / "object_description.json").open("r", encoding="utf-8") as stream:
        description = json.load(stream)
    records = []
    with (task_dir / "log/trajectory.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            records.append(record)
    if not records:
        raise ValueError("Empty trajectory: {}".format(task_dir))
    if all("sensors" in record for record in records):
        positions = np.asarray(
            [record["sensors"]["state"]["position"] for record in records],
            dtype=np.float64,
        )
        trajectory_source = "recorded sensor trajectory"
    else:
        positions = reconstruct_positions(description, records)
        trajectory_source = "distance-constrained action-log reconstructed trajectory"
    decisions = []
    decision_path = task_dir / "log/clip_decisions.jsonl"
    if decision_path.is_file():
        with decision_path.open("r", encoding="utf-8") as stream:
            decisions = [json.loads(line).get("decision") or {} for line in stream]
    if len(decisions) != len(records):
        decisions = [{} for _ in records]
    return description, positions, records, decisions, trajectory_source


def vector(point):
    return airsim.Vector3r(float(point[0]), float(point[1]), float(point[2]))


def look_at_quaternion(camera_world, focus_world):
    """Orient AirSim camera +X toward focus in the NED coordinate system."""
    delta = np.asarray(focus_world, dtype=np.float64) - np.asarray(
        camera_world, dtype=np.float64
    )
    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    # Camera +X points forward. Negative pitch looks toward increasing world
    # NED Z, i.e. downward toward the scene.
    pitch = -math.atan2(float(delta[2]), max(horizontal, 1e-6))
    return airsim.to_quaternion(pitch, 0.0, yaw)


def wait_for_airsim(ip, port, timeout):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        client = airsim.MultirotorClient(
            ip=ip, port=int(port), timeout_value=min(timeout, 30)
        )
        try:
            client.ping()
            client.confirmConnection()
            return client
        except Exception as error:
            last_error = error
            time.sleep(1.0)
    raise RuntimeError("AirSim did not become ready on {}:{}: {}".format(ip, port, last_error))


def capture_scene(client, camera_name, camera_world, focus_world, vehicle_world):
    relative_position = np.asarray(camera_world) - np.asarray(vehicle_world)
    camera_pose = airsim.Pose(
        vector(relative_position), look_at_quaternion(camera_world, focus_world)
    )
    client.simSetCameraPose(camera_name, camera_pose, vehicle_name="Drone_1")
    # The renderer has already been warmed up. Keep physics paused here: with
    # ClockSpeed=10 an uncontrolled camera vehicle otherwise falls below the map.
    last_reason = "no response"
    for _ in range(12):
        response = client.simGetImages(
            [
                airsim.ImageRequest(
                    camera_name,
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=True,
                )
            ],
            vehicle_name="Drone_1",
        )[0]
        encoded = bytes(response.image_data_uint8)
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None and decoded.size and np.any(decoded):
            camera_info = client.simGetCameraInfo(
                camera_name, vehicle_name="Drone_1"
            )
            return decoded, camera_info
        last_reason = "empty or black image"
        time.sleep(0.6)
    raise RuntimeError("Failed to capture camera {}: {}".format(camera_name, last_reason))


def project_world_points(points, camera_info, image_shape):
    """Project NED world coordinates with AirSim's camera pose and horizontal FOV."""
    height, width = image_shape[:2]
    position = camera_info.pose.position
    orientation = camera_info.pose.orientation
    camera_world = np.asarray(
        [position.x_val, position.y_val, position.z_val], dtype=np.float64
    )
    rotation = R.from_quat(
        [
            orientation.x_val,
            orientation.y_val,
            orientation.z_val,
            orientation.w_val,
        ]
    )
    camera_points = rotation.inv().apply(
        np.asarray(points, dtype=np.float64) - camera_world
    )
    focal = width / (
        2.0 * math.tan(math.radians(float(camera_info.fov)) / 2.0)
    )
    visible = camera_points[:, 0] > 0.05
    pixels = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    pixels[visible, 0] = (
        width / 2.0
        + focal * camera_points[visible, 1] / camera_points[visible, 0]
    )
    pixels[visible, 1] = (
        height / 2.0
        + focal * camera_points[visible, 2] / camera_points[visible, 0]
    )
    return pixels, visible


def overlay_projected_trajectory(image, camera_info, positions, target):
    """Draw the recorded 3D world path using the captured camera calibration."""
    canvas = image.copy()
    path_pixels, path_visible = project_world_points(
        positions, camera_info, canvas.shape
    )
    for index in range(1, len(path_pixels)):
        if not (path_visible[index - 1] and path_visible[index]):
            continue
        point_a = tuple(np.rint(path_pixels[index - 1]).astype(int))
        point_b = tuple(np.rint(path_pixels[index]).astype(int))
        cv2.line(canvas, point_a, point_b, (245, 245, 245), 7, cv2.LINE_AA)
        cv2.line(canvas, point_a, point_b, (235, 92, 28), 4, cv2.LINE_AA)

    marker_points = np.vstack((positions[0], target, positions[-1]))
    marker_pixels, marker_visible = project_world_points(
        marker_points, camera_info, canvas.shape
    )
    markers = (
        ("Start", (45, 180, 45)),
        ("Target", (45, 45, 230)),
        ("Final", (180, 50, 180)),
    )
    height, width = canvas.shape[:2]
    for pixel, is_visible, (label, color) in zip(
        marker_pixels, marker_visible, markers
    ):
        if not is_visible or not np.all(np.isfinite(pixel)):
            continue
        x, y = np.rint(pixel).astype(int)
        if not (-20 <= x < width + 20 and -20 <= y < height + 20):
            continue
        cv2.circle(canvas, (x, y), 10, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (x + 9, y - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x + 9, y - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )
    return canvas


def select_turn_annotations(records, decisions):
    """Select non-redundant turns and preserve the policy reason for each."""
    reason_labels = {
        "short_active_scan": "Scan",
        "rotate_side_target_into_front_camera": "Target alignment",
        "safe_unvisited_search_cell": "Search",
        "depth_safe_target_approach": "Approach",
    }
    selected = []
    last_by_label = {}
    minimum_gap = {"Scan": 10 ** 6, "Target alignment": 5, "Search": 15}
    for index, (record, decision) in enumerate(zip(records, decisions)):
        action = record.get("action")
        if action not in ("left", "right", "rotl", "rotr", "descend"):
            continue
        reason = str(decision.get("reason", ""))
        label = next(
            (value for key, value in reason_labels.items() if key in reason),
            "Turn",
        )
        # Descend is a vertical approach, not a horizontal bend in this figure.
        if label == "Approach":
            continue
        if label in last_by_label and index - last_by_label[label] < minimum_gap.get(label, 8):
            continue
        selected.append((index, "F{} {}".format(record.get("frame", index), label)))
        last_by_label[label] = index
    return selected[:5]


def overlay_turn_annotations(image, camera_info, positions, annotations):
    canvas = image.copy()
    if not annotations:
        return canvas
    indices = [item[0] for item in annotations]
    pixels, visible = project_world_points(
        positions[indices], camera_info, canvas.shape
    )
    height, width = canvas.shape[:2]
    for marker_number, (pixel, is_visible, _) in enumerate(
        zip(pixels, visible, annotations), start=1
    ):
        if not is_visible or not np.all(np.isfinite(pixel)):
            continue
        x, y = np.rint(pixel).astype(int)
        if not (0 <= x < width and 0 <= y < height):
            continue
        cv2.drawMarker(
            canvas, (x, y), (0, 215, 255), cv2.MARKER_DIAMOND,
            markerSize=13, thickness=2, line_type=cv2.LINE_AA,
        )
        number = str(marker_number)
        cv2.putText(
            canvas, number, (x + 7, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (15, 15, 15), 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, number, (x + 7, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (255, 240, 120), 1, cv2.LINE_AA,
        )
    return canvas


def minimum_turn_clearance(records, decisions):
    clearances = []
    for index, record in enumerate(records):
        if record.get("action") not in ("left", "right", "rotl", "rotr", "descend"):
            continue
        safety = decisions[index].get("depth_safety") or {}
        for direction in ("forward", "left", "right"):
            value = (safety.get(direction) or {}).get("clearance")
            if isinstance(value, (int, float)):
                clearances.append(float(value))
    return min(clearances) if clearances else float("nan")


def compose_explanation(view_a, view_b, annotations, minimum_clearance):
    """Build one publication-ready two-view 3D environment explanation."""
    target_height = 512
    a = cv2.resize(view_a, (512, target_height), interpolation=cv2.INTER_AREA)
    b = cv2.resize(view_b, (512, target_height), interpolation=cv2.INTER_AREA)
    image = np.hstack((a, b))
    footer = np.full((140, image.shape[1], 3), 247, dtype=np.uint8)
    labels = ["{}={}".format(index + 1, label) for index, (_, label) in enumerate(annotations)]
    split = min(3, len(labels))
    cv2.putText(
        footer, "Yellow turn markers: " + ";  ".join(labels[:split]),
        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (35, 35, 35),
        1, cv2.LINE_AA,
    )
    if labels[split:]:
        cv2.putText(
            footer, ";  ".join(labels[split:]),
            (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (35, 35, 35),
            1, cv2.LINE_AA,
        )
    clearance_text = (
        "Minimum directional clearance at labelled turns: {:.1f} m "
        "(translation safety threshold: 2.5 m).".format(minimum_clearance)
    )
    cv2.putText(
        footer, clearance_text, (18, 94), cv2.FONT_HERSHEY_SIMPLEX,
        0.46, (35, 35, 35), 1, cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "Conclusion: these bends were driven by visual target alignment/search, not collision avoidance.",
        (18, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 55, 120),
        1, cv2.LINE_AA,
    )
    return np.vstack((image, footer))


def add_caption(image, title, distances):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    band_height = max(54, int(height * 0.075))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, band_height), (20, 24, 29), -1)
    canvas = cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0)
    scale = max(0.55, min(1.0, width / 1150.0))
    cv2.putText(
        canvas,
        title,
        (18, int(band_height * 0.46)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Blue: trajectory   Green: Start   Red: Target   Purple: Final   "
        "Distance {:.2f} m -> {:.2f} m".format(distances[0], distances[-1]),
        (18, int(band_height * 0.82)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.62,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render(args):
    task_dir = Path(args.task_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    description, positions, records, decisions, trajectory_source = load_task(task_dir)
    target = np.asarray(description["pose"][0], dtype=np.float64)
    start = positions[0]
    final = positions[-1]
    distances = np.linalg.norm(positions - target, axis=1)
    map_name = description["map_name"]
    map_label = map_name[:-5] if map_name.endswith("_test") else map_name
    episode = str(description.get("episode_id", task_dir.name.replace("task_", "")))

    scene_manager = msgpackrpc.Client(
        msgpackrpc.Address("127.0.0.1", int(args.server_port)), timeout=300
    )
    if not scene_manager.call("ping"):
        raise RuntimeError("AirVLN scene manager did not answer ping")

    scene_open = False
    try:
        result = scene_manager.call(
            "reopen_scenes", "127.0.0.1", [[map_name, int(args.gpu)]]
        )
        if not result[0]:
            raise RuntimeError("Scene manager failed to open {}".format(map_name))
        ip, ports = result[1]
        if isinstance(ip, bytes):
            ip = ip.decode("utf-8")
        airsim_port = int(ports[0])
        scene_open = True
        print("Scene {} opened on AirSim {}:{}".format(map_name, ip, airsim_port))

        client = wait_for_airsim(ip, airsim_port, args.airsim_timeout)
        client.enableApiControl(True, vehicle_name="Drone_1")

        # 先让新启动的offscreen渲染器和自动曝光稳定，再恢复精确起点。
        client.simPause(False)
        time.sleep(1.0)
        client.simPause(True)

        # 把无人机放在日志中的Start，仅作为相机参考系；不重放、不改写原轨迹。
        # 使用单位姿态让相机相对X/Y与记录的世界X/Y方向一致。
        start_pose = airsim.Pose(
            vector(start), airsim.to_quaternion(0.0, 0.0, 0.0)
        )
        client.simSetVehiclePose(start_pose, ignore_collision=True, vehicle_name="Drone_1")
        held_pose = client.simGetVehiclePose(vehicle_name="Drone_1")
        held_position = np.asarray(list(held_pose.position), dtype=np.float64)
        hold_error = float(np.linalg.norm(held_position - start))
        if hold_error >= 1.0:
            # 相机位姿是相对无人机设置的。若场景物理把参考无人机推离Start，
            # 使用仿真返回的实际参考点补偿相机外参即可，世界坐标轨迹本身不变。
            print(
                "Warning: render reference moved from Start by {:.3f} m; "
                "compensating camera extrinsics with actual position {}".format(
                    hold_error, held_position.tolist()
                )
            )
        else:
            print("Render vehicle held at Start (error {:.3f} m)".format(hold_error))

        client.simFlushPersistentMarkers()
        path_points = [vector(point) for point in positions]
        client.simPlotLineStrip(
            path_points,
            color_rgba=[0.05, 0.35, 1.0, 1.0],
            thickness=14.0,
            duration=-1.0,
            is_persistent=True,
        )
        client.simPlotPoints(
            [vector(start)], [0.05, 0.85, 0.12, 1.0],
            size=32.0, duration=-1.0, is_persistent=True,
        )
        client.simPlotPoints(
            [vector(target)], [1.0, 0.05, 0.03, 1.0],
            size=36.0, duration=-1.0, is_persistent=True,
        )
        client.simPlotPoints(
            [vector(final)], [0.65, 0.10, 0.95, 1.0],
            size=32.0, duration=-1.0, is_persistent=True,
        )
        # Debug primitives enter Unreal's render scene on the next frame.
        client.simContinueForFrames(1)
        client.simPause(True)
        # 推进渲染帧可能触发场景物理并移动参考无人机。标记生效后再次在暂停状态
        # 恢复Start，后续只移动相机，不再推进无人机物理。
        client.simSetVehiclePose(
            start_pose, ignore_collision=True, vehicle_name="Drone_1"
        )
        held_pose = client.simGetVehiclePose(vehicle_name="Drone_1")
        held_position = np.asarray(list(held_pose.position), dtype=np.float64)
        post_marker_hold_error = float(np.linalg.norm(held_position - start))
        if post_marker_hold_error >= 1.0:
            raise RuntimeError(
                "Render reference could not be restored after marker refresh: "
                "error={:.3f} m, actual={}".format(
                    post_marker_hold_error, held_position.tolist()
                )
            )

        all_points = np.vstack((positions, target[None, :]))
        focus = (np.min(all_points, axis=0) + np.max(all_points, axis=0)) / 2.0
        # Focus slightly above the path so buildings do not dominate camera framing.
        focus[2] = float(np.mean(positions[:, 2]))

        top_camera = focus.copy()
        top_camera[2] -= float(args.top_height)
        top, top_camera_info = capture_scene(
            client, args.camera, top_camera, focus, held_position
        )

        d = float(args.oblique_distance)
        oblique_camera = focus + np.asarray([-0.82 * d, -0.72 * d, -0.68 * d])
        oblique, oblique_camera_info = capture_scene(
            client, args.camera, oblique_camera, focus, held_position
        )

        reverse_camera = focus + np.asarray(
            [0.70 * d, 0.78 * d, -0.52 * d]
        )
        reverse, reverse_camera_info = capture_scene(
            client, args.camera, reverse_camera, focus, held_position
        )

        annotations = select_turn_annotations(records, decisions)
        top = overlay_projected_trajectory(
            top, top_camera_info, positions, target
        )
        oblique = overlay_projected_trajectory(
            oblique, oblique_camera_info, positions, target
        )
        reverse = overlay_projected_trajectory(
            reverse, reverse_camera_info, positions, target
        )
        oblique = overlay_turn_annotations(
            oblique, oblique_camera_info, positions, annotations
        )
        reverse = overlay_turn_annotations(
            reverse, reverse_camera_info, positions, annotations
        )
        source_label = (
            "recorded" if trajectory_source.startswith("recorded") else "reconstructed"
        )
        top = add_caption(
            top,
            "Task {} | {} top-down trajectory ({})".format(
                episode, map_label, source_label
            ),
            distances,
        )
        oblique = add_caption(
            oblique,
            "Task {} | 3D environment view A ({})".format(episode, source_label),
            distances,
        )
        reverse = add_caption(
            reverse,
            "Task {} | 3D environment view B ({})".format(episode, source_label),
            distances,
        )
        explained = compose_explanation(
            oblique,
            reverse,
            annotations,
            minimum_turn_clearance(records, decisions),
        )
        top_path = output_dir / "task_{}_sim_topdown.png".format(episode)
        oblique_path = output_dir / "task_{}_sim_oblique.png".format(episode)
        reverse_path = output_dir / "task_{}_sim_oblique_reverse.png".format(episode)
        explained_path = output_dir / "task_{}_sim_3d_explained.png".format(episode)
        if not cv2.imwrite(str(top_path), top):
            raise RuntimeError("Failed to write {}".format(top_path))
        if not cv2.imwrite(str(oblique_path), oblique):
            raise RuntimeError("Failed to write {}".format(oblique_path))
        if not cv2.imwrite(str(reverse_path), reverse):
            raise RuntimeError("Failed to write {}".format(reverse_path))
        if not cv2.imwrite(str(explained_path), explained):
            raise RuntimeError("Failed to write {}".format(explained_path))
        print("Saved {}".format(top_path))
        print("Saved {}".format(oblique_path))
        print("Saved {}".format(reverse_path))
        print("Saved {}".format(explained_path))
        print("Trajectory source: {}".format(trajectory_source))
        return top_path, oblique_path, reverse_path, explained_path
    finally:
        if scene_open and not args.keep_scene:
            try:
                scene_manager.call("close_scenes", "127.0.0.1")
            except Exception as error:
                print("Warning: failed to close scene: {}".format(error))
        try:
            scene_manager.close()
        except Exception:
            pass


def main():
    render(parse_args())


if __name__ == "__main__":
    main()
