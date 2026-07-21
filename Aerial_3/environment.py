import copy
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import airsim
import numpy as np

from .airsim_bridge import AirVLNSimulatorClientTool
from .logging_utils import logger


def _minimum_target_distance(position, target_positions):
    current = np.asarray(position, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    if targets.ndim == 2 and targets.shape[1] == 3:
        return float(np.linalg.norm(targets - current[None, :], axis=1).min())
    return float(np.linalg.norm(targets - current))


def _yaw_from_quaternion(quaternion):
    x, y, z, w = [float(value) for value in quaternion]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pose_from_lists(position, quaternion):
    return airsim.Pose(
        airsim.Vector3r(*[float(value) for value in position]),
        airsim.Quaternionr(
            x_val=float(quaternion[0]),
            y_val=float(quaternion[1]),
            z_val=float(quaternion[2]),
            w_val=float(quaternion[3]),
        ),
    )


def get_next_pose(
    current_pose,
    action,
    step_size,
    horizontal_step,
    vertical_step,
    rotate_angle,
    is_fixed,
):
    position = np.array(
        [
            current_pose.position.x_val,
            current_pose.position.y_val,
            current_pose.position.z_val,
        ],
        dtype=np.float64,
    )
    orientation = current_pose.orientation
    pitch, roll, yaw = airsim.to_eularian_angles(orientation)
    horizontal_distance = horizontal_step if is_fixed else float(step_size)
    vertical_distance = vertical_step if is_fixed else float(step_size)
    turn_degrees = rotate_angle if is_fixed else float(step_size)
    new_position = position.copy()
    new_orientation = orientation
    fly_type = "stop"

    if action == "forward":
        new_position += np.array([math.cos(yaw), math.sin(yaw), 0.0]) * horizontal_distance
        fly_type = "move"
    elif action in ("left", "right"):
        right_vector = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        direction = -1.0 if action == "left" else 1.0
        new_position += direction * right_vector * horizontal_distance
        fly_type = "move"
    elif action in ("ascend", "descend"):
        # AirSim uses NED coordinates: smaller z is higher.
        direction = -1.0 if action == "ascend" else 1.0
        new_position += np.array([0.0, 0.0, direction * vertical_distance])
        fly_type = "move"
    elif action in ("rotl", "rotr"):
        direction = -1.0 if action == "rotl" else 1.0
        new_yaw = yaw + direction * math.radians(turn_degrees)
        new_yaw = (new_yaw + math.pi) % (2.0 * math.pi) - math.pi
        new_orientation = airsim.to_quaternion(pitch, roll, new_yaw)
        fly_type = "rotate"

    return airsim.Pose(
        airsim.Vector3r(*new_position.tolist()),
        airsim.Quaternionr(
            x_val=new_orientation.x_val,
            y_val=new_orientation.y_val,
            z_val=new_orientation.z_val,
            w_val=new_orientation.w_val,
        ),
    ), fly_type


@dataclass
class EpisodeState:
    task: Dict
    step: int = 0
    is_end: bool = False
    oracle_success: bool = False
    is_collisioned: bool = False
    move_distance: float = 0.0
    commanded_move_distance: float = 0.0
    heading_changes: List[float] = field(default_factory=list)
    trajectory: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        start = self.task["start_pose"]
        distance = _minimum_target_distance(
            start["start_position"], self.task["object_position"]
        )
        self.trajectory = [
            {
                "sensors": {
                    "state": {
                        "position": list(start["start_position"]),
                        "quaternionr": list(start["start_quaternionr"]),
                    }
                },
                "move_distance": 0.0,
                "step_move_distance": 0.0,
                "commanded_move_distance": 0.0,
                "commanded_step_distance": 0.0,
                "pose_error": 0.0,
                "orientation_error_deg": 0.0,
                "pose_attempts": 0,
                "execution_status": "episode_start",
                "requested_position": list(start["start_position"]),
                "requested_quaternion": list(start["start_quaternionr"]),
                "distance_to_target": distance,
            }
        ]

    @property
    def position(self):
        return self.trajectory[-1]["sensors"]["state"]["position"]

    @property
    def quaternion(self):
        return self.trajectory[-1]["sensors"]["state"]["quaternionr"]

    @property
    def pose(self):
        return _pose_from_lists(self.position, self.quaternion)


class AerialNavigationEnv:
    """Minimal evaluation environment independent of the legacy ``src`` tree."""

    SUCCESS_DISTANCE = 20.0
    MOVE_RETRY_TOLERANCE = 0.75
    ROTATE_RETRY_TOLERANCE_DEG = 2.0
    ROTATE_POSITION_TOLERANCE = 1.0
    ACTION_POSE_HARD_LIMIT = 5.0
    MOVE_POSE_ATTEMPTS = 3
    ROTATE_POSE_ATTEMPTS = 3

    def __init__(self, args):
        self.args = args
        self.batch_size = int(args.batchSize)
        self.data = self._load_dataset(args.dataset_path)
        self.data = self._group_scenes(self.data)
        self.index_data = 0
        self.batch: List[Dict] = []
        self.states: List[EpisodeState] = []
        self.simulator_tool: Optional[AirVLNSimulatorClientTool] = None
        self.machines_info = []
        self.last_scene_list: List[str] = []

    @staticmethod
    def _load_dataset(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as dataset_file:
            raw_items = json.load(dataset_file)

        tasks = []
        for item in raw_items:
            tasks.append(
                {
                    "map_name": item["map_name"],
                    "object_name": item["true_name"],
                    "asset_name": item.get("object_name", ""),
                    "object_size": item["size"],
                    "object_position": item["pose"],
                    "start_pose": item["start_pose"],
                    "description": item["description"],
                    "distance_to_target": item["info"]["euclidean_distance"],
                    "task_id": str(item["episode_id"]),
                    "dataset_item": item,
                }
            )
        logger.info("Loaded %d evaluation tasks from %s", len(tasks), dataset_path)
        return tasks

    @staticmethod
    def _group_scenes(tasks):
        scene_order = OrderedDict()
        for task in tasks:
            scene_order.setdefault(task["map_name"], len(scene_order))
        return sorted(tasks, key=lambda item: scene_order[item["map_name"]])

    def __len__(self):
        return len(self.data)

    def next_minibatch(self):
        if self.index_data >= len(self.data):
            self.batch = []
            return None
        end = min(self.index_data + self.batch_size, len(self.data))
        self.batch = copy.deepcopy(self.data[self.index_data:end])
        self.index_data = end
        return self.batch

    def _machine_configuration(self, scene_list):
        return [
            {
                "MACHINE_IP": "127.0.0.1",
                "SOCKET_PORT": int(self.args.simulator_tool_port),
                "MAX_SCENE_NUM": 16,
                "open_scenes": list(scene_list),
                "gpus": [int(self.args.gpu_id)] * len(scene_list),
            }
        ]

    def _ensure_scenes(self):
        scenes = [task["map_name"] for task in self.batch]
        if self.simulator_tool is not None and scenes == self.last_scene_list:
            logger.info("Reusing simulator scenes: %s", scenes)
            return

        self.machines_info = self._machine_configuration(scenes)
        logger.info("Opening simulator scenes: %s", scenes)
        last_error = None
        for _ in range(10):
            try:
                self.simulator_tool = AirVLNSimulatorClientTool(self.machines_info)
                self.simulator_tool.run_call()
                self.last_scene_list = list(scenes)
                return
            except Exception as error:
                last_error = error
                logger.error("Failed to open scenes: %s", error)
                time.sleep(3)
        raise RuntimeError("unable to start simulator scenes") from last_error

    def _set_drones(self):
        # The shared client's setPoses() waits in simContinueForFrames(50),
        # which can block indefinitely on an off-screen Unreal instance.  Keep
        # this transport workaround local to Aerial_3: teleport each drone,
        # let the renderer advance briefly in real time, then pause again.
        clients = [
            client
            for machine_clients in self.simulator_tool.airsim_clients
            for client in machine_clients
        ]
        if len(clients) != len(self.batch):
            raise RuntimeError("AirSim client count does not match active tasks")

        for index, (client, task) in enumerate(zip(clients, self.batch)):
            if client is None:
                raise RuntimeError("AirSim returned an empty vehicle client")
            start = task["start_pose"]
            pose = _pose_from_lists(
                start["start_position"], start["start_quaternionr"]
            )
            vehicles = client.listVehicles()
            if vehicles:
                client.simSetObjectScale(
                    vehicles[0], airsim.Vector3r(0.5, 0.5, 0.5)
                )
            requested = np.asarray(start["start_position"], dtype=np.float64)
            # The Unreal/AirSim bridge can occasionally drop the first pose
            # command immediately after renderer warm-up.  Retry the same
            # idempotent reset handshake; never accept a previous episode's
            # position as the new origin.
            for reset_attempt in range(3):
                client.simPause(False)
                client.cancelLastTask()
                client.simSetVehiclePose(pose=pose, ignore_collision=True)
                client.simContinueForFrames(1)
                actual_pose = client.simGetVehiclePose()
                actual = np.array(
                    [
                        actual_pose.position.x_val,
                        actual_pose.position.y_val,
                        actual_pose.position.z_val,
                    ],
                    dtype=np.float64,
                )
                error = float(np.linalg.norm(actual - requested))
                if error < 1.00:
                    break
                logger.warning(
                    "Reset pose attempt %d/3 failed for task %s: error=%.3f m",
                    reset_attempt + 1,
                    task["task_id"],
                    error,
                )
            # A single physics frame may settle a newly teleported vehicle by a
            # few decimetres.  Keep that real pose as the episode origin, while
            # rejecting metre-scale reset failures/cross-task contamination.
            if error >= 1.00:
                raise RuntimeError(
                    "AirSim reset pose integrity failure for task {}: "
                    "error={:.3f} m requested={} actual={}".format(
                        task["task_id"], error, requested.tolist(), actual.tolist()
                    )
                )
            quaternion = actual_pose.orientation
            actual_position = actual.tolist()
            self.states[index].trajectory[0] = {
                "sensors": {
                    "state": {
                        "position": [float(value) for value in actual_position],
                        "quaternionr": [
                            float(quaternion.x_val),
                            float(quaternion.y_val),
                            float(quaternion.z_val),
                            float(quaternion.w_val),
                        ],
                    }
                },
                "move_distance": 0.0,
                "step_move_distance": 0.0,
                "commanded_move_distance": 0.0,
                "commanded_step_distance": 0.0,
                "pose_error": round(error, 6),
                "orientation_error_deg": 0.0,
                "pose_attempts": reset_attempt + 1,
                "execution_status": (
                    "reset_aligned"
                    if reset_attempt == 0
                    else "reset_aligned_after_retry"
                ),
                "requested_position": requested.tolist(),
                "requested_quaternion": list(start["start_quaternionr"]),
                "distance_to_target": _minimum_target_distance(
                    actual_position, task["object_position"]
                ),
            }

    def reset(self):
        if not self.batch:
            raise RuntimeError("next_minibatch must be called before reset")
        self._ensure_scenes()
        self.states = [EpisodeState(task=task) for task in self.batch]
        self._set_drones()
        # Renderer warm-up temporarily advances real-time physics.  Discard
        # that image, cancel any residual controller task, restore the exact
        # dataset start pose, and only then expose the first observation.
        self.get_obs(warmup_renderer=True)
        self._set_drones()
        return self.get_obs(warmup_renderer=False)

    def _flatten_image_responses(self, responses):
        if responses is None:
            raise RuntimeError("AirSim returned no image responses")
        flattened = []
        for machine_responses in responses:
            flattened.extend(machine_responses)
        if len(flattened) != len(self.states):
            raise RuntimeError(
                "image response count does not match active batch: {} != {}".format(
                    len(flattened), len(self.states)
                )
            )
        return flattened

    def _observation(self, index, rgb_images, depth_images):
        state = self.states[index]
        task = state.task
        observation = copy.deepcopy(state.trajectory[-1])
        observation.update(
            {
                "task_id": task["task_id"],
                "description": task["description"],
                "object_name": task["object_name"],
                "object_size": task["object_size"],
                "rgb": rgb_images,
                "depth": depth_images,
                "pre_poses": [
                    item["sensors"]["state"] for item in state.trajectory[-10:]
                ],
                "step": state.step,
                "move_distance": state.move_distance,
                "start_position": state.trajectory[0]["sensors"]["state"][
                    "position"
                ],
                "start_quaternionr": state.trajectory[0]["sensors"]["state"][
                    "quaternionr"
                ],
                "avg_heading_changes": (
                    float(np.mean(state.heading_changes))
                    if state.heading_changes
                    else 0.0
                ),
            }
        )
        return observation

    def get_obs(self, warmup_renderer=False):
        responses = self._flatten_image_responses(
            self.simulator_tool.getImageResponses(
                warmup_renderer=warmup_renderer
            )
        )
        outputs = []
        for index, response in enumerate(responses):
            rgb_images, depth_images = response
            state = self.states[index]
            outputs.append(
                (
                    self._observation(index, rgb_images, depth_images),
                    state.is_end,
                    state.is_collisioned,
                    state.oracle_success,
                )
            )
        return outputs

    @staticmethod
    def _actual_pose(move_result, fallback_pose):
        try:
            sensor_state = move_result["states"][-1]["sensors"]["state"]
            position = sensor_state["position"]
            quaternion = sensor_state.get("orientation")
            if quaternion is None:
                quaternion = sensor_state.get("quaternionr")
            if quaternion is None or len(position) != 3 or len(quaternion) != 4:
                raise ValueError("incomplete AirSim state")
            return list(position), list(quaternion)
        except (KeyError, IndexError, TypeError, ValueError):
            return (
                [
                    fallback_pose.position.x_val,
                    fallback_pose.position.y_val,
                    fallback_pose.position.z_val,
                ],
                [
                    fallback_pose.orientation.x_val,
                    fallback_pose.orientation.y_val,
                    fallback_pose.orientation.z_val,
                    fallback_pose.orientation.w_val,
                ],
            )

    def make_actions(self, actions: Sequence[str], step_sizes: Sequence[float]):
        if len(actions) != len(self.states) or len(step_sizes) != len(self.states):
            raise ValueError("one action and step size are required per active episode")

        requested_actions = list(actions)
        was_ended = [state.is_end for state in self.states]
        target_poses = []
        fly_types = []
        previous_yaws = []
        for index, state in enumerate(self.states):
            action = "stop" if state.is_end else requested_actions[index]
            previous_yaws.append(_yaw_from_quaternion(state.quaternion))
            target_pose, fly_type = get_next_pose(
                state.pose,
                action,
                step_sizes[index],
                horizontal_step=self.args.xOy_step_size,
                vertical_step=self.args.z_step_size,
                rotate_angle=self.args.rotateAngle,
                is_fixed=self.args.is_fixed,
            )
            target_poses.append(target_pose)
            fly_types.append(fly_type)

        # ``AirVLNSimulatorClientTool.move_to_next_pose`` advances 150 Unreal
        # frames for every two-metre action.  On the off-screen evaluation
        # renderer that costs roughly five wall-clock seconds and, because the
        # async move is not joined, can also leave the vehicle between grid
        # cells.  Aerial_3 uses fixed discrete actions, so execute those
        # actions as collision-aware pose updates instead.  Depth has already
        # certified the complete step plus the configured safety margin.
        clients = [
            client
            for machine_clients in self.simulator_tool.airsim_clients
            for client in machine_clients
        ]
        if len(clients) != len(self.states):
            raise RuntimeError("AirSim client count does not match active states")

        move_results = [[]]
        for client, target_pose, fly_type in zip(
            clients, target_poses, fly_types
        ):
            before_collision = client.simGetCollisionInfo()
            after_collision = before_collision
            pose_attempts = 0
            if fly_type == "move":
                # AirSim occasionally drops a pose command after the renderer
                # has been queried.  Retry only when the measured endpoint is
                # outside the round-4 P95-derived 0.75 m alignment band.  Each
                # attempt still uses collision-aware placement and one fixed
                # physics frame, so retries do not bypass obstacle contacts.
                for attempt in range(self.MOVE_POSE_ATTEMPTS):
                    pose_attempts = attempt + 1
                    client.simPause(False)
                    client.cancelLastTask()
                    client.simSetVehiclePose(
                        pose=target_pose,
                        ignore_collision=False,
                    )
                    client.simContinueForFrames(1)
                    after_collision = client.simGetCollisionInfo()
                    collision = bool(
                        after_collision.has_collided
                        and after_collision.time_stamp
                        > before_collision.time_stamp
                    )
                    actual_pose = client.simGetVehiclePose()
                    actual_position = np.array(
                        [
                            actual_pose.position.x_val,
                            actual_pose.position.y_val,
                            actual_pose.position.z_val,
                        ],
                        dtype=np.float64,
                    )
                    target_position = np.array(
                        [
                            target_pose.position.x_val,
                            target_pose.position.y_val,
                            target_pose.position.z_val,
                        ],
                        dtype=np.float64,
                    )
                    attempt_error = float(
                        np.linalg.norm(actual_position - target_position)
                    )
                    if collision or attempt_error <= self.MOVE_RETRY_TOLERANCE:
                        break
            elif fly_type == "rotate":
                # Letting physics run for an arbitrary wall-clock sleep caused
                # 0.3-4 m drift even for yaw-only actions and made path/success
                # metrics depend on host load.  Conversely, freezing
                # immediately can drop the yaw command entirely.  Commit one
                # deterministic frame and retry only when yaw or position is
                # outside the explicit alignment bands.
                target_quaternion_for_retry = [
                    target_pose.orientation.x_val,
                    target_pose.orientation.y_val,
                    target_pose.orientation.z_val,
                    target_pose.orientation.w_val,
                ]
                target_position_for_retry = np.array(
                    [
                        target_pose.position.x_val,
                        target_pose.position.y_val,
                        target_pose.position.z_val,
                    ],
                    dtype=np.float64,
                )
                for attempt in range(self.ROTATE_POSE_ATTEMPTS):
                    pose_attempts = attempt + 1
                    client.simPause(False)
                    client.cancelLastTask()
                    client.simSetVehiclePose(
                        pose=target_pose,
                        ignore_collision=True,
                    )
                    client.simContinueForFrames(1)
                    actual_pose = client.simGetVehiclePose()
                    actual_position_for_retry = np.array(
                        [
                            actual_pose.position.x_val,
                            actual_pose.position.y_val,
                            actual_pose.position.z_val,
                        ],
                        dtype=np.float64,
                    )
                    actual_quaternion_for_retry = [
                        actual_pose.orientation.x_val,
                        actual_pose.orientation.y_val,
                        actual_pose.orientation.z_val,
                        actual_pose.orientation.w_val,
                    ]
                    retry_position_error = float(
                        np.linalg.norm(
                            actual_position_for_retry
                            - target_position_for_retry
                        )
                    )
                    retry_orientation_error = abs(
                        math.degrees(
                            (
                                _yaw_from_quaternion(actual_quaternion_for_retry)
                                - _yaw_from_quaternion(target_quaternion_for_retry)
                                + math.pi
                            )
                            % (2.0 * math.pi)
                            - math.pi
                        )
                    )
                    if (
                        retry_position_error
                        <= self.ROTATE_POSITION_TOLERANCE
                        and retry_orientation_error
                        <= self.ROTATE_RETRY_TOLERANCE_DEG
                    ):
                        break
                after_collision = client.simGetCollisionInfo()
            else:
                actual_pose = client.simGetVehiclePose()

            collision = bool(
                fly_type == "move"
                and after_collision.has_collided
                and after_collision.time_stamp > before_collision.time_stamp
            )
            actual_pose = client.simGetVehiclePose()
            target_position = np.array(
                [
                    target_pose.position.x_val,
                    target_pose.position.y_val,
                    target_pose.position.z_val,
                ],
                dtype=np.float64,
            )
            actual_position = np.array(
                [
                    actual_pose.position.x_val,
                    actual_pose.position.y_val,
                    actual_pose.position.z_val,
                ],
                dtype=np.float64,
            )
            pose_error = float(np.linalg.norm(actual_position - target_position))
            target_quaternion = [
                target_pose.orientation.x_val,
                target_pose.orientation.y_val,
                target_pose.orientation.z_val,
                target_pose.orientation.w_val,
            ]
            actual_quaternion = [
                actual_pose.orientation.x_val,
                actual_pose.orientation.y_val,
                actual_pose.orientation.z_val,
                actual_pose.orientation.w_val,
            ]
            orientation_error_deg = abs(
                math.degrees(
                    (
                        _yaw_from_quaternion(actual_quaternion)
                        - _yaw_from_quaternion(target_quaternion)
                        + math.pi
                    )
                    % (2.0 * math.pi)
                    - math.pi
                )
            )
            if not collision and pose_error >= self.ACTION_POSE_HARD_LIMIT:
                raise RuntimeError(
                    "AirSim action pose integrity failure: fly_type={} "
                    "error={:.3f} m requested={} actual={}".format(
                        fly_type,
                        pose_error,
                        target_position.tolist(),
                        actual_position.tolist(),
                    )
                )
            if collision:
                execution_status = "airsim_contact"
            elif fly_type == "move" and pose_error > self.MOVE_RETRY_TOLERANCE:
                execution_status = "accepted_actual_pose"
            elif fly_type == "move" and pose_attempts > 1:
                execution_status = "aligned_after_retry"
            elif fly_type == "move":
                execution_status = "aligned"
            elif fly_type == "rotate":
                if (
                    pose_error > self.ROTATE_POSITION_TOLERANCE
                    or orientation_error_deg
                    > self.ROTATE_RETRY_TOLERANCE_DEG
                ):
                    execution_status = "accepted_actual_rotation"
                elif pose_attempts > 1:
                    execution_status = "rotation_aligned_after_retry"
                else:
                    execution_status = "rotation_aligned"
            else:
                execution_status = "stopped"
            move_results[0].append(
                {
                    "actual_pose": actual_pose,
                    "collision": collision,
                    "collision_source": (
                        "airsim_contact" if collision else "none"
                    ),
                    "pose_error": pose_error,
                    "orientation_error_deg": orientation_error_deg,
                    "pose_attempts": pose_attempts,
                    "execution_status": execution_status,
                    "requested_position": target_position.tolist(),
                    "requested_quaternion": target_quaternion,
                }
            )

        for index, state in enumerate(self.states):
            result = move_results[0][index]
            if was_ended[index]:
                continue
            action = "stop" if state.is_end else requested_actions[index]
            previous_position = np.asarray(state.position, dtype=np.float64)
            actual_pose = result.get("actual_pose", target_poses[index])
            position = [
                actual_pose.position.x_val,
                actual_pose.position.y_val,
                actual_pose.position.z_val,
            ]
            quaternion = [
                actual_pose.orientation.x_val,
                actual_pose.orientation.y_val,
                actual_pose.orientation.z_val,
                actual_pose.orientation.w_val,
            ]
            current_position = np.asarray(position, dtype=np.float64)
            current_yaw = _yaw_from_quaternion(quaternion)
            delta_yaw = abs(
                math.degrees(
                    (current_yaw - previous_yaws[index] + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
            )

            state.heading_changes.append(delta_yaw)
            step_move_distance = float(
                np.linalg.norm(current_position - previous_position)
            )
            requested_position = np.asarray(
                result.get("requested_position", position), dtype=np.float64
            )
            commanded_step_distance = (
                float(np.linalg.norm(requested_position - previous_position))
                if fly_types[index] == "move"
                else 0.0
            )
            state.move_distance += step_move_distance
            state.commanded_move_distance += commanded_step_distance
            state.step += 1
            state.is_collisioned = bool(result.get("collision", False))
            distance = _minimum_target_distance(
                current_position, state.task["object_position"]
            )
            if distance < self.SUCCESS_DISTANCE:
                state.oracle_success = True
            if action == "stop":
                state.is_end = True

            state.trajectory.append(
                {
                    "sensors": {
                        "state": {
                            "position": [float(value) for value in position],
                            "quaternionr": [float(value) for value in quaternion],
                        }
                    },
                    "move_distance": round(state.move_distance, 4),
                    "step_move_distance": round(step_move_distance, 4),
                    "commanded_move_distance": round(
                        state.commanded_move_distance, 4
                    ),
                    "commanded_step_distance": round(
                        commanded_step_distance, 4
                    ),
                    "pose_error": round(float(result.get("pose_error", 0.0)), 6),
                    "orientation_error_deg": round(
                        float(result.get("orientation_error_deg", 0.0)), 6
                    ),
                    "pose_attempts": int(result.get("pose_attempts", 0)),
                    "execution_status": result.get(
                        "execution_status", "unknown"
                    ),
                    "collision_source": result.get("collision_source", "none"),
                    "requested_position": result.get(
                        "requested_position", [float(value) for value in position]
                    ),
                    "requested_quaternion": result.get(
                        "requested_quaternion", [float(value) for value in quaternion]
                    ),
                    "distance_to_target": round(distance, 4),
                }
            )

    def close(self):
        if self.simulator_tool is None:
            return
        try:
            self.simulator_tool.closeScenes()
        except Exception as error:
            logger.warning("Failed to close simulator scenes: %s", error)
        try:
            self.simulator_tool._closeConnection()
        except Exception as error:
            logger.warning("Failed to close AirSim client connections: %s", error)
