import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .clip_policy import CollisionAwareCLIPPolicy
from .config import parse_args, policy_config_from_args
from .environment import AerialNavigationEnv
from .logging_utils import logger


def _load_existing_summary(eval_save_path):
    """Return one completed result per task from a previous interrupted run."""
    results = {}
    pattern = "*/task_*/result.json"
    for result_path in sorted(Path(eval_save_path).glob(pattern)):
        with open(result_path, "r", encoding="utf-8") as result_file:
            result = json.load(result_file)
        task_id = str(result["task_id"])
        if task_id in results:
            raise RuntimeError(
                "duplicate completed task {} under {}".format(
                    task_id, eval_save_path
                )
            )
        results[task_id] = result
    return [results[key] for key in sorted(results, key=lambda item: int(item))]


def _save_rgbd(observation, root, task_id, step_index):
    if not root:
        return
    step_dir = Path(root) / "task_{}".format(task_id) / "step_{:03d}".format(
        step_index
    )
    step_dir.mkdir(parents=True, exist_ok=True)
    for camera_index, encoded in enumerate(observation.get("rgb", [])):
        with open(
            step_dir / "camera_{}_rgb.png".format(camera_index), "wb"
        ) as image_file:
            image_file.write(bytes(encoded))
    for camera_index, depth in enumerate(observation.get("depth", [])):
        Image.fromarray(np.asarray(depth, dtype=np.uint8)).save(
            step_dir / "camera_{}_depth.png".format(camera_index),
            compress_level=1,
        )


def _episode_directory(args, task, success, oracle_success):
    prefix = "success_" if success else "oracle_" if oracle_success else ""
    dataset_name = os.path.basename(args.dataset_path)
    return (
        Path(args.eval_save_path)
        / "{}{}".format(prefix, dataset_name)
        / "task_{}".format(task["task_id"])
    )


def _write_episode(
    args,
    state,
    collisions,
    actions,
    step_sizes,
    decisions,
    success,
    oracle_success,
):
    episode_dir = _episode_directory(
        args, state.task, success=success, oracle_success=oracle_success
    )
    log_dir = episode_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    trajectory = state.trajectory
    if not (
        len(trajectory)
        == len(collisions)
        == len(actions)
        == len(step_sizes)
        == len(decisions)
    ):
        raise RuntimeError(
            "trajectory and decision histories are not aligned for task {}: "
            "trajectory={}, collisions={}, actions={}, steps={}, decisions={}".format(
                state.task["task_id"],
                len(trajectory),
                len(collisions),
                len(actions),
                len(step_sizes),
                len(decisions),
            )
        )

    trajectory_path = log_dir / "trajectory.jsonl"
    decision_path = log_dir / "clip_decisions.jsonl"
    with open(trajectory_path, "w", encoding="utf-8") as trajectory_file, open(
        decision_path, "w", encoding="utf-8"
    ) as decision_file:
        for frame, item in enumerate(trajectory):
            record = {
                "frame": frame,
                "is_collision": bool(collisions[frame]),
                "action": actions[frame],
                "step_size": round(float(step_sizes[frame]), 3),
                "step_move_distance": round(
                    float(item.get("step_move_distance", 0.0)), 3
                ),
                "distance_to_end": round(float(item["distance_to_target"]), 2),
            }
            trajectory_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            decision_file.write(
                json.dumps(
                    {"frame": frame, "decision": decisions[frame]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(
        episode_dir / "object_description.json", "w", encoding="utf-8"
    ) as object_file:
        json.dump(
            state.task["dataset_item"],
            object_file,
            ensure_ascii=False,
            indent=2,
        )
    with open(episode_dir / "result.json", "w", encoding="utf-8") as result_file:
        json.dump(
            {
                "task_id": state.task["task_id"],
                "success": bool(success),
                "oracle_success": bool(oracle_success),
                "collision": bool(collisions[-1]),
                "steps": len(actions) - 1,
                "final_distance": round(
                    float(trajectory[-1]["distance_to_target"]), 3
                ),
            },
            result_file,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Saved task %s to %s", state.task["task_id"], episode_dir)


def evaluate(args):
    if args.resume and not args.jepa_checkpoint_path:
        resume_checkpoint = Path(args.eval_save_path) / "jepa_world_model.pt"
        if resume_checkpoint.exists():
            args.jepa_checkpoint_path = str(resume_checkpoint)
    policy_config = policy_config_from_args(args)
    env = AerialNavigationEnv(args)
    existing_summary = (
        _load_existing_summary(args.eval_save_path) if args.resume else []
    )
    if existing_summary:
        completed_ids = {str(item["task_id"]) for item in existing_summary}
        env.data = [
            task for task in env.data if str(task["task_id"]) not in completed_ids
        ]
        logger.info(
            "Resume enabled: keeping %d completed tasks and evaluating %d remaining tasks",
            len(existing_summary),
            len(env.data),
        )
    policy = CollisionAwareCLIPPolicy(
        policy_config=policy_config,
        model_path=args.clip_model_path,
        device=args.device,
        allow_cpu=args.allow_cpu,
    )
    Path(args.eval_save_path).mkdir(parents=True, exist_ok=True)
    summary = list(existing_summary)
    completed = len(existing_summary)
    total_tasks = completed + len(env)

    try:
        while True:
            batch = env.next_minibatch()
            if batch is None:
                break

            outputs = env.reset()
            observations = [output[0] for output in outputs]
            count = len(batch)
            finished = [False] * count
            collisions = [[bool(outputs[index][2])] for index in range(count)]
            actions = [[None] for _ in range(count)]
            step_sizes = [[0.0] for _ in range(count)]
            decisions = [[None] for _ in range(count)]
            successes = [False] * count
            oracle_successes = [False] * count

            for index, observation in enumerate(observations):
                _save_rgbd(
                    observation,
                    args.image_save_path,
                    batch[index]["task_id"],
                    0,
                )

            for step_index in range(args.maxActions):
                step_start = time.perf_counter()
                decision_start = time.perf_counter()
                (
                    proposed_actions,
                    proposed_steps,
                    proposed_dones,
                    diagnostics,
                ) = policy.act(observations)
                decision_time = time.perf_counter() - decision_start

                for index in range(count):
                    if finished[index]:
                        proposed_actions[index] = "stop"
                        proposed_steps[index] = 0.0
                        proposed_dones[index] = True

                simulator_start = time.perf_counter()
                env.make_actions(proposed_actions, proposed_steps)
                simulator_time = time.perf_counter() - simulator_start
                observation_start = time.perf_counter()
                outputs = env.get_obs()
                observation_time = time.perf_counter() - observation_start
                next_observations = [output[0] for output in outputs]

                for index in range(count):
                    if finished[index]:
                        continue
                    collision = bool(outputs[index][2])
                    actions[index].append(proposed_actions[index])
                    step_sizes[index].append(proposed_steps[index])
                    decisions[index].append(diagnostics[index])
                    collisions[index].append(collision)
                    _save_rgbd(
                        next_observations[index],
                        args.image_save_path,
                        batch[index]["task_id"],
                        step_index + 1,
                    )

                    current_distance = env.states[index].trajectory[-1][
                        "distance_to_target"
                    ]
                    model_stopped = bool(proposed_dones[index])
                    hit_limit = step_index == args.maxActions - 1
                    success = bool(
                        model_stopped
                        and current_distance <= env.SUCCESS_DISTANCE
                        and not collision
                    )
                    # Match the legacy evaluator's OSR definition: the route
                    # counts when it entered the 20 m success radius at any
                    # point, even if the policy stopped there before the action
                    # limit.  A later collision still invalidates that route.
                    oracle_success = bool(
                        env.states[index].oracle_success
                        and not collision
                        and not success
                    )

                    if collision or model_stopped or hit_limit:
                        finished[index] = True
                        env.states[index].is_end = True
                        successes[index] = success
                        oracle_successes[index] = oracle_success
                        _write_episode(
                            args,
                            env.states[index],
                            collisions[index],
                            actions[index],
                            step_sizes[index],
                            decisions[index],
                            success,
                            oracle_success,
                        )
                        completed += 1
                        summary.append(
                            {
                                "task_id": batch[index]["task_id"],
                                "success": success,
                                "oracle_success": oracle_success,
                                "collision": collision,
                                "steps": len(actions[index]) - 1,
                                "final_distance": round(
                                    float(current_distance), 3
                                ),
                            }
                        )

                observations = next_observations
                logger.info(
                    "step=%d completed=%d/%d decision=%.3fs simulator=%.3fs "
                    "observation=%.3fs total=%.3fs actions=%s",
                    step_index,
                    completed,
                    total_tasks,
                    decision_time,
                    simulator_time,
                    observation_time,
                    time.perf_counter() - step_start,
                    proposed_actions,
                )
                if all(finished):
                    break
    finally:
        policy.save_world_model(
            str(Path(args.eval_save_path) / "jepa_world_model.pt")
        )
        env.close()

    summary_path = Path(args.eval_save_path) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    success_count = sum(item["success"] for item in summary)
    collision_count = sum(item["collision"] for item in summary)
    logger.info(
        "Evaluation complete: tasks=%d success=%d collisions=%d summary=%s",
        len(summary),
        success_count,
        collision_count,
        summary_path,
    )


def main(argv=None):
    args = parse_args(argv)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    evaluate(args)


if __name__ == "__main__":
    main()
