"""LeWM-inspired action-conditioned JEPA planning over RGB-D voxel states.

The public LeWM checkpoints target different robots and action spaces.  This
module therefore keeps its encode/predict/rollout interface while learning a
small UAV-specific latent transition model from executed, unlabeled RGB-D
transitions.  Geometry remains an explicit planning prior so an untrained
neural residual can never bypass the depth safety shield.
"""

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .depth_safety import DepthSafetyAnalyzer


ACTIONS = ("forward", "left", "right", "descend", "rotl", "rotr", "stop")
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}


@dataclass(frozen=True)
class VoxelSpec:
    z_cells: int = 8
    y_cells: int = 24
    x_cells: int = 24
    voxel_size: float = 1.0
    z_min: float = -4.0
    max_depth: float = 20.0
    samples_per_axis: int = 12


class RGBDVoxelizer:
    """Project four AirSim depth cameras into a local 3D occupancy grid."""

    def __init__(self, spec: VoxelSpec):
        self.spec = spec

    def _index(self, point):
        x, y, z = [float(value) for value in point]
        ix = int(math.floor(x / self.spec.voxel_size + self.spec.x_cells / 2))
        iy = int(math.floor(y / self.spec.voxel_size + self.spec.y_cells / 2))
        iz = int(math.floor((z - self.spec.z_min) / self.spec.voxel_size))
        if not (
            0 <= ix < self.spec.x_cells
            and 0 <= iy < self.spec.y_cells
            and 0 <= iz < self.spec.z_cells
        ):
            return None
        return iz, iy, ix

    @staticmethod
    def _horizontal_ray(u, v, yaw_offset):
        base = np.asarray([1.0, u, v], dtype=np.float32)
        c, s = math.cos(yaw_offset), math.sin(yaw_offset)
        ray = np.asarray(
            [c * base[0] - s * base[1], s * base[0] + c * base[1], base[2]],
            dtype=np.float32,
        )
        return ray / max(float(np.linalg.norm(ray)), 1e-6)

    @staticmethod
    def _down_ray(u, v):
        ray = np.asarray([v, u, 1.0], dtype=np.float32)
        return ray / max(float(np.linalg.norm(ray)), 1e-6)

    def build(self, observation):
        grid = np.zeros(
            (3, self.spec.z_cells, self.spec.y_cells, self.spec.x_cells),
            dtype=np.float32,
        )
        local_endpoints = []
        offsets = (0.0, -math.pi / 2.0, math.pi / 2.0, None)
        for camera, raw_depth in enumerate(observation["depth"]):
            depth = DepthSafetyAnalyzer.to_meters(raw_depth)
            height, width = depth.shape
            ys = np.linspace(0, height - 1, self.spec.samples_per_axis).astype(int)
            xs = np.linspace(0, width - 1, self.spec.samples_per_axis).astype(int)
            for iy in ys:
                v = (float(iy) + 0.5 - height / 2.0) / max(height / 2.0, 1.0)
                for ix in xs:
                    value = float(depth[iy, ix])
                    if not math.isfinite(value) or value < 0.4:
                        continue
                    u = (float(ix) + 0.5 - width / 2.0) / max(width / 2.0, 1.0)
                    ray = (
                        self._down_ray(u, v)
                        if offsets[camera] is None
                        else self._horizontal_ray(u, v, offsets[camera])
                    )
                    visible = min(value, self.spec.max_depth)
                    for fraction in (0.25, 0.50, 0.75):
                        index = self._index(ray * visible * fraction)
                        if index is not None:
                            grid[(1,) + index] = 1.0
                            grid[(2,) + index] = 1.0
                    if value <= self.spec.max_depth:
                        endpoint = ray * value
                        index = self._index(endpoint)
                        if index is not None:
                            grid[(0,) + index] = 1.0
                            grid[(2,) + index] = 1.0
                            local_endpoints.append(endpoint)
        # The vehicle's own voxel is known free and observed.
        center = self._index((0.0, 0.0, 0.0))
        if center is not None:
            grid[(1,) + center] = 1.0
            grid[(2,) + center] = 1.0
        return grid, local_endpoints


class VoxelJEPA(nn.Module):
    """Compact end-to-end voxel encoder and action-conditioned predictor."""

    def __init__(self, spec: VoxelSpec, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 12, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            nn.Flatten(),
            nn.Linear(24 * 2 * 3 * 3, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.action_encoder = nn.Embedding(len(ACTIONS), 24)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 24, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, voxels):
        return self.encoder(voxels.float())

    def predict(self, embedding, action_index):
        action = self.action_encoder(action_index.long())
        return embedding + 0.25 * self.predictor(torch.cat([embedding, action], -1))

    def rollout(self, voxels, action_sequences):
        """Autoregressively predict every candidate's latent trajectory."""
        initial = self.encode(voxels)
        if initial.shape[0] == 1 and action_sequences.shape[0] > 1:
            initial = initial.expand(action_sequences.shape[0], -1)
        states = [initial]
        current = initial
        for step in range(action_sequences.shape[1]):
            current = self.predict(current, action_sequences[:, step])
            states.append(current)
        return torch.stack(states, dim=1)


@dataclass
class PlannerState:
    task_id: str
    occupied: Counter = field(default_factory=Counter)
    visited: Counter = field(default_factory=Counter)
    semantic_cells: Dict[Tuple[int, int], float] = field(default_factory=dict)
    latent_history: deque = field(default_factory=lambda: deque(maxlen=24))
    previous_voxel: np.ndarray = None
    previous_action: int = None
    current_voxel: np.ndarray = None
    goal_latent: torch.Tensor = None
    goal_semantic: float = -1.0


class VoxelJEPAPlanner:
    """MPC planner combining CLIP priors, voxel geometry and JEPA rollouts."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.spec = VoxelSpec(
            z_cells=config.voxel_z_cells,
            y_cells=config.voxel_xy_cells,
            x_cells=config.voxel_xy_cells,
            voxel_size=config.voxel_size,
            max_depth=config.voxel_max_depth,
            samples_per_axis=config.voxel_samples_per_axis,
        )
        self.voxelizer = RGBDVoxelizer(self.spec)
        self.model = VoxelJEPA(
            self.spec,
            latent_dim=config.jepa_latent_dim,
            hidden_dim=config.jepa_hidden_dim,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.jepa_learning_rate, weight_decay=1e-4
        )
        self.states: Dict[int, PlannerState] = {}
        self.replay = deque(maxlen=config.jepa_replay_capacity)
        self.rng = random.Random(42)
        self.last_loss = None
        if config.jepa_checkpoint_path:
            checkpoint = torch.load(config.jepa_checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.last_loss = checkpoint.get("last_loss")

    def save(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "last_loss": self.last_loss,
                "replay_size": len(self.replay),
            },
            path,
        )

    @staticmethod
    def _yaw(quaternion):
        x, y, z, w = [float(value) for value in quaternion]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _state(self, slot, observation):
        task_id = str(observation.get("task_id", slot))
        state = self.states.get(slot)
        if state is None or state.task_id != task_id:
            state = PlannerState(task_id=task_id)
            self.states[slot] = state
        return state

    def _world_key(self, point):
        size = self.spec.voxel_size
        return tuple(int(round(float(value) / size)) for value in point)

    def observe(self, slot, observation):
        state = self._state(slot, observation)
        voxel, endpoints = self.voxelizer.build(observation)
        position = np.asarray(observation["sensors"]["state"]["position"], dtype=float)
        yaw = self._yaw(observation["sensors"]["state"]["quaternionr"])
        c, s = math.cos(yaw), math.sin(yaw)
        for endpoint in endpoints:
            world = position + np.asarray(
                [c * endpoint[0] - s * endpoint[1], s * endpoint[0] + c * endpoint[1], endpoint[2]]
            )
            state.occupied[self._world_key(world)] += 1
        cell = tuple(int(round(float(value) / self.config.visit_cell_size)) for value in position[:2])
        state.visited[cell] += 1
        if state.previous_voxel is not None and state.previous_action is not None:
            self.replay.append((state.previous_voxel, state.previous_action, voxel))
            self._train_once()
        state.current_voxel = voxel
        with torch.inference_mode():
            tensor = torch.from_numpy(voxel).unsqueeze(0).to(self.device)
            latent = self.model.encode(tensor)[0].float().cpu()
        state.latent_history.append(latent)
        return state

    def commit(self, slot, action):
        state = self.states.get(slot)
        if state is None or state.current_voxel is None:
            return
        state.previous_voxel = state.current_voxel.copy()
        state.previous_action = ACTION_INDEX.get(action, ACTION_INDEX["stop"])

    def _train_once(self):
        if not self.config.jepa_online_training or len(self.replay) < self.config.jepa_batch_size:
            return
        batch = self.rng.sample(list(self.replay), self.config.jepa_batch_size)
        before = torch.from_numpy(np.stack([item[0] for item in batch])).to(self.device)
        actions = torch.tensor([item[1] for item in batch], device=self.device)
        after = torch.from_numpy(np.stack([item[2] for item in batch])).to(self.device)
        self.model.train()
        current = self.model.encode(before)
        prediction = self.model.predict(current, actions)
        with torch.no_grad():
            target = self.model.encode(after)
        prediction_loss = F.mse_loss(prediction, target)
        centered = current - current.mean(0, keepdim=True)
        std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
        variance_loss = F.relu(0.5 - std).mean()
        covariance = centered.T @ centered / max(current.shape[0] - 1, 1)
        covariance = covariance - torch.diag(torch.diag(covariance))
        covariance_loss = covariance.square().mean()
        loss = prediction_loss + self.config.jepa_regularization * (
            variance_loss + covariance_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()
        self.last_loss = float(loss.detach().cpu())

    def _advance(self, pose, action):
        x, y, z, yaw = pose
        if action == "rotl":
            yaw -= math.radians(self.config.rotate_angle)
        elif action == "rotr":
            yaw += math.radians(self.config.rotate_angle)
        elif action == "forward":
            x += math.cos(yaw) * self.config.horizontal_step
            y += math.sin(yaw) * self.config.horizontal_step
        elif action in ("left", "right"):
            sign = -1.0 if action == "left" else 1.0
            x += sign * -math.sin(yaw) * self.config.horizontal_step
            y += sign * math.cos(yaw) * self.config.horizontal_step
        elif action == "descend":
            z += self.config.vertical_step
        return x, y, z, yaw

    def _collision_risk(self, state, pose):
        center = self._world_key(pose[:3])
        hits = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if state.occupied[(center[0] + dx, center[1] + dy, center[2] + dz)] > 0:
                        hits += 1
        return min(1.0, hits / 4.0)

    def _semantic_priors(self, detection):
        scores = [float(item["score"]) for item in detection["cameras"]]
        return {
            "forward": scores[0], "left": scores[1], "right": scores[2],
            "rotl": scores[1], "rotr": scores[2], "descend": scores[3],
        }

    def plan(self, slot, observation, detection, safety, policy_action, detected):
        state = self.states[slot]
        base = {
            "world_model_enabled": True,
            "planner_selected_action": policy_action,
            "planner_overrode_policy": False,
            "planning_horizon": self.config.jepa_planning_horizon,
            "candidate_count": 0,
            "best_sequence": [policy_action],
            "best_cost": 0.0,
            "predicted_collision_risk": 0.0,
            "semantic_reward": 0.0,
            "novelty_reward": 0.0,
            "uncertainty": 0.0,
            "jepa_replay_size": len(self.replay),
            "jepa_train_loss": self.last_loss,
            "voxel_occupied_fraction": float(np.mean(state.current_voxel[0])),
            "voxel_observed_fraction": float(np.mean(state.current_voxel[2])),
        }
        if detected or policy_action == "stop":
            return policy_action, base
        safe_roots = [a for a in ("forward", "left", "right") if safety[a].safe]
        safe_roots.extend(("rotl", "rotr"))
        if not safe_roots:
            return policy_action, base
        position = observation["sensors"]["state"]["position"]
        yaw = self._yaw(observation["sensors"]["state"]["quaternionr"])
        start_pose = (float(position[0]), float(position[1]), float(position[2]), yaw)
        priors = self._semantic_priors(detection)
        current_cell = tuple(
            int(round(float(value) / self.config.visit_cell_size))
            for value in start_pose[:2]
        )
        semantic_evidence = float(
            detection.get(
                "best_score",
                max(float(item["score"]) for item in detection["cameras"]),
            )
        ) + 2.0 * max(
            0.0, float(detection.get("specificity", 0.0))
        )
        state.semantic_cells[current_cell] = max(
            semantic_evidence, state.semantic_cells.get(current_cell, -1.0)
        )
        if (
            semantic_evidence >= self.config.jepa_goal_min_score
            and float(detection.get("specificity", 0.0)) >= -0.01
            and semantic_evidence > state.goal_semantic
            and state.latent_history
        ):
            state.goal_semantic = semantic_evidence
            state.goal_latent = state.latent_history[-1].clone()
        beam = [(0.0, tuple(), start_pose, 0.0, 0.0, 0.0)]
        for step in range(self.config.jepa_planning_horizon):
            expanded = []
            actions = safe_roots if step == 0 else ("forward", "left", "right", "rotl", "rotr")
            for cost, sequence, pose, max_risk, semantic, novelty in beam:
                for action in actions:
                    next_pose = self._advance(pose, action)
                    risk = self._collision_risk(state, next_pose)
                    cell = tuple(int(round(v / self.config.visit_cell_size)) for v in next_pose[:2])
                    revisit = 1.0 if state.visited[cell] else 0.0
                    remembered_semantic = state.semantic_cells.get(cell, 0.0)
                    sem = priors.get(action, 0.0) if step == 0 else 0.0
                    prior_penalty = 0.25 if step == 0 and action != policy_action else 0.0
                    next_cost = cost + self.config.jepa_collision_weight * risk + self.config.jepa_revisit_weight * revisit + prior_penalty - self.config.jepa_semantic_weight * sem - self.config.jepa_semantic_memory_weight * remembered_semantic
                    expanded.append((next_cost, sequence + (action,), next_pose, max(max_risk, risk), semantic + sem, novelty + (1.0 - revisit)))
            expanded.sort(key=lambda item: item[0])
            beam = expanded[: self.config.jepa_beam_width]
        sequences = [item[1] for item in beam]
        action_tensor = torch.tensor([[ACTION_INDEX[a] for a in seq] for seq in sequences], device=self.device)
        voxel_tensor = torch.from_numpy(state.current_voxel).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            latent_rollouts = self.model.rollout(voxel_tensor, action_tensor)
            drift = torch.linalg.vector_norm(latent_rollouts[:, 1:] - latent_rollouts[:, :-1], dim=-1).mean(1)
            if state.latent_history:
                history = torch.stack(list(state.latent_history)).to(self.device)
                novelty = torch.cdist(latent_rollouts[:, -1], history).min(1).values
            else:
                novelty = torch.zeros_like(drift)
            if state.goal_latent is not None:
                goal = state.goal_latent.to(self.device).unsqueeze(0)
                goal_distance = torch.linalg.vector_norm(
                    latent_rollouts[:, -1] - goal, dim=-1
                )
            else:
                goal_distance = torch.zeros_like(drift)
        trust = min(1.0, len(self.replay) / max(float(self.config.jepa_trust_transitions), 1.0))
        rescored = []
        for index, item in enumerate(beam):
            latent_novelty = float(novelty[index].cpu())
            uncertainty = float(drift[index].cpu())
            score = item[0] + trust * (
                self.config.jepa_uncertainty_weight * uncertainty
                + self.config.jepa_goal_latent_weight * float(goal_distance[index].cpu())
            )
            rescored.append((score, item, latent_novelty, uncertainty))
        best = min(rescored, key=lambda item: item[0])
        policy_candidates = [item for item in rescored if item[1][1][0] == policy_action]
        policy_best = min(policy_candidates, key=lambda item: item[0]) if policy_candidates else None
        selected = best[1][1][0]
        predicted_risk_reduction = (
            float(policy_best[1][3] - best[1][3]) if policy_best is not None else 0.0
        )
        if (
            policy_best is not None
            and selected != policy_action
            and (
                best[0] + self.config.jepa_override_margin >= policy_best[0]
                or (
                    self.config.jepa_override_min_risk_reduction > 0.0
                    and predicted_risk_reduction
                    < self.config.jepa_override_min_risk_reduction
                )
            )
        ):
            best = policy_best
            selected = policy_action
        immediate_pose = self._advance(start_pose, selected)
        immediate_risk = self._collision_risk(state, immediate_pose)
        if (
            self.config.jepa_hard_collision_enabled
            and selected in ("forward", "left", "right", "descend")
            and immediate_risk >= self.config.jepa_hard_collision_risk
        ):
            selected = "rotl" if safety["left"].clearance > safety["right"].clearance + 0.5 else "rotr"
        base.update({
            "planner_selected_action": selected,
            "planner_overrode_policy": selected != policy_action,
            "candidate_count": len(sequences),
            "best_sequence": list(best[1][1]),
            "best_cost": round(float(best[0]), 5),
            "predicted_collision_risk": round(float(best[1][3]), 5),
            "semantic_reward": round(float(best[1][4]), 5),
            "novelty_reward": round(float(best[1][5]), 5),
            "uncertainty": round(float(best[3]), 5),
            "goal_latent_active": state.goal_latent is not None,
            "goal_semantic": round(float(state.goal_semantic), 5),
            "goal_latent_distance": round(float(goal_distance[sequences.index(best[1][1])].cpu()), 5),
            "policy_best_cost": round(float(policy_best[0]), 5) if policy_best else None,
            "override_margin": self.config.jepa_override_margin,
            "predicted_risk_reduction": round(predicted_risk_reduction, 5),
            "hard_collision_fallback": selected != best[1][1][0],
            "immediate_collision_risk": round(float(immediate_risk), 5),
        })
        return selected, base
