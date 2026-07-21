import io
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .depth_safety import DepthSafetyAnalyzer
from .logging_utils import logger
from .voxel_jepa import VoxelJEPAPlanner


CAMERA_NAMES = ("front", "left", "right", "down")
CROP_SPECS = (
    ("full", (0.0, 0.0, 1.0, 1.0)),
    # Four overlapping quadrants retain small-object sensitivity.  They are
    # evaluated only for the strongest coarse camera (8 images/episode total).
    ("top_left", (0.0, 0.0, 0.62, 0.62)),
    ("top_right", (0.38, 0.0, 1.0, 0.62)),
    ("bottom_left", (0.0, 0.38, 0.62, 1.0)),
    ("bottom_right", (0.38, 0.38, 1.0, 1.0)),
)

# Stop-only spatial refinement.  These nine overlapping windows are composed
# inside the best coarse-to-fine crop, so RGB semantics and depth statistics
# are evaluated over the same substantially smaller image region.  Navigation
# continues to use CROP_SPECS, keeping the log-8 SEARCH/APPROACH policy fixed.
SPATIAL_CROP_SPECS = tuple(
    (
        "spatial_{}_{}".format(row_name, column_name),
        (x0, y0, x0 + 0.5, y0 + 0.5),
    )
    for row_name, y0 in (("top", 0.0), ("middle", 0.25), ("bottom", 0.5))
    for column_name, x0 in (("left", 0.0), ("center", 0.25), ("right", 0.5))
)


def _clean_object_name(name):
    cleaned = str(name or "").strip()
    cleaned = re.sub(r"^(SM|BP)_", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"_\d+$", "", cleaned)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _yaw_degrees(quaternion):
    x, y, z, w = [float(value) for value in quaternion]
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return math.degrees(yaw)


@dataclass
class PolicyState:
    task_id: str
    start_position: Tuple[float, float, float]
    mode: str = "SEARCH"
    scan_remaining: int = 0
    moves_since_scan: int = 0
    search_translations: int = 0
    detection_streak: int = 0
    stop_streak: int = 0
    lost_streak: int = 0
    tracked_camera: int = -1
    previous_action: str = ""
    visits: Counter = field(default_factory=Counter)
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=4))
    recent_cells: deque = field(default_factory=lambda: deque(maxlen=12))


class CollisionAwareCLIPPolicy:
    """Stateful CLIP policy with a depth-based safety shield.

    Target coordinates are deliberately absent from this class.  Every action
    is selected from RGB, depth, odometry, and task text only.
    """

    def __init__(
        self,
        policy_config,
        model_path,
        device="cuda:0",
        allow_cpu=False,
    ):
        self.config = policy_config
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            if not allow_cpu:
                raise RuntimeError(
                    "CUDA was requested but is unavailable. CPU CLIP inference is "
                    "much slower; fix CUDA/CUDA_VISIBLE_DEVICES or explicitly pass "
                    "--allow_cpu true."
                )
            logger.warning("CUDA is unavailable; explicit CPU fallback enabled")
            device = "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info(
                "CLIP inference device: %s (%s)",
                self.device,
                torch.cuda.get_device_name(self.device),
            )
        else:
            logger.warning("CLIP inference is running on CPU and will be slow")
        logger.info("Loading CLIP model %s on %s", model_path, self.device)
        try:
            self.model = CLIPModel.from_pretrained(
                model_path, local_files_only=True
            )
            self.processor = CLIPProcessor.from_pretrained(
                model_path, local_files_only=True
            )
            logger.info("Loaded CLIP from the local Hugging Face cache")
        except OSError:
            logger.warning(
                "CLIP is not complete in the local cache; trying normal loading"
            )
            self.model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = self.model.to(self.device).eval()
        self.depth_analyzer = DepthSafetyAnalyzer(
            horizontal_step=policy_config.horizontal_step,
            vertical_step=policy_config.vertical_step,
            safety_margin=policy_config.safety_margin,
            percentile=policy_config.collision_percentile,
        )
        self.states: Dict[int, PolicyState] = {}
        self.text_feature_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self.target_component_cache: Dict[
            Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.generic_features = self._encode_texts(
            [
                "an outdoor scene",
                "a street with buildings",
                "sky, ground and landscape",
                "a generic urban environment",
            ]
        )
        self.world_model = (
            VoxelJEPAPlanner(policy_config, self.device)
            if policy_config.world_model_enabled
            else None
        )

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self.device.type == "cuda")

    def save_world_model(self, path):
        if self.world_model is not None:
            self.world_model.save(path)

    def _encode_texts(self, texts):
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode(), self._autocast():
            features = self.model.get_text_features(**inputs)
        return F.normalize(features.float(), dim=-1)

    def _target_features(self, object_name, description):
        clean_name = _clean_object_name(object_name)
        clean_description = str(description or "").strip()
        cache_key = (clean_name, clean_description)
        if cache_key not in self.text_feature_cache:
            prompts = [
                "a photo of a {}".format(clean_name),
                "a close-up photo of a {}".format(clean_name),
                "an aerial navigation target: {}".format(clean_name),
                "a {} with this appearance: {}".format(
                    clean_name, clean_description
                ),
                clean_description,
            ]
            features = self._encode_texts(prompts)
            # Keep both components for round-by-round calibration.  The mixed
            # feature remains description-heavy (the best round-5 setting),
            # while diagnostics expose whether category and detail agree on
            # the same localized region.
            category_feature = F.normalize(
                features[:3].mean(dim=0, keepdim=True), dim=-1
            )
            detail_feature = F.normalize(
                features[3:].mean(dim=0, keepdim=True), dim=-1
            )
            self.target_component_cache[cache_key] = (
                category_feature,
                detail_feature,
            )
            self.text_feature_cache[cache_key] = F.normalize(
                0.30 * category_feature + 0.70 * detail_feature, dim=-1
            )
        return self.text_feature_cache[cache_key]

    def _target_components(self, object_name, description):
        clean_name = _clean_object_name(object_name)
        clean_description = str(description or "").strip()
        cache_key = (clean_name, clean_description)
        self._target_features(object_name, description)
        return self.target_component_cache[cache_key]

    @staticmethod
    def _decode_rgb(encoded):
        image = Image.open(io.BytesIO(bytes(encoded)))
        image.load()
        return image.convert("RGB")

    @staticmethod
    def _crop(image, normalized_box):
        width, height = image.size
        x0, y0, x1, y1 = normalized_box
        return image.crop(
            (
                int(x0 * width),
                int(y0 * height),
                max(int(x1 * width), 1),
                max(int(y1 * height), 1),
            )
        )

    @staticmethod
    def _compose_box(parent_box, relative_box):
        px0, py0, px1, py1 = parent_box
        rx0, ry0, rx1, ry1 = relative_box
        width = px1 - px0
        height = py1 - py0
        return (
            px0 + rx0 * width,
            py0 + ry0 * height,
            px0 + rx1 * width,
            py0 + ry1 * height,
        )

    def _encode_pil_images(self, images):
        processed = self.processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        feature_chunks = []
        with torch.inference_mode():
            for start in range(0, len(images), self.config.image_batch_size):
                batch = pixel_values[
                    start : start + self.config.image_batch_size
                ].to(self.device, non_blocking=self.device.type == "cuda")
                with self._autocast():
                    features = self.model.get_image_features(pixel_values=batch)
                feature_chunks.append(F.normalize(features.float(), dim=-1))
        return torch.cat(feature_chunks, dim=0)

    def _visual_detections(self, observations):
        target_features = torch.cat(
            [
                self._target_features(
                    observation.get("object_name", "object"),
                    observation.get("description", ""),
                )
                for observation in observations
            ],
            dim=0,
        )
        component_pairs = [
            self._target_components(
                observation.get("object_name", "object"),
                observation.get("description", ""),
            )
            for observation in observations
        ]
        category_features = torch.cat(
            [pair[0] for pair in component_pairs], dim=0
        )
        detail_features = torch.cat(
            [pair[1] for pair in component_pairs], dim=0
        )
        opened_images = []
        localized_crops = []
        try:
            for observation in observations:
                rgb_images = observation.get("rgb", [])
                if len(rgb_images) != 4:
                    raise ValueError("each observation must contain four RGB images")
                opened_images.extend(self._decode_rgb(item) for item in rgb_images)

            # Coarse-to-fine inference: score the four full camera views first,
            # then spend crop-level compute only on the best camera.  This uses
            # eight images per episode instead of twenty while retaining the
            # small-object crops that are needed for precise stopping.
            full_features = self._encode_pil_images(opened_images).reshape(
                len(observations), 4, -1
            )
            coarse_scores = torch.einsum(
                "ecd,ed->ec", full_features, target_features
            )
            best_coarse_cameras = coarse_scores.argmax(dim=1).tolist()
            crop_locations = []
            for episode_index, camera_index in enumerate(best_coarse_cameras):
                image = opened_images[episode_index * 4 + camera_index]
                for crop_index, (_, crop_box) in enumerate(CROP_SPECS[1:], 1):
                    localized_crops.append(self._crop(image, crop_box))
                    crop_locations.append(
                        (episode_index, camera_index, crop_index)
                    )

            feature_grid = full_features.unsqueeze(2).expand(
                -1, -1, len(CROP_SPECS), -1
            ).clone()
            if localized_crops:
                localized_features = self._encode_pil_images(localized_crops)
                for feature, location in zip(localized_features, crop_locations):
                    feature_grid[location] = feature
        finally:
            for crop in localized_crops:
                crop.close()
            for image in opened_images:
                image.close()

        per_episode = []
        target_scores = torch.einsum(
            "eckd,ed->eck", feature_grid, target_features
        )
        category_scores = torch.einsum(
            "eckd,ed->eck", feature_grid, category_features
        )
        detail_scores = torch.einsum(
            "eckd,ed->eck", feature_grid, detail_features
        )
        generic_scores = torch.einsum(
            "eckd,gd->eckg", feature_grid, self.generic_features
        ).amax(dim=-1)
        target_values = target_scores.cpu().numpy()
        category_values = category_scores.cpu().numpy()
        detail_values = detail_scores.cpu().numpy()
        generic_values = generic_scores.cpu().numpy()

        for episode_index, observation in enumerate(observations):
            cameras = []
            for camera_index in range(4):
                camera_values = target_values[episode_index, camera_index]
                best_crop_index = int(np.argmax(camera_values))
                best_score = float(camera_values[best_crop_index])
                category_score = float(
                    category_values[
                        episode_index, camera_index, best_crop_index
                    ]
                )
                detail_score = float(
                    detail_values[episode_index, camera_index, best_crop_index]
                )
                full_score = float(camera_values[0])
                camera_score = 0.72 * best_score + 0.28 * full_score
                specificity = best_score - float(
                    generic_values[episode_index, camera_index, best_crop_index]
                )
                cameras.append(
                    {
                        "camera": camera_index,
                        "name": CAMERA_NAMES[camera_index],
                        "score": camera_score,
                        "best_score": best_score,
                        "full_score": full_score,
                        "specificity": specificity,
                        "category_score": category_score,
                        "detail_score": detail_score,
                        "crop": CROP_SPECS[best_crop_index][0],
                        "box": CROP_SPECS[best_crop_index][1],
                    }
                )

            ranked = sorted(cameras, key=lambda item: item["score"], reverse=True)
            best = ranked[0]
            margin = best["score"] - ranked[1]["score"]
            target_depth = self.depth_analyzer.target_region_depth(
                observation["depth"][best["camera"]], best["box"]
            )
            per_episode.append(
                {
                    "cameras": cameras,
                    "best_camera": best["camera"],
                    "best_score": best["score"],
                    "best_patch_score": best["best_score"],
                    "full_score": best["full_score"],
                    "specificity": best["specificity"],
                    "category_score": best["category_score"],
                    "detail_score": best["detail_score"],
                    "margin": margin,
                    "target_depth": target_depth,
                    "best_box": best["box"],
                    "best_crop": best["crop"],
                }
            )
        self._refine_spatial_stop_evidence(
            observations,
            per_episode,
            target_features,
            category_features,
            detail_features,
        )
        return per_episode

    def _base_stop_semantic_candidate(self, detection):
        localized = bool(
            detection["best_crop"] != "full"
            or detection["margin"] >= 2.0 * self.config.direction_margin
        )
        return bool(
            detection["best_score"] >= self.config.approach_score
            and localized
            and detection["best_patch_score"] >= self.config.stop_score
            and detection["specificity"] >= self.config.stop_specificity
            and detection["category_score"] >= self.config.stop_category_score
            and detection["detail_score"] >= self.config.stop_detail_score
        )

    def _refine_spatial_stop_evidence(
        self,
        observations,
        detections,
        target_features,
        category_features,
        detail_features,
    ):
        for detection in detections:
            detection.update(
                {
                    "spatial_refined": False,
                    "spatial_crop": "none",
                    "spatial_box": detection["best_box"],
                    "spatial_score": 0.0,
                    "spatial_specificity": 0.0,
                    "spatial_category_score": 0.0,
                    "spatial_detail_score": 0.0,
                    "spatial_target_depth": 100.0,
                    "spatial_depth_median": 100.0,
                    "spatial_depth_support": 0.0,
                    "spatial_depth_valid_fraction": 0.0,
                }
            )

        if not self.config.spatial_stop_enabled:
            return

        candidate_indices = [
            index
            for index, detection in enumerate(detections)
            if self._base_stop_semantic_candidate(detection)
        ]
        if not candidate_indices:
            return

        spatial_images = []
        spatial_locations = []
        opened_images = []
        try:
            for episode_index in candidate_indices:
                detection = detections[episode_index]
                camera_index = detection["best_camera"]
                image = self._decode_rgb(
                    observations[episode_index]["rgb"][camera_index]
                )
                opened_images.append(image)
                for crop_name, relative_box in SPATIAL_CROP_SPECS:
                    absolute_box = self._compose_box(
                        detection["best_box"], relative_box
                    )
                    spatial_images.append(self._crop(image, absolute_box))
                    spatial_locations.append(
                        (episode_index, crop_name, absolute_box)
                    )

            image_features = self._encode_pil_images(spatial_images)
            episode_lookup = torch.tensor(
                [item[0] for item in spatial_locations],
                device=self.device,
                dtype=torch.long,
            )
            selected_targets = target_features.index_select(0, episode_lookup)
            selected_categories = category_features.index_select(0, episode_lookup)
            selected_details = detail_features.index_select(0, episode_lookup)
            target_scores = torch.einsum("nd,nd->n", image_features, selected_targets)
            category_scores = torch.einsum(
                "nd,nd->n", image_features, selected_categories
            )
            detail_scores = torch.einsum("nd,nd->n", image_features, selected_details)
            generic_scores = torch.einsum(
                "nd,gd->ng", image_features, self.generic_features
            ).amax(dim=-1)

            target_values = target_scores.cpu().numpy()
            category_values = category_scores.cpu().numpy()
            detail_values = detail_scores.cpu().numpy()
            generic_values = generic_scores.cpu().numpy()
            offsets = {index: [] for index in candidate_indices}
            for offset, (episode_index, _, _) in enumerate(spatial_locations):
                offsets[episode_index].append(offset)

            for episode_index in candidate_indices:
                candidate_offsets = offsets[episode_index]
                best_offset = max(
                    candidate_offsets,
                    key=lambda offset: float(target_values[offset]),
                )
                _, crop_name, absolute_box = spatial_locations[best_offset]
                detection = detections[episode_index]
                evidence = self.depth_analyzer.target_region_evidence(
                    observations[episode_index]["depth"][detection["best_camera"]],
                    absolute_box,
                    self.config.min_stop_depth,
                    self.config.stop_depth,
                )
                spatial_score = float(target_values[best_offset])
                detection.update(
                    {
                        "spatial_refined": True,
                        "spatial_crop": crop_name,
                        "spatial_box": absolute_box,
                        "spatial_score": spatial_score,
                        "spatial_specificity": spatial_score
                        - float(generic_values[best_offset]),
                        "spatial_category_score": float(
                            category_values[best_offset]
                        ),
                        "spatial_detail_score": float(detail_values[best_offset]),
                        "spatial_target_depth": evidence.depth,
                        "spatial_depth_median": evidence.median_depth,
                        "spatial_depth_support": evidence.window_fraction,
                        "spatial_depth_valid_fraction": evidence.valid_fraction,
                    }
                )
        finally:
            for crop in spatial_images:
                crop.close()
            for image in opened_images:
                image.close()

    def _state_for(self, slot, observation):
        task_id = str(observation.get("task_id", slot))
        existing = self.states.get(slot)
        if existing is None or existing.task_id != task_id:
            start_position = tuple(
                float(value) for value in observation["start_position"]
            )
            existing = PolicyState(
                task_id=task_id,
                start_position=start_position,
                scan_remaining=self.config.scan_turns,
            )
            self.states[slot] = existing
        return existing

    def _cell(self, position):
        size = max(self.config.visit_cell_size, 0.1)
        return tuple(int(round(float(value) / size)) for value in position[:2])

    @staticmethod
    def _predict_position(position, yaw_degrees, action, step):
        yaw = math.radians(yaw_degrees)
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        right = np.array([-math.sin(yaw), math.cos(yaw)])
        current = np.asarray(position[:2], dtype=np.float64)
        if action == "forward":
            current += forward * step
        elif action == "left":
            current -= right * step
        elif action == "right":
            current += right * step
        return current

    def _within_search_bounds(self, state, predicted_xy):
        start = np.asarray(state.start_position[:2], dtype=np.float64)
        return bool(np.all(np.abs(predicted_xy - start) <= self.config.search_radius))

    def _fixed_step(self, action, safety):
        if action in ("rotl", "rotr"):
            return float(self.config.rotate_angle)
        return float(
            self.config.vertical_step
            if action in ("ascend", "descend")
            else self.config.horizontal_step
        )

    def _motion_step(self, action, safety, detection, detected):
        """Use short physical steps near likely targets or marginal clearance."""
        base = self._fixed_step(action, safety)
        if action not in safety:
            return base

        step = base
        clearance = float(safety[action].clearance)
        if clearance < safety[action].required_clearance + 6.0:
            step = min(step, base * 0.5)

        if detected:
            target_depth = float(detection["target_depth"])
            if target_depth <= 8.0:
                step = min(step, 0.5)
            elif target_depth <= self.config.stop_depth:
                step = min(step, 1.0)
        return max(0.5, float(step))

    def _recovery_rotation(self, safety):
        left = safety["left"]
        right = safety["right"]
        if left.clearance > right.clearance + 0.5:
            return "rotl", "rotate_toward_larger_left_clearance"
        return "rotr", "rotate_toward_larger_right_clearance"

    def _search_action(self, state, observation, safety, detection=None):
        if (
            self.config.search_translation_budget > 0
            and state.search_translations
            >= self.config.search_translation_budget
        ):
            return "stop", "search_translation_budget_exhausted"

        if state.scan_remaining > 0:
            state.scan_remaining -= 1
            return "rotr", "short_active_scan"

        if state.moves_since_scan >= self.config.search_moves_per_scan:
            state.moves_since_scan = 0
            state.scan_remaining = self.config.periodic_scan_turns
            if state.scan_remaining > 0:
                state.scan_remaining -= 1
                return "rotr", "short_periodic_search_scan"

        position = observation["sensors"]["state"]["position"]
        quaternion = observation["sensors"]["state"]["quaternionr"]
        yaw = _yaw_degrees(quaternion)
        semantic_preferences = {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
        }
        if detection is not None:
            directional_scores = np.asarray(
                [item["score"] for item in detection["cameras"][:3]],
                dtype=np.float64,
            )
            score_span = float(np.ptp(directional_scores))
            if score_span >= self.config.direction_margin:
                normalized = (
                    directional_scores - directional_scores.min()
                ) / score_span
                semantic_preferences = dict(
                    zip(("forward", "left", "right"), normalized.tolist())
                )
        candidates = []
        for action in ("forward", "left", "right"):
            action_safety = safety[action]
            if not action_safety.safe:
                continue
            step = self._fixed_step(action, safety)
            predicted = self._predict_position(position, yaw, action, step)
            if not self._within_search_bounds(state, predicted):
                continue
            visits = state.visits[self._cell(predicted)]
            novelty = 1.0 / (1.0 + visits)
            clearance_reward = min(action_safety.clearance, 20.0) / 20.0
            inverse_lateral = bool(
                (action == "left" and state.previous_action == "right")
                or (action == "right" and state.previous_action == "left")
            )
            recent_cell_penalty = (
                1.0 if self._cell(predicted) in state.recent_cells else 0.0
            )
            utility = (
                3.0 * novelty
                + 1.0 * clearance_reward
                + 1.0 * semantic_preferences[action]
                - 2.0 * action_safety.risk
                - 1.5 * recent_cell_penalty
                - 0.75 * inverse_lateral
            )
            candidates.append((utility, action))

        if not candidates:
            state.mode = "RECOVERY"
            return self._recovery_rotation(safety)
        state.moves_since_scan += 1
        candidates.sort(reverse=True)
        return candidates[0][1], "safe_unvisited_search_cell"

    def _approach_action(self, state, detection, safety):
        camera = detection["best_camera"]
        if camera == 3:
            horizontal = max(
                detection["cameras"][:3], key=lambda item: item["score"]
            )
            if horizontal["score"] >= detection["best_score"] - 0.015:
                camera = horizontal["camera"]
        desired = {0: "forward", 1: "rotl", 2: "rotr", 3: "descend"}[camera]

        if (
            (state.previous_action == "rotl" and desired == "rotr")
            or (state.previous_action == "rotr" and desired == "rotl")
        ):
            front_score = detection["cameras"][0]["score"]
            if (
                front_score >= detection["best_score"] - 2.0 * self.config.direction_margin
                and safety["forward"].safe
            ):
                return "forward", "target_crossed_image_center"

        if desired in ("rotl", "rotr"):
            return desired, "rotate_side_target_into_front_camera"
        if safety[desired].safe:
            return desired, "depth_safe_target_approach"

        state.mode = "RECOVERY"
        return self._recovery_rotation(safety)

    def _update_detection_state(self, state, detection):
        score_ok = detection["best_score"] >= self.config.approach_score
        down_evidence_ok = bool(
            detection["best_camera"] != 3
            or detection["specificity"] >= self.config.stop_specificity
            or detection["best_patch_score"]
            >= self.config.approach_score + 0.03
        )
        evidence_ok = bool(
            down_evidence_ok
            and
            detection["specificity"] >= self.config.min_specificity
            and (
                detection["margin"] >= self.config.direction_margin
                or detection["best_patch_score"]
                >= self.config.approach_score + 0.02
            )
        )
        detected = bool(score_ok and evidence_ok)
        if detected:
            camera_changed = bool(
                state.tracked_camera >= 0
                and state.tracked_camera != detection["best_camera"]
            )
            if camera_changed:
                state.detection_streak = 1
                state.stop_streak = 0
            else:
                state.detection_streak += 1
            state.lost_streak = 0
            state.tracked_camera = detection["best_camera"]
            state.recent_scores.append(detection["best_score"])
        else:
            state.detection_streak = 0
            state.stop_streak = 0
            state.lost_streak += 1
            if state.lost_streak >= self.config.lost_target_frames:
                state.tracked_camera = -1
                state.mode = "SEARCH"
        return detected

    def _stop_ready(self, state, detection, detected):
        localized = bool(
            detection["best_crop"] != "full"
            or detection["margin"] >= 2.0 * self.config.direction_margin
        )
        if self.config.spatial_stop_enabled:
            depth_and_spatial_evidence = bool(
                detection["spatial_refined"]
                and detection["spatial_score"] >= self.config.stop_score
                and detection["spatial_specificity"]
                >= self.config.stop_specificity
                and detection["spatial_category_score"]
                >= self.config.stop_category_score
                and detection["spatial_detail_score"]
                >= self.config.stop_detail_score
                and detection["spatial_target_depth"]
                >= self.config.min_stop_depth
                and detection["spatial_target_depth"] <= self.config.stop_depth
                and detection["spatial_depth_support"]
                >= self.config.min_stop_depth_support
            )
        else:
            depth_and_spatial_evidence = bool(
                detection["target_depth"] >= self.config.min_stop_depth
                and detection["target_depth"] <= self.config.stop_depth
            )
        candidate = bool(
            detected
            and localized
            and detection["best_patch_score"] >= self.config.stop_score
            and detection["specificity"] >= self.config.stop_specificity
            and detection["category_score"]
            >= self.config.stop_category_score
            and detection["detail_score"] >= self.config.stop_detail_score
            and depth_and_spatial_evidence
        )
        state.stop_streak = state.stop_streak + 1 if candidate else 0
        return state.stop_streak >= self.config.stop_confirm_frames

    def _quick_stop_ready(self, detection, detected):
        return bool(
            self.config.quick_stop_enabled
            and detected
            and detection["best_patch_score"] >= self.config.quick_stop_score
            and detection["specificity"] >= self.config.quick_stop_specificity
            and detection["target_depth"] >= self.config.quick_stop_min_depth
            and detection["target_depth"] <= self.config.quick_stop_depth
        )

    @staticmethod
    def _rounded_camera_scores(cameras):
        return [
            {
                "camera": item["name"],
                "score": round(float(item["score"]), 5),
                "best_score": round(float(item["best_score"]), 5),
                "specificity": round(float(item["specificity"]), 5),
                "category_score": round(float(item["category_score"]), 5),
                "detail_score": round(float(item["detail_score"]), 5),
                "crop": item["crop"],
                "box": [round(float(value), 3) for value in item["box"]],
            }
            for item in cameras
        ]

    def act(self, observations: Sequence[Dict]):
        detections = self._visual_detections(observations)
        actions, step_sizes, dones, diagnostics = [], [], [], []

        for slot, (observation, detection) in enumerate(
            zip(observations, detections)
        ):
            state = self._state_for(slot, observation)
            position = observation["sensors"]["state"]["position"]
            current_cell = self._cell(position)
            state.visits[current_cell] += 1
            state.recent_cells.append(current_cell)
            safety = self.depth_analyzer.analyze(observation["depth"])
            world_model = getattr(self, "world_model", None)
            if world_model is not None:
                world_model.observe(slot, observation)
            detected = self._update_detection_state(state, detection)
            stop_ready = self._stop_ready(state, detection, detected)
            quick_stop_ready = self._quick_stop_ready(detection, detected)
            rotation_limit = self.config.recovery_rotation_limit
            rotation_loop = bool(
                rotation_limit > 0
                and len(state.recent_actions) >= rotation_limit
                and all(
                    action in ("rotl", "rotr")
                    for action in list(state.recent_actions)[-rotation_limit:]
                )
            )
            translation_budget_exhausted = bool(
                self.config.search_translation_budget > 0
                and state.search_translations
                >= self.config.search_translation_budget
            )

            if translation_budget_exhausted:
                state.mode = "STOP"
                action, reason = "stop", "translation_budget_exhausted"
            elif rotation_loop:
                state.mode = "STOP"
                action, reason = "stop", "recovery_rotation_limit"
            elif quick_stop_ready:
                state.mode = "STOP"
                action, reason = "stop", "high_confidence_single_frame_stop"
            elif stop_ready:
                state.mode = "STOP"
                action, reason = "stop", "stable_visual_and_depth_stop_evidence"
            elif detected and state.detection_streak >= self.config.approach_confirm_frames:
                state.mode = "APPROACH"
                action, reason = self._approach_action(state, detection, safety)
            else:
                action, reason = self._search_action(
                    state,
                    observation,
                    safety,
                    detection,
                )

            if (
                action == "descend"
                and len(state.recent_actions) >= 3
                and all(
                    item == "descend"
                    for item in list(state.recent_actions)[-3:]
                )
            ):
                action, search_reason = self._search_action(
                    state, observation, safety, detection
                )
                state.mode = "RECOVERY"
                reason = "descend_streak_limit;{}".format(search_reason)

            policy_action = action
            world_diagnostic = {"world_model_enabled": False}
            if world_model is not None:
                action, world_diagnostic = world_model.plan(
                    slot,
                    observation,
                    detection,
                    safety,
                    policy_action,
                    detected,
                )
                if action != policy_action:
                    reason = "voxel_jepa_mpc_override_{}_to_{}".format(
                        policy_action, action
                    )
                    state.mode = "SEARCH"

            # Final safety shield: no semantic or exploration branch can issue
            # a translation that the corresponding directional depth rejects.
            proposed_action = policy_action
            horizontal_safe_count = sum(
                safety[item].safe for item in ("forward", "left", "right")
            )
            if (
                action in ("forward", "left", "right")
                and horizontal_safe_count < 2
            ):
                action, shield_reason = self._recovery_rotation(safety)
                state.mode = "RECOVERY"
                reason = "horizontal_safety_consensus_blocked_{};{}".format(
                    proposed_action, shield_reason
                )
            if action in safety and not safety[action].safe:
                action, shield_reason = self._recovery_rotation(safety)
                state.mode = "RECOVERY"
                reason = "depth_shield_blocked_{};{}".format(
                    proposed_action, shield_reason
                )

            step_size = self._motion_step(action, safety, detection, detected)
            done = action == "stop"
            if world_model is not None:
                world_diagnostic["executed_action"] = action
                world_model.commit(slot, action)
            if action in (
                "forward",
                "left",
                "right",
                "ascend",
                "descend",
            ):
                state.search_translations += 1
            state.previous_action = action
            state.recent_actions.append(action)
            actions.append(action)
            step_sizes.append(step_size)
            dones.append(done)
            diagnostics.append(
                {
                    "task_id": state.task_id,
                    "mode": state.mode,
                    "action": action,
                    "proposed_action": proposed_action,
                    "step_size": round(step_size, 3),
                    "reason": reason,
                    "detected": detected,
                    "detection_streak": state.detection_streak,
                    "stop_streak": state.stop_streak,
                    "best_camera": CAMERA_NAMES[detection["best_camera"]],
                    "best_score": round(float(detection["best_score"]), 5),
                    "best_patch_score": round(
                        float(detection["best_patch_score"]), 5
                    ),
                    "specificity": round(float(detection["specificity"]), 5),
                    "category_score": round(
                        float(detection["category_score"]), 5
                    ),
                    "detail_score": round(
                        float(detection["detail_score"]), 5
                    ),
                    "direction_margin": round(float(detection["margin"]), 5),
                    "target_depth": round(float(detection["target_depth"]), 3),
                    "spatial_refined": detection["spatial_refined"],
                    "spatial_crop": detection["spatial_crop"],
                    "spatial_box": [
                        round(float(value), 3)
                        for value in detection["spatial_box"]
                    ],
                    "spatial_score": round(float(detection["spatial_score"]), 5),
                    "spatial_specificity": round(
                        float(detection["spatial_specificity"]), 5
                    ),
                    "spatial_category_score": round(
                        float(detection["spatial_category_score"]), 5
                    ),
                    "spatial_detail_score": round(
                        float(detection["spatial_detail_score"]), 5
                    ),
                    "spatial_target_depth": round(
                        float(detection["spatial_target_depth"]), 3
                    ),
                    "spatial_depth_median": round(
                        float(detection["spatial_depth_median"]), 3
                    ),
                    "spatial_depth_support": round(
                        float(detection["spatial_depth_support"]), 4
                    ),
                    "spatial_depth_valid_fraction": round(
                        float(detection["spatial_depth_valid_fraction"]), 4
                    ),
                    "camera_scores": self._rounded_camera_scores(
                        detection["cameras"]
                    ),
                    "depth_safety": self.depth_analyzer.serializable(safety),
                    "world_model": world_diagnostic,
                }
            )

        return actions, step_sizes, dones, diagnostics
