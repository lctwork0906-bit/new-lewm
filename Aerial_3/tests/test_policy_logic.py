import io
import unittest

import numpy as np
import torch
from PIL import Image

from Aerial_3.clip_policy import (
    CROP_SPECS,
    SPATIAL_CROP_SPECS,
    CollisionAwareCLIPPolicy,
    PolicyState,
)
from Aerial_3.config import PolicyConfig
from Aerial_3.depth_safety import DepthSafetyAnalyzer, DirectionSafety


def safety(clearance=20.0, safe=True, risk=0.0):
    return DirectionSafety(
        clearance=clearance,
        required_clearance=2.5,
        near_fraction=0.0 if safe else 0.5,
        valid_fraction=1.0,
        safe=safe,
        risk=risk,
    )


class PolicyLogicTest(unittest.TestCase):
    def setUp(self):
        self.policy = object.__new__(CollisionAwareCLIPPolicy)
        self.policy.config = PolicyConfig(scan_turns=0, spatial_stop_enabled=True)
        self.state = PolicyState(
            task_id="0", start_position=(0.0, 0.0, -10.0), scan_remaining=0
        )
        self.clear = {
            "forward": safety(),
            "left": safety(),
            "right": safety(),
            "descend": safety(),
        }

    @staticmethod
    def detection(camera):
        scores = [0.20, 0.20, 0.20, 0.20]
        scores[camera] = 0.30
        return {
            "best_camera": camera,
            "best_score": 0.30,
            "best_patch_score": 0.30,
            "specificity": 0.04,
            "category_score": 0.30,
            "detail_score": 0.30,
            "margin": 0.10,
            "target_depth": 7.0,
            "best_crop": "center",
            "spatial_refined": True,
            "spatial_crop": "spatial_middle_center",
            "spatial_box": (0.25, 0.25, 0.75, 0.75),
            "spatial_score": 0.30,
            "spatial_specificity": 0.04,
            "spatial_category_score": 0.30,
            "spatial_detail_score": 0.30,
            "spatial_target_depth": 7.0,
            "spatial_depth_median": 7.0,
            "spatial_depth_support": 0.50,
            "spatial_depth_valid_fraction": 1.0,
            "cameras": [
                {
                    "name": ("front", "left", "right", "down")[index],
                    "score": score,
                    "best_score": score,
                    "specificity": 0.0,
                    "category_score": 0.0,
                    "detail_score": 0.0,
                    "crop": "full",
                    "box": (0.0, 0.0, 1.0, 1.0),
                }
                for index, score in enumerate(scores)
            ],
        }

    def test_side_target_rotates_before_translation(self):
        left_action, _ = self.policy._approach_action(
            self.state, self.detection(1), self.clear
        )
        right_action, _ = self.policy._approach_action(
            self.state, self.detection(2), self.clear
        )
        self.assertEqual(left_action, "rotl")
        self.assertEqual(right_action, "rotr")

    def test_crop_layout_supports_coarse_to_fine_localization(self):
        self.assertEqual(len(CROP_SPECS), 5)
        self.assertEqual(4 + len(CROP_SPECS) - 1, 8)
        self.assertEqual(len(SPATIAL_CROP_SPECS), 9)

    def test_unsafe_forward_uses_rotation_recovery(self):
        blocked = dict(self.clear)
        blocked["forward"] = safety(clearance=4.0, safe=False, risk=1.0)
        action, _ = self.policy._approach_action(
            self.state, self.detection(0), blocked
        )
        self.assertIn(action, ("rotl", "rotr"))
        self.assertEqual(self.state.mode, "RECOVERY")

    def test_stop_uses_single_localized_frame_in_near_depth_window(self):
        detection = self.detection(0)
        self.assertTrue(self.policy._stop_ready(self.state, detection, True))

    def test_stop_rejects_implausibly_close_patch_depth(self):
        detection = self.detection(0)
        detection["spatial_target_depth"] = 2.5
        for _ in range(4):
            self.assertFalse(self.policy._stop_ready(self.state, detection, True))

    def test_stop_requires_category_and_detail_agreement(self):
        detection = self.detection(0)
        detection["category_score"] = 0.29
        detection["detail_score"] = 0.24
        self.assertFalse(self.policy._stop_ready(self.state, detection, True))
        detection["detail_score"] = 0.25
        self.assertTrue(self.policy._stop_ready(self.state, detection, True))

    def test_stop_requires_refined_spatial_semantic_agreement(self):
        detection = self.detection(0)
        detection["spatial_detail_score"] = 0.24
        self.assertFalse(self.policy._stop_ready(self.state, detection, True))

    def test_stop_requires_depth_support_inside_refined_crop(self):
        detection = self.detection(0)
        detection["spatial_depth_support"] = 0.19
        self.assertFalse(self.policy._stop_ready(self.state, detection, True))

    def test_spatial_refinement_matches_semantics_and_depth_in_nested_crop(self):
        self.policy.device = torch.device("cpu")
        self.policy.generic_features = torch.zeros((4, 2), dtype=torch.float32)
        self.policy.depth_analyzer = DepthSafetyAnalyzer()
        self.policy._encode_pil_images = lambda images: torch.tensor(
            [[0.8, 0.6]] * len(images), dtype=torch.float32
        )
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), color=(100, 120, 80)).save(
            buffer, format="PNG"
        )
        rgb = buffer.getvalue()
        observations = [
            {
                "rgb": [rgb] * 4,
                "depth": [np.full((32, 32), 6.0, dtype=np.float32)] * 4,
            }
        ]
        detections = [
            {
                "best_camera": 0,
                "best_score": 0.30,
                "best_patch_score": 0.30,
                "specificity": 0.04,
                "category_score": 0.30,
                "detail_score": 0.30,
                "margin": 0.10,
                "best_box": (0.0, 0.0, 1.0, 1.0),
                "best_crop": "full",
            }
        ]
        feature = torch.tensor([[0.8, 0.6]], dtype=torch.float32)
        self.policy._refine_spatial_stop_evidence(
            observations,
            detections,
            feature,
            feature,
            feature,
        )
        refined = detections[0]
        self.assertTrue(refined["spatial_refined"])
        self.assertEqual(refined["spatial_crop"], "spatial_top_left")
        self.assertAlmostEqual(refined["spatial_score"], 1.0)
        self.assertAlmostEqual(refined["spatial_target_depth"], 6.0)
        self.assertAlmostEqual(refined["spatial_depth_support"], 1.0)

    def test_detected_near_target_uses_short_translation(self):
        detection = self.detection(0)
        detection["target_depth"] = 7.0
        step = self.policy._motion_step(
            "forward", self.clear, detection, detected=True
        )
        self.assertEqual(step, 0.5)

    def test_marginal_clearance_uses_half_step(self):
        marginal = dict(self.clear)
        marginal["forward"] = safety(clearance=7.0)
        step = self.policy._motion_step(
            "forward", marginal, self.detection(0), detected=False
        )
        self.assertEqual(step, 1.0)

    def test_search_selects_only_safe_translation(self):
        directional = dict(self.clear)
        directional["forward"] = safety(clearance=4.0, safe=False, risk=1.0)
        directional["left"] = safety(clearance=4.0, safe=False, risk=1.0)
        observation = {
            "sensors": {
                "state": {
                    "position": [0.0, 0.0, -10.0],
                    "quaternionr": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
        action, _ = self.policy._search_action(
            self.state, observation, directional
        )
        self.assertEqual(action, "right")

    def test_search_translation_budget_stops_by_default(self):
        self.state.search_translations = 32
        observation = {
            "sensors": {
                "state": {
                    "position": [0.0, 0.0, -10.0],
                    "quaternionr": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
        action, reason = self.policy._search_action(
            self.state, observation, self.clear
        )
        self.assertEqual(action, "stop")
        self.assertEqual(reason, "search_translation_budget_exhausted")

    def test_zero_translation_budget_disables_limit(self):
        self.policy.config = PolicyConfig(search_translation_budget=0)
        self.state.search_translations = 10_000
        observation = {
            "sensors": {
                "state": {
                    "position": [0.0, 0.0, -10.0],
                    "quaternionr": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
        action, reason = self.policy._search_action(
            self.state, observation, self.clear
        )
        self.assertNotEqual(action, "stop")
        self.assertNotIn("budget", reason)

    def test_six_consecutive_rotations_force_policy_stop_by_default(self):
        class ClearDepthAnalyzer:
            def __init__(self, clear):
                self.clear = clear

            def analyze(self, _depth):
                return self.clear

            @staticmethod
            def serializable(_safety):
                return {}

        detection = self.detection(0)
        detection["best_score"] = 0.0
        detection["best_patch_score"] = 0.0
        detection["specificity"] = -1.0
        detection["category_score"] = 0.0
        detection["detail_score"] = 0.0
        self.state.recent_actions.extend(["rotr"] * 6)
        self.policy.depth_analyzer = ClearDepthAnalyzer(self.clear)
        self.policy._visual_detections = lambda _observations: [detection]
        self.policy._state_for = lambda _slot, _observation: self.state
        observation = {
            "task_id": "0",
            "depth": [],
            "sensors": {
                "state": {
                    "position": [0.0, 0.0, -10.0],
                    "quaternionr": [0.0, 0.0, 0.0, 1.0],
                }
            },
        }
        actions, _, dones, diagnostics = self.policy.act([observation])
        self.assertEqual(actions[0], "stop")
        self.assertTrue(dones[0])
        self.assertEqual(diagnostics[0]["reason"], "recovery_rotation_limit")

    def test_zero_rotation_limit_disables_rotation_stop(self):
        class ClearDepthAnalyzer:
            def __init__(self, clear):
                self.clear = clear

            def analyze(self, _depth):
                return self.clear

            @staticmethod
            def serializable(_safety):
                return {}

        self.policy.config = PolicyConfig(recovery_rotation_limit=0)
        detection = self.detection(0)
        detection["best_score"] = 0.0
        detection["best_patch_score"] = 0.0
        detection["specificity"] = -1.0
        detection["category_score"] = 0.0
        detection["detail_score"] = 0.0
        self.state.recent_actions.extend(["rotr"] * 6)
        self.policy.depth_analyzer = ClearDepthAnalyzer(self.clear)
        self.policy._visual_detections = lambda _observations: [detection]
        self.policy._state_for = lambda _slot, _observation: self.state
        observation = {
            "task_id": "0",
            "depth": [],
            "sensors": {
                "state": {
                    "position": [0.0, 0.0, -10.0],
                    "quaternionr": [0.0, 0.0, 0.0, 1.0],
                }
            },
        }
        actions, _, dones, diagnostics = self.policy.act([observation])
        self.assertNotEqual(actions[0], "stop")
        self.assertFalse(dones[0])
        self.assertNotEqual(diagnostics[0]["reason"], "recovery_rotation_limit")

    def test_periodic_search_uses_one_rotation_after_twelve_moves(self):
        self.policy.config = PolicyConfig(
            scan_turns=0,
            periodic_scan_turns=1,
            search_moves_per_scan=12,
        )
        self.state.moves_since_scan = 12
        observation = {
            "sensors": {
                "state": {
                    "position": [0.0, 0.0, -10.0],
                    "quaternionr": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }

        first, _ = self.policy._search_action(
            self.state, observation, self.clear
        )
        second, _ = self.policy._search_action(
            self.state, observation, self.clear
        )

        self.assertEqual(first, "rotr")
        self.assertNotIn(second, ("rotl", "rotr"))

    def test_episode_start_does_not_force_rotation(self):
        self.policy.config = PolicyConfig(
            scan_turns=0,
            periodic_scan_turns=1,
            search_moves_per_scan=12,
        )
        observation = {
            "sensors": {
                "state": {
                    "position": [0.0, 0.0, -10.0],
                    "quaternionr": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
        action, _ = self.policy._search_action(
            self.state, observation, self.clear
        )
        self.assertNotIn(action, ("rotl", "rotr"))

    def test_quick_stop_requires_high_confidence_and_close_depth(self):
        self.policy.config = PolicyConfig(
            scan_turns=0, quick_stop_enabled=True
        )
        detection = self.detection(0)
        detection["best_patch_score"] = 0.29
        detection["specificity"] = 0.05
        detection["target_depth"] = 7.0
        self.assertTrue(self.policy._quick_stop_ready(detection, True))
        detection["specificity"] = 0.02
        self.assertFalse(self.policy._quick_stop_ready(detection, True))

    def test_quick_stop_is_disabled_by_default(self):
        detection = self.detection(0)
        detection["best_patch_score"] = 0.40
        detection["specificity"] = 0.10
        detection["target_depth"] = 7.0
        self.assertFalse(self.policy._quick_stop_ready(detection, True))


if __name__ == "__main__":
    unittest.main()
