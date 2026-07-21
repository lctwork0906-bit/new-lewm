import unittest

import numpy as np

from Aerial_3.depth_safety import DepthSafetyAnalyzer


def encoded_depth(meters, shape=(256, 256)):
    value = int(round(float(meters) / 100.0 * 255.0))
    return np.full(shape, value, dtype=np.uint8)


class DepthSafetyAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.analyzer = DepthSafetyAnalyzer(
            horizontal_step=2.0,
            vertical_step=1.0,
            safety_margin=0.5,
            percentile=5.0,
        )

    def test_all_four_clear_directions_are_safe(self):
        result = self.analyzer.analyze([encoded_depth(30.0)] * 4)
        self.assertEqual(set(result), {"forward", "left", "right", "descend"})
        self.assertTrue(all(item.safe for item in result.values()))

    def test_obstacle_only_blocks_matching_camera_direction(self):
        result = self.analyzer.analyze(
            [
                encoded_depth(2.0),
                encoded_depth(30.0),
                encoded_depth(30.0),
                encoded_depth(30.0),
            ]
        )
        self.assertFalse(result["forward"].safe)
        self.assertTrue(result["left"].safe)
        self.assertTrue(result["right"].safe)
        self.assertTrue(result["descend"].safe)

    def test_isolated_depth_noise_does_not_block_motion(self):
        depth = encoded_depth(30.0)
        depth[128, 128] = 0
        result = self.analyzer.analyze_image(depth)
        self.assertTrue(result.safe)
        self.assertLess(result.near_fraction, 0.001)

    def test_small_but_material_near_obstacle_fraction_blocks_motion(self):
        depth = encoded_depth(30.0)
        depth[80:176, 112:120] = encoded_depth(2.0, shape=(96, 8))
        result = self.analyzer.analyze_image(depth)
        self.assertFalse(result.safe)
        self.assertGreaterEqual(result.near_fraction, 0.01)

    def test_narrow_obstacle_band_is_not_diluted_by_open_pixels(self):
        depth = encoded_depth(30.0)
        # This band occupies roughly one third of the central flight corridor.
        depth[:, 82:113] = encoded_depth(2.0, shape=(256, 31))
        result = self.analyzer.analyze_image(depth)
        self.assertFalse(result.safe)
        self.assertLess(result.clearance, 2.5)

    def test_clearance_must_be_strictly_greater_than_step_plus_margin(self):
        equal_to_threshold = np.full((256, 256), 2.5, dtype=np.float32)
        above_threshold = np.full((256, 256), 2.51, dtype=np.float32)
        self.assertFalse(self.analyzer.analyze_image(equal_to_threshold).safe)
        self.assertTrue(self.analyzer.analyze_image(above_threshold).safe)

    def test_downward_direction_uses_one_point_five_meter_threshold(self):
        result = self.analyzer.analyze(
            [
                np.full((256, 256), 2.6, dtype=np.float32),
                np.full((256, 256), 2.6, dtype=np.float32),
                np.full((256, 256), 2.6, dtype=np.float32),
                np.full((256, 256), 1.6, dtype=np.float32),
            ]
        )
        self.assertTrue(result["forward"].safe)
        self.assertTrue(result["descend"].safe)
        self.assertEqual(result["forward"].required_clearance, 2.5)
        self.assertEqual(result["descend"].required_clearance, 1.5)

    def test_target_depth_uses_matching_normalized_region(self):
        depth = encoded_depth(40.0)
        depth[64:192, 64:192] = encoded_depth(12.0, shape=(128, 128))
        target_depth = self.analyzer.target_region_depth(
            depth, (0.25, 0.25, 0.75, 0.75)
        )
        self.assertAlmostEqual(target_depth, 12.0, delta=0.5)

    def test_target_region_evidence_measures_stop_window_support(self):
        depth = np.full((100, 100), 30.0, dtype=np.float32)
        depth[:50, :50] = 6.0
        evidence = self.analyzer.target_region_evidence(
            depth,
            (0.0, 0.0, 1.0, 1.0),
            min_depth=3.0,
            max_depth=10.0,
        )
        self.assertAlmostEqual(evidence.depth, 6.0)
        self.assertAlmostEqual(evidence.window_fraction, 0.25)
        self.assertAlmostEqual(evidence.valid_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
