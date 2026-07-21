import unittest

from Aerial_3.config import parse_args


class ConfigTest(unittest.TestCase):
    def test_approach_requires_two_consistent_semantic_frames(self):
        args = parse_args([])
        self.assertEqual(args.approach_confirm_frames, 2)

    def test_none_string_disables_rgbd_output(self):
        args = parse_args(["--image_save_path", "None"])
        self.assertIsNone(args.image_save_path)

    def test_fixed_motion_is_default(self):
        args = parse_args([])
        self.assertTrue(args.is_fixed)
        self.assertEqual(args.xOy_step_size, 2.0)
        self.assertEqual(args.z_step_size, 1.0)
        self.assertEqual(args.rotateAngle, 15.0)
        self.assertEqual(args.safety_margin, 0.5)
        self.assertFalse(args.allow_cpu)
        self.assertEqual(args.scan_turns, 0)
        self.assertEqual(args.periodic_scan_turns, 1)
        self.assertEqual(args.search_moves_per_scan, 12)
        self.assertEqual(args.search_translation_budget, 32)
        self.assertEqual(args.recovery_rotation_limit, 6)
        self.assertFalse(args.resume)
        self.assertEqual(args.stop_confirm_frames, 1)
        self.assertEqual(args.min_stop_depth, 3.0)
        self.assertEqual(args.stop_depth, 10.0)
        self.assertFalse(args.spatial_stop_enabled)
        self.assertEqual(args.min_stop_depth_support, 0.20)
        self.assertFalse(args.quick_stop_enabled)
        self.assertTrue(args.world_model_enabled)
        self.assertEqual(args.jepa_planning_horizon, 6)

    def test_resume_flag(self):
        args = parse_args(["--resume", "true"])
        self.assertTrue(args.resume)

    def test_zero_disables_strategy_limits(self):
        args = parse_args(
            [
                "--search_translation_budget",
                "0",
                "--recovery_rotation_limit",
                "0",
            ]
        )
        self.assertEqual(args.search_translation_budget, 0)
        self.assertEqual(args.recovery_rotation_limit, 0)


if __name__ == "__main__":
    unittest.main()
