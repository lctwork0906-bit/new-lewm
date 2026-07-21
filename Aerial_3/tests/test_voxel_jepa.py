import unittest

import numpy as np
import torch

from Aerial_3.config import PolicyConfig
from Aerial_3.depth_safety import DirectionSafety
from Aerial_3.voxel_jepa import RGBDVoxelizer, VoxelJEPA, VoxelJEPAPlanner, VoxelSpec


def observation(task_id="0", depth=8.0):
    return {
        "task_id": task_id,
        "depth": [np.full((32, 32), depth, dtype=np.float32) for _ in range(4)],
        "sensors": {
            "state": {
                "position": [0.0, 0.0, -10.0],
                "quaternionr": [0.0, 0.0, 0.0, 1.0],
            }
        },
    }


def safe_direction():
    return DirectionSafety(20.0, 2.5, 0.0, 1.0, True, 0.0)


class VoxelJEPATest(unittest.TestCase):
    def test_rgbd_projection_builds_three_channel_voxel_state(self):
        spec = VoxelSpec(samples_per_axis=6)
        grid, endpoints = RGBDVoxelizer(spec).build(observation())
        self.assertEqual(grid.shape, (3, 8, 24, 24))
        self.assertGreater(len(endpoints), 0)
        self.assertGreater(float(grid[0].sum()), 0.0)
        self.assertGreater(float(grid[2].sum()), float(grid[0].sum()))

    def test_jepa_rollout_predicts_full_requested_horizon(self):
        spec = VoxelSpec()
        model = VoxelJEPA(spec, latent_dim=16, hidden_dim=24)
        voxels = torch.zeros((1, 3, 8, 24, 24))
        actions = torch.tensor([[0, 0, 4, 0], [2, 2, 5, 0]])
        rollout = model.rollout(voxels, actions)
        self.assertEqual(tuple(rollout.shape), (2, 5, 16))

    def test_planner_never_selects_an_unsafe_translation(self):
        config = PolicyConfig(
            voxel_samples_per_axis=6,
            jepa_latent_dim=16,
            jepa_hidden_dim=24,
            jepa_planning_horizon=2,
            jepa_beam_width=4,
            jepa_online_training=False,
        )
        planner = VoxelJEPAPlanner(config, torch.device("cpu"))
        obs = observation()
        planner.observe(0, obs)
        unsafe = DirectionSafety(1.0, 2.5, 0.5, 1.0, False, 1.0)
        safety = {
            "forward": unsafe,
            "left": safe_direction(),
            "right": safe_direction(),
            "descend": safe_direction(),
        }
        detection = {"cameras": [{"score": value} for value in (0.4, 0.2, 0.3, 0.1)]}
        action, diagnostic = planner.plan(
            0, obs, detection, safety, "forward", detected=False
        )
        self.assertNotEqual(action, "forward")
        self.assertIn(action, ("left", "right", "rotl", "rotr"))
        self.assertEqual(diagnostic["planning_horizon"], 2)

    def test_online_transition_enters_replay_without_target_labels(self):
        config = PolicyConfig(
            voxel_samples_per_axis=6,
            jepa_latent_dim=16,
            jepa_hidden_dim=24,
            jepa_online_training=False,
        )
        planner = VoxelJEPAPlanner(config, torch.device("cpu"))
        planner.observe(0, observation(depth=8.0))
        planner.commit(0, "forward")
        planner.observe(0, observation(depth=7.0))
        self.assertEqual(len(planner.replay), 1)


if __name__ == "__main__":
    unittest.main()
