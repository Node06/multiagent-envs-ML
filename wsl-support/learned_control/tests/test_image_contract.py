import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE.parent), str(ROOT)]
from course_audit.audit import create_case, errors, original_files, random_obstacle_cases
from vision_mpc.env_adapter import render_rgb, close
from learned_control.perception import PixelNet
from learned_control.runtime import LearnedController
from vision_ppo.model import ActorCritic


class ImageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.pixel = PixelNet()
        cls.pixel.load_state_dict(torch.load(HERE/'outputs/main_20261014/perception.pt', map_location='cpu', weights_only=True))
        cls.pixel.eval()

    def test_learned_detector_follows_moved_objects_and_obstacles(self):
        for case in random_obstacle_cases(8, 2026101488):
            env = create_case(case)
            try:
                detected = self.pixel.detect(render_rgb(env))
                diagnostic = errors(env, detected)
                self.assertLess(max(diagnostic[k] for k in ('hunter', 'evader', 'goal', 'obstacles')), 3.)
            finally:
                close(env)

    def test_rgb_policy_outputs_bounded_action_without_environment_argument(self):
        case = random_obstacle_cases(1, 2026101489)[0]
        env = create_case(case)
        try:
            model = LearnedController(self.pixel, ActorCritic(256))
            action = model.act(render_rgb(env))
            self.assertEqual(action.shape, (2,))
            self.assertTrue(np.isfinite(action).all())
            self.assertLessEqual(np.linalg.norm(action), 1.000001)
        finally:
            close(env)

    def test_original_integrity_with_cross_platform_line_endings(self):
        recorded = json.loads((HERE.parent/'course_audit/original_integrity.json').read_text())
        self.assertEqual(original_files(), recorded['sha256'])


if __name__ == '__main__':
    unittest.main()
