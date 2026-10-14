import ast
import json
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from vision_mpc.perception import detect, DetectionError, pixel_to_world
from vision_mpc.geometry import action5, segment_distance
from vision_mpc.state_estimator import Estimator
from vision_mpc.global_planner import Planner
from vision_mpc.dynamics import advance
from vision_mpc.controller import Controller
from vision_mpc.controller_v2 import Controller as ImprovedController
from vision_mpc.stratified_cases import generate, valid, GROUPS
from vision_mpc.evaluator import terminal_reason, hunter_obstacle_distance
from types import SimpleNamespace


def image():
    rgb = np.full((800, 800, 3), 255, np.uint8)
    cv2.circle(rgb, (600, 200), 10, (172, 236, 172), -1)
    cv2.circle(rgb, (400, 400), 10, (236, 172, 172), -1)
    cv2.rectangle(rgb, (190, 590), (210, 610), (204, 153, 204), -1)
    for point in [(260, 260), (540, 260), (400, 540)]:
        cv2.circle(rgb, point, 20, (64, 64, 64), -1)
    return rgb


class CoreTests(unittest.TestCase):
    def test_v2_recovers_from_goal_biased_close_pursuit(self):
        cfg = json.loads((HERE/'config.json').read_text())
        rgb = image()
        rgb[:] = 255
        cv2.circle(rgb, (306, 676), 10, (172, 236, 172), -1)
        cv2.circle(rgb, (336, 661), 10, (236, 172, 172), -1)
        cv2.rectangle(rgb, (532, 352), (552, 372), (204, 153, 204), -1)
        for point in [(260, 260), (540, 260), (400, 540)]:
            cv2.circle(rgb, point, 20, (64, 64, 64), -1)
        state = Estimator(.1).update(detect(rgb))
        controller = ImprovedController(cfg, Planner(state.detection, cfg))
        command, debug = controller.choose(state)
        self.assertTrue(debug['recovery'])
        # The old close-pursuit failure accelerated right, toward the hunter.
        self.assertLess(command[0], 0)
        self.assertTrue(np.isfinite(command).all())
        self.assertLessEqual(np.linalg.norm(command), 1.000001)

    def test_stratified_cases_are_valid_and_reproducible(self):
        first = generate(2, 20261013)
        self.assertEqual(first, generate(2, 20261013))
        self.assertEqual(len(first), 2*len(GROUPS))
        self.assertEqual(len({c['id'] for c in first}), len(first))
        for case in first:
            self.assertTrue(valid(np.array([case['hunter'], case['evader'], case['goal']]), case['group']))
            if case['group'].startswith('hunter_gap_'):
                bounds = {'hunter_gap_008_012': (.08, .12), 'hunter_gap_012_020': (.12, .20),
                          'hunter_gap_020_035': (.20, .35)}[case['group']]
                self.assertGreaterEqual(case['initial_hunter_gap'], bounds[0])
                self.assertLessEqual(case['initial_hunter_gap'], bounds[1])

    def test_hunter_obstacle_contact_does_not_become_failure(self):
        entity = lambda p: SimpleNamespace(state=SimpleNamespace(p_pos=np.array(p, dtype=float)))
        env = SimpleNamespace(world=SimpleNamespace(agents=[entity([0, .06]), entity([.5, .5])],
                                                   landmarks=[entity([0, 0])], check=[entity([-.5, -.5])]))
        self.assertLess(hunter_obstacle_distance(env), .075)
        self.assertIsNone(terminal_reason(env, [False, False]))

    def test_obstacle_done_is_not_goal_success(self):
        entity = lambda p: SimpleNamespace(state=SimpleNamespace(p_pos=np.array(p, dtype=float)))
        env = SimpleNamespace(world=SimpleNamespace(agents=[entity([.6, .6]), entity([0, .06])],
                                                   landmarks=[entity([0, 0])], check=[entity([-.5, -.5])]))
        self.assertEqual(terminal_reason(env, [False, True]), 'obstacle_collision')

    def test_rgb_coordinates_and_border_exclusion(self):
        rgb = image()
        rgb[:20] = (204, 102, 102)
        d = detect(rgb)
        np.testing.assert_allclose(d.evader, [.50125, .49875], atol=1e-6)
        self.assertEqual(len(d.obstacles), 3)
        np.testing.assert_allclose(d.hunter, [.00125, -.00125], atol=1e-6)

    def test_missing_detection_is_explicit(self):
        with self.assertRaises(DetectionError):
            detect(np.full((800, 800, 3), 255, np.uint8))

    def test_velocity_from_images(self):
        first = detect(image()); second = detect(image())
        second.evader = second.evader + [.01, -.02]
        est = Estimator(.1)
        np.testing.assert_allclose(est.update(first).evader_velocity, [0, 0])
        np.testing.assert_allclose(est.update(second).evader_velocity, [.1, -.2])

    def test_five_dimensional_signed_action(self):
        np.testing.assert_allclose(action5([-.5, .25]), [0, 0, .5, .25, 0])
        np.testing.assert_allclose(action5([0, 0]), [1, 0, 0, 0, 0])

    def test_swept_collision_and_zero_segment(self):
        self.assertAlmostEqual(float(segment_distance(np.array([-1., 0]), np.array([1., 0]), np.array([0., .1]))), .1)
        self.assertAlmostEqual(float(segment_distance(np.zeros(2), np.zeros(2), np.array([1., 0]))), 1.)

    def test_astar_avoids_inflated_obstacles(self):
        cfg = json.loads((HERE/'config.json').read_text())
        d = detect(image()); plan = Planner(d, cfg); route = plan.plan(d.evader)
        self.assertLess(np.linalg.norm(route[-1]-d.goal), .075)
        for a, b in zip(route, route[1:]):
            self.assertGreaterEqual(float(segment_distance(a, b, d.obstacles).min()), .085)

    def test_semi_implicit_dynamics_and_speed_limit(self):
        p, v = advance(np.zeros((1, 2)), np.zeros((1, 2)), np.array([[1., 0]]),
                       .25, .1, np.empty((0, 2)), np.empty(0))
        np.testing.assert_allclose(p, [[.01, 0]])
        np.testing.assert_allclose(v, [[.1, 0]])
        _, v = advance(p, np.array([[.25, 0]]), np.array([[1., 0]]), .25, .1,
                       np.empty((0, 2)), np.empty(0))
        self.assertAlmostEqual(float(np.linalg.norm(v)), .25)

    def test_delay_first_prediction_ignores_new_command(self):
        cfg = json.loads((HERE/'config.json').read_text()); cfg['horizon'] = 2
        d = detect(image()); state = Estimator(.1).update(d)
        controller = Controller(cfg, Planner(d, cfg))
        controller.route = controller.planner.plan(d.evader)
        noise = np.array([[[1., 0]], [[-1., 0]]])
        _, _, trajectory, _ = controller.rollout(state, noise)
        np.testing.assert_allclose(trajectory[0, 0], trajectory[0, 1])
        self.assertGreater(np.linalg.norm(trajectory[1, 0]-trajectory[1, 1]), 1e-4)

    def test_policy_modules_have_no_hidden_state_access(self):
        for name in ['controller.py', 'controller_v1.py', 'controller_v2.py', 'perception.py', 'state_estimator.py', 'global_planner.py']:
            tree = ast.parse((HERE/name).read_text())
            forbidden = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                         and n.attr in {'world', 'p_pos', 'p_vel', 'agents'}]
            self.assertEqual(forbidden, [], name)


if __name__ == '__main__':
    unittest.main()
