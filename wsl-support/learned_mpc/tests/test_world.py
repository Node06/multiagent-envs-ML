from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learned_mpc.model import NeuralDynamics, WorldModel, inputs
from learned_mpc.controller import Controller
from vision_mpc.perception import Detection
from vision_mpc.state_estimator import State
from vision_mpc.global_planner import Planner
import json


class WorldTests(unittest.TestCase):
    def test_learned_weights_change_the_predicted_response(self):
        first = NeuralDynamics(); second = NeuralDynamics()
        second.load_state_dict(first.state_dict())
        with torch.no_grad(): second.gain.fill_(.12)
        obstacles = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        radii = np.full(4, .05)
        p, v, a = np.zeros((1, 2)), np.zeros((1, 2)), np.array([[1., 0.]])
        a_p, _, _ = WorldModel([first]).advance(p, v, a, 0, obstacles, radii)
        b_p, _, _ = WorldModel([second]).advance(p, v, a, 0, obstacles, radii)
        self.assertGreater(b_p[0, 0], a_p[0, 0]*2)

    def test_numpy_is_equivalent_to_the_trained_torch_network(self):
        torch.set_num_threads(1); torch.manual_seed(8)
        model = NeuralDynamics(); world = WorldModel([model])
        rng = np.random.default_rng(8)
        obstacles = rng.uniform(-.7, .7, (8, 2)); radii = np.full(8, .05)
        x = inputs(rng.uniform(-.7, .7, (100, 2)), rng.normal(0, .1, (100, 2)), rng.uniform(-1, 1, (100, 2)), 1, obstacles, radii)
        with torch.no_grad(): expected = model(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(world.numpy_predict(x, world.parameters[0]), expected, atol=1e-6, rtol=1e-5)

    def test_model_rollout_uses_no_original_advance(self):
        model = NeuralDynamics(); world = WorldModel([model])
        d = Detection(np.array([.6, .6]), np.array([-.5, -.5]), np.array([-.6, .5]),
                      np.array([[-.35, .35], [0, -.35], [.35, .35]]), np.full(3, .05), {})
        cfg = json.loads((Path(__file__).resolve().parents[2]/'vision_mpc/config.json').read_text())
        cfg.update(horizon=3, samples=8, iterations=1)
        control = Controller(cfg, Planner(d, cfg), world=world)
        control.route = control.planner.plan(d.evader)
        state = State(d, np.zeros(2), np.zeros(2))
        with patch('vision_mpc.dynamics.advance', side_effect=AssertionError('Analytic fallback forbidden')):
            result = control.rollout(state, np.zeros((8, 1, 2)))
        self.assertEqual(result[0].shape, (8,))
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertGreater(world.prediction_calls, 0)


if __name__ == '__main__': unittest.main()
