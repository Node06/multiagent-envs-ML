import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.distributions import TransformedDistribution, TanhTransform

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from vision_ppo.buffer import gae
from vision_ppo.model import ActorCritic
from vision_ppo.rewards import reward
from vision_ppo.ppo import update
from vision_mpc.geometry import action5
import cv2


class CoreTests(unittest.TestCase):
    def test_opencv_color_masks_match_original_thresholds(self):
        image = np.random.default_rng(7).integers(0, 256, (180, 180, 3), dtype=np.uint8)
        r, g, b = np.moveaxis(image.astype(np.int16), -1, 0)
        old = [(g > r+25) & (g > b+25) & (g > 215),
               (r > g+25) & (r > b+25) & (r > 215),
               (r > g+25) & (b > g+25) & (abs(r-b) < 15) & (r > 180),
               np.maximum(np.maximum(r, g), b) < 100]
        r, g, b = cv2.split(image)
        gt = lambda a, t: cv2.compare(a, t, cv2.CMP_GT)
        diff = lambda a, b: gt(cv2.subtract(a, b), 25)
        new = [diff(g, r) & diff(g, b) & gt(g, 215),
               diff(r, g) & diff(r, b) & gt(r, 215),
               diff(r, g) & diff(b, g) & cv2.compare(cv2.absdiff(r, b), 15, cv2.CMP_LT) & gt(r, 180),
               cv2.inRange(image, (0, 0, 0), (99, 99, 99))]
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a, b > 0)

    def test_gae_truncation_bootstraps_without_cross_episode_leak(self):
        a, _ = gae(np.array([1., 100.]), np.zeros(2), np.array([10., 0.]),
                   np.array([False, True]), np.array([True, True]), .9, .95)
        np.testing.assert_allclose(a, [10., 100.])

    def test_gae_terminal_does_not_bootstrap(self):
        a, _ = gae(np.array([1.]), np.array([2.]), np.array([100.]),
                   np.array([True]), np.array([True]), .99, .95)
        np.testing.assert_allclose(a, [-1.])

    def test_bounded_policy_logprob_matches_torch_transform(self):
        model = ActorCritic()
        dist = model.distribution(torch.zeros(3, 21))
        z = torch.tensor([[.1, -.4], [1., -1.], [2., 0.]])
        transformed = TransformedDistribution(dist, [TanhTransform(cache_size=1)])
        torch.testing.assert_close(model.log_prob(dist, z), transformed.log_prob(z.tanh()).sum(-1))
        self.assertTrue(torch.isfinite(model.log_prob(dist, torch.tensor([[40., -40.]]))).all())

    def test_action_mapping_preserves_small_and_zero_controls(self):
        np.testing.assert_allclose(action5([.1, -.2])[1:], [.1, 0., 0., .2])
        np.testing.assert_allclose(action5([0., 0.])[1:], np.zeros(4))

    def test_potential_absorbing_terminal_and_timeout(self):
        cfg = {'gamma': .9, 'success_reward': 20., 'failure_reward': -20., 'step_cost': .01}
        self.assertAlmostEqual(reward(-5., -4., 'success', cfg), 24.99)
        self.assertAlmostEqual(reward(-5., -4., 'timeout', cfg), 1.39)

    def test_ppo_update_changes_weights_and_is_finite(self):
        torch.manual_seed(5); np.random.seed(5)
        model = ActorCritic(32); opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        obs = torch.randn(64, 21)
        with torch.no_grad():
            _, z, lp, v = model.sample(obs)
        before = model.actor[0].weight.detach().clone()
        cfg = {'epochs': 2, 'batch_size': 32, 'clip': .2, 'value_coef': .5,
               'entropy_coef': .01, 'max_grad_norm': .5, 'target_kl': .1}
        stats = update(model, opt, [obs.numpy(), z.numpy(), lp.numpy(), np.linspace(-1, 1, 64), v.numpy()+1], cfg)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertFalse(torch.equal(before, model.actor[0].weight))


if __name__ == '__main__':
    unittest.main()
