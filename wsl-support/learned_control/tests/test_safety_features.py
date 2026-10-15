from pathlib import Path
import sys
import unittest
import numpy as np
import torch
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from learned_control.safety_features import augment, DIM
from learned_control.safety_model import SafetyActorCritic
from vision_ppo.model import ActorCritic
from learned_control.optimize import difficult_cases
from learned_control.optimize import regularized_update
from learned_control.safety_loss import risk


class SafetyTests(unittest.TestCase):
    def test_feature_extension_preserves_pretrained_actor(self):
        torch.set_num_threads(1); torch.manual_seed(1)
        old = ActorCritic(256); new = SafetyActorCritic(256)
        new.initialize_from(old.state_dict())
        x = np.random.default_rng(1).normal(0, .2, (10, 21)).astype(np.float32)
        with torch.no_grad():
            torch.testing.assert_close(old.deterministic(torch.from_numpy(x)), new.deterministic(torch.from_numpy(augment(x))))

    def test_closing_motion_predicts_obstacle_and_hunter_approach(self):
        x = np.zeros(21, np.float32)
        x[2] = 1  # prey vx=.25
        x[4] = .1  # hunter .2 ahead
        x[8] = .3
        x[10:19] = [.1, 0, .5, -.2, .2, .5, .2, -.2, .5]
        a = augment(x)
        self.assertEqual(a.shape, (DIM,))
        self.assertTrue(np.isfinite(a).all())
        self.assertAlmostEqual(a[24], 1)  # radial closing / .25
        self.assertAlmostEqual(a[25], -.3, places=5)  # closest net clearance
        self.assertAlmostEqual(a[50], .4, places=5)  # hunter closest time .8s /2
        self.assertAlmostEqual(a[51], -.1, places=5)

    def test_difficult_cases_keep_original_fixed_obstacles_and_legal_starts(self):
        cases = difficult_cases(30, 44)
        self.assertEqual(cases, difficult_cases(30, 44))
        obstacles = np.array([[-.35, .35], [.35, .35], [0, -.35]])
        for case in cases:
            self.assertNotIn('obstacles', case)
            points = np.array([case[n] for n in ('hunter', 'evader', 'goal')])
            self.assertGreaterEqual(np.min(np.linalg.norm(points[:, None]-obstacles[None], axis=-1)), .13)
            self.assertGreaterEqual(np.linalg.norm(points[0]-points[1]), .25)

    def test_joint_ppo_imitation_update_is_finite_and_updates_the_actor(self):
        torch.set_num_threads(1); torch.manual_seed(4)
        rng = np.random.default_rng(4); actor = SafetyActorCritic(32)
        x = rng.normal(0, .2, (64, DIM)).astype(np.float32)
        with torch.no_grad():
            _, z, lp, _ = actor.sample(torch.from_numpy(x))
        before = actor.actor[0].weight.detach().clone()
        cfg = dict(epochs=2, batch_size=32, target_kl=.02, clip=.1, value_coef=.5, bc_coef=1, entropy_coef=.0001, safety_coef=.1)
        data = [x, z.numpy(), lp.numpy(), rng.normal(size=64), rng.normal(size=64)]
        result = regularized_update(actor, torch.optim.Adam(actor.parameters(), lr=1e-4), data, cfg,
                                    torch.from_numpy(x), torch.zeros(64, 2), rng)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        self.assertFalse(torch.equal(before, actor.actor[0].weight))

    def test_training_risk_gradient_discourages_moving_into_an_obstacle(self):
        x = np.zeros((1, 21), np.float32); x[0, 2] = .8; x[0, 4] = -.5
        x[0, 10:19] = [.075, 0, .5, -.2, .2, .5, -.2, -.2, .5]
        features = torch.from_numpy(augment(x))
        action = torch.tensor([[.2, 0.]], requires_grad=True)
        loss = risk(features, action); loss.backward()
        self.assertGreater(action.grad[0, 0].item(), 0)
        self.assertGreater(risk(features, torch.tensor([[1., 0.]])).item(), risk(features, torch.tensor([[-1., 0.]])).item())


if __name__ == '__main__': unittest.main()
