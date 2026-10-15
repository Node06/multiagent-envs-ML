from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learned_control.directional_model import DirectionalActorCritic


class DirectionalTests(unittest.TestCase):
    def test_equal_opposite_modes_choose_a_direction_instead_of_averaging(self):
        torch.set_num_threads(1)
        actor = DirectionalActorCritic(32)
        for p in actor.actor.parameters(): p.data.zero_()
        # Full acceleration in the positive and negative y directions.
        actor.actor[-1].bias.data[105] = 10
        actor.actor[-1].bias.data[121] = 10
        chosen = actor.deterministic(torch.zeros(56))
        self.assertAlmostEqual(torch.linalg.vector_norm(chosen).item(), 1)
        self.assertGreater(abs(chosen[1].item()), .99)

    def test_actions_are_bounded_and_sampling_logprob_matches_the_categorical_policy(self):
        actor = DirectionalActorCritic(32)
        self.assertLessEqual(torch.linalg.vector_norm(actor.action_table, dim=-1).max().item(), 1.000001)
        x = torch.zeros(10, 56)
        action, index, lp, _ = actor.sample(x)
        torch.testing.assert_close(lp, actor.distribution(x).log_prob(index))
        torch.testing.assert_close(action, actor.action_table[index])
        self.assertTrue(torch.isfinite(action).all())


if __name__ == '__main__': unittest.main()
