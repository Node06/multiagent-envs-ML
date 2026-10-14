import numpy as np


def potential(state, cfg):
    return -cfg['potential_scale'] * float(np.linalg.norm(state.detection.goal-state.detection.evader))


def reward(previous_phi, next_phi, reason, cfg):
    # Absorbing terminal states have zero potential; time limits bootstrap.
    terminal = reason is not None and reason != 'timeout'
    shaped = cfg['gamma'] * (0. if terminal else next_phi) - previous_phi
    event = cfg['success_reward'] if reason == 'success' else (
        cfg['failure_reward'] if terminal else 0.)
    return float(event - cfg['step_cost'] + shaped)
