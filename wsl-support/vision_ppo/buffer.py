import numpy as np


def gae(rewards, values, next_values, terminated, ended, gamma, lam):
    """Bootstrap truncations, but never propagate GAE across episode resets."""
    advantage = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.
    for i in range(len(rewards)-1, -1, -1):
        delta = rewards[i] + gamma * next_values[i] * (1-terminated[i]) - values[i]
        carry = delta + gamma*lam*(1-ended[i])*carry
        advantage[i] = carry
    return advantage, advantage + values
