import numpy as np


def encode(state, pending):
    """21 fixed-scale features. Accepts visual estimates, never an environment."""
    d = state.detection
    # Detection sorts obstacles left-to-right, so feature slots remain stable.
    obstacle = np.column_stack(((d.obstacles-d.evader)/2, d.obstacle_radii/.1))
    return np.concatenate((d.evader, state.evader_velocity/.25,
                           (d.hunter-d.evader)/2, state.hunter_velocity/.28,
                           (d.goal-d.evader)/2, obstacle.ravel(), pending)).astype(np.float32)
