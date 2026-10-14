import numpy as np
from .geometry import limit, unit


def borders():
    # Public fixed border geometry from core/scenario, not live hidden positions.
    seq = np.arange(-.95, 1., .1)
    return np.concatenate((np.column_stack((seq, np.full(20, -1.))),
                           np.column_stack((np.full(20, 1.), seq)),
                           np.column_stack((seq, np.full(20, 1.))),
                           np.column_stack((np.full(20, -1.), seq))))


def collision_force(positions, obstacles, radii):
    delta = positions[:, None, :]-obstacles[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    penetration = np.logaddexp(0., - (distance-radii-.025)/.001)*.001
    return np.sum(100*delta/np.maximum(distance[..., None], 1e-9)*penetration[..., None], axis=1)


def advance(position, velocity, acceleration, max_speed, dt, obstacles, radii):
    velocity = limit(velocity+(acceleration+collision_force(position, obstacles, radii))*dt, max_speed)
    return position+velocity*dt, velocity
