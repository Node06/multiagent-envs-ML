import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def limit(v, maximum):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True) / maximum, 1)


def segment_distance(a, b, points):
    """Broadcast segment/point distances, including zero-length segments."""
    delta = b-a
    t = np.sum((points-a)*delta, axis=-1) / np.maximum(np.sum(delta*delta, axis=-1), 1e-12)
    projection = a + np.clip(t, 0, 1)[..., None]*delta
    return np.linalg.norm(points-projection, axis=-1)


def action5(q):
    q = limit(np.asarray(q, dtype=float), 1.)
    return np.array([float(np.linalg.norm(q) < 1e-9), max(q[0], 0), max(-q[0], 0),
                     max(q[1], 0), max(-q[1], 0)])
