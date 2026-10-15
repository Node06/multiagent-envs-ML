"""Safety geometry derived solely from the existing image feature vector."""
import numpy as np

DIM = 56


def augment(base):
    x = np.asarray(base, dtype=np.float32)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    p, v, h, hv, goal, pending = x[:, :2], x[:, 2:4]*.25, x[:, 4:6]*2, x[:, 6:8]*.28, x[:, 8:10]*2, x[:, 19:21]
    obstacles = x[:, 10:19].reshape(-1, 3, 3)
    relative = obstacles[:, :, :2]*2
    dist = np.linalg.norm(relative, axis=-1)
    direction = relative/np.maximum(dist[..., None], 1e-6)
    closing = np.sum(direction*v[:, None], axis=-1)
    speed2 = np.sum(v*v, axis=-1)
    t = np.clip(np.sum(relative*v[:, None], axis=-1)/np.maximum(speed2[:, None], 1e-6), 0, 2)
    closest = np.linalg.norm(relative-v[:, None]*t[..., None], axis=-1)
    gu = goal/np.maximum(np.linalg.norm(goal, axis=-1, keepdims=True), 1e-6)
    projection = np.sum(relative*gu[:, None], axis=-1)
    cross = relative[:, :, 0]*gu[:, None, 1]-relative[:, :, 1]*gu[:, None, 0]
    queued_v = v + .1*pending
    queued_v *= np.minimum(1, .25/np.maximum(np.linalg.norm(queued_v, axis=-1, keepdims=True), 1e-6))
    queued_clearance = np.linalg.norm(relative-.1*queued_v[:, None], axis=-1)-.075
    obs = np.stack(((dist-.075)/.25, direction[..., 0], direction[..., 1], closing/.25,
                    (closest-.075)/.25, t/2, projection, cross, queued_clearance/.25), axis=-1).reshape(-1, 27)
    rv = v-hv
    ht = np.clip(np.sum(h*rv, axis=-1)/np.maximum(np.sum(rv*rv, axis=-1), 1e-6), 0, 2)
    hd = np.linalg.norm(h, axis=-1)
    hunter = np.stack(((hd-.05)/.5, np.sum(h*rv, axis=-1)/np.maximum(hd, 1e-6)/.53,
                       ht/2, (np.linalg.norm(h-rv*ht[:, None], axis=-1)-.05)/.5), axis=-1)
    walls = np.concatenate(((.825-p)/.5, (.825+p)/.5), axis=-1)
    result = np.clip(np.concatenate((x, obs, hunter, walls), axis=-1), -5, 5).astype(np.float32)
    assert result.shape[-1] == DIM
    return result[0] if single else result
