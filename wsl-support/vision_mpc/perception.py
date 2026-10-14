from dataclasses import dataclass
import cv2
import numpy as np


class DetectionError(RuntimeError):
    pass


@dataclass
class Detection:
    evader: np.ndarray
    hunter: np.ndarray
    goal: np.ndarray
    obstacles: np.ndarray
    obstacle_radii: np.ndarray
    pixels: dict


def pixel_to_world(points, width, height):
    p = np.asarray(points, dtype=float)
    return np.stack((2 * (p[..., 0] + .5) / width - 1,
                     1 - 2 * (p[..., 1] + .5) / height), axis=-1)


def components(mask, min_area=30, max_area=4000):
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(centers[i], int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)
            if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]


def detect(rgb):
    """Only pixels are accepted; no environment, initialization or true positions."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DetectionError('Expected an RGB image')
    # OpenCV uint8 kernels avoid multiple full-frame int16 temporaries.
    # Saturating subtraction is equivalent for the positive-difference tests.
    r, g, b = cv2.split(rgb)
    gt = lambda a, threshold: cv2.compare(a, threshold, cv2.CMP_GT)
    diff = lambda a, b: gt(cv2.subtract(a, b), 25)
    masks = {
        'evader': diff(g, r) & diff(g, b) & gt(g, 215),
        'hunter': diff(r, g) & diff(r, b) & gt(r, 215),
        'goal': diff(r, g) & diff(b, g) & cv2.compare(cv2.absdiff(r, b), 15, cv2.CMP_LT) & gt(r, 180),
        'obstacles': cv2.inRange(rgb, (0, 0, 0), (99, 99, 99)),
    }
    pixels, positions = {}, {}
    h, w = rgb.shape[:2]
    for name in ('evader', 'hunter', 'goal'):
        found = components(masks[name])
        if len(found) != 1:
            raise DetectionError(f'{name}: expected 1 component, got {len(found)}')
        pixels[name] = found[0][0]
        positions[name] = pixel_to_world(found[0][0], w, h)
    obs = components(masks['obstacles'], min_area=200)
    if len(obs) != 3:
        raise DetectionError(f'obstacles: expected 3 components, got {len(obs)}')
    obs.sort(key=lambda item: (item[0][0], item[0][1]))
    pixels['obstacles'] = np.array([item[0] for item in obs])
    centers = pixel_to_world(pixels['obstacles'], w, h)
    radii = np.sqrt(np.array([item[1] for item in obs]) / np.pi) * 2 / w
    return Detection(positions['evader'], positions['hunter'], positions['goal'],
                     centers, radii, pixels)
