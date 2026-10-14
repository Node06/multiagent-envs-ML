from dataclasses import dataclass
import numpy as np
from .perception import Detection


@dataclass
class State:
    detection: Detection
    evader_velocity: np.ndarray
    hunter_velocity: np.ndarray


class Estimator:
    def __init__(self, dt):
        self.dt, self.previous = dt, None

    def update(self, detection):
        if self.previous is None:
            ve = vh = np.zeros(2)
        else:
            ve = (detection.evader - self.previous.evader) / self.dt
            vh = (detection.hunter - self.previous.hunter) / self.dt
        self.previous = detection
        return State(detection, ve, vh)
