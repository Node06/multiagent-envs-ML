import json
from pathlib import Path
import numpy as np
import torch
from learned_control.perception import PixelNet
from vision_mpc.state_estimator import Estimator
from vision_mpc.global_planner import Planner
from .model import load_world
from .controller import Controller

HERE = Path(__file__).resolve().parent


class LearnedMPC:
    """RGB-only public interface; every predicted transition calls the model."""
    def __init__(self, world, perception, cfg):
        self.world, self.perception, self.cfg = world, perception.eval(), dict(cfg)
        self.reset()

    def reset(self):
        self.estimator = Estimator(self.cfg['dt']); self.controller = None
        self.world.prediction_calls = 0

    def act(self, rgb):
        self.state = self.estimator.update(self.perception.detect(rgb))
        if self.controller is None:
            self.controller = Controller(self.cfg, Planner(self.state.detection, self.cfg), world=self.world)
        command, self.debug = self.controller.choose(self.state)
        return command


def load_controller(checkpoint, config=None):
    path = Path(checkpoint)
    world, cp = load_world(path)
    pixel = PixelNet(); pixel.load_state_dict(torch.load(path.parent/'perception.pt', map_location='cpu', weights_only=True))
    cfg = json.loads((HERE.parent/'vision_mpc/config.json').read_text())
    if config: cfg.update(config)
    return LearnedMPC(world, pixel, cfg), cp


class AnalyticImageMPC:
    """Explicit comparison controller with the same trained CNN, not a fallback."""
    def __init__(self, perception, cfg):
        self.perception, self.cfg = perception.eval(), dict(cfg)
        self.reset()

    def reset(self):
        self.estimator = Estimator(self.cfg['dt']); self.controller = None

    def act(self, rgb):
        from vision_mpc.controller_v2 import Controller as AnalyticController
        self.state = self.estimator.update(self.perception.detect(rgb))
        if self.controller is None:
            self.controller = AnalyticController(self.cfg, Planner(self.state.detection, self.cfg))
        command, self.debug = self.controller.choose(self.state)
        return command
