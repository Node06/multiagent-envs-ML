import numpy as np
import torch
from vision_mpc.state_estimator import Estimator
from vision_mpc.geometry import limit
from vision_ppo.observation import encode
from vision_ppo.model import ActorCritic
from .perception import PixelNet


class LearnedController:
    """Public policy input is raw RGB. No environment or true coordinates accepted.

    Trained CNN masks -> internal visual features -> trained actor. Intermediate
    coordinates are image estimates, not externally supplied simulator state.
    This is a staged trained pipeline, not end-to-end differentiable training.
    """
    def __init__(self, perception, actor, dt=.1, feature_version='base21'):
        self.perception, self.actor, self.dt = perception.eval(), actor.eval(), dt
        self.feature_version = feature_version
        self.reset()

    def reset(self):
        self.estimator = Estimator(self.dt)
        self.pending = np.zeros(2)

    def observe(self, rgb):
        self.state = self.estimator.update(self.perception.detect(rgb))
        self.features = encode(self.state, self.pending)
        if self.feature_version == 'safety56':
            from .safety_features import augment
            self.features = augment(self.features)
        return self.features

    def act(self, rgb):
        features = self.observe(rgb)
        with torch.no_grad():
            action = self.actor.deterministic(torch.from_numpy(features)).numpy()
        self.pending = limit(action, 1.)
        return self.pending.copy()


def load_controller(path):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    pixel = PixelNet(); pixel.load_state_dict(cp['perception'])
    version = cp.get('feature_version', 'base21')
    if version == 'safety56':
        if cp.get('policy_type') == 'categorical129':
            from .directional_model import DirectionalActorCritic
            actor = DirectionalActorCritic(cp['hidden'], temperature=cp.get('temperature', .4))
        else:
            from .safety_model import SafetyActorCritic
            actor = SafetyActorCritic(cp['hidden'])
    elif version == 'base21':
        actor = ActorCritic(cp['hidden'])
    else:
        raise ValueError('Unsupported feature version: '+version)
    actor.load_state_dict(cp['model'])
    return LearnedController(pixel, actor, cp['config']['dt'], version), cp
