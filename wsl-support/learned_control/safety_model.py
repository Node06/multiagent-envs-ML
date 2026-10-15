import numpy as np
import torch
from torch import nn
from vision_ppo.model import ActorCritic as Base
from .safety_features import DIM


class SafetyActorCritic(Base):
    def __init__(self, hidden=256):
        super().__init__(hidden)
        for name in ('actor', 'critic'):
            old = getattr(self, name)
            first = nn.Linear(DIM, hidden)
            nn.init.orthogonal_(first.weight, np.sqrt(2)); nn.init.zeros_(first.bias)
            setattr(self, name, nn.Sequential(first, *list(old.children())[1:]))

    def initialize_from(self, state):
        own = self.state_dict()
        for key, value in state.items():
            if key.endswith('0.weight'):
                own[key].zero_(); own[key][:, :21].copy_(value)
            else:
                own[key].copy_(value)
        self.load_state_dict(own)
