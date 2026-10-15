"""Categorical bounded acceleration policy: choose modes instead of averaging."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .safety_features import DIM


class DirectionalActorCritic(nn.Module):
    def __init__(self, hidden=256, directions=32, temperature=.4):
        super().__init__()
        angles = np.arange(directions)*2*np.pi/directions
        vectors = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        table = np.concatenate((np.zeros((1, 2)), *[scale*vectors for scale in (.25, .5, .75, 1)]))
        self.register_buffer('action_table', torch.tensor(table, dtype=torch.float32))
        self.temperature = temperature
        self.actor = nn.Sequential(nn.Linear(DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, len(table)))
        self.critic = nn.Sequential(nn.Linear(DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2)); nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, .01)

    def initialize_backbone(self, state):
        own = self.state_dict()
        for key in own:
            if key in state and own[key].shape == state[key].shape:
                own[key].copy_(state[key])
        self.load_state_dict(own)

    def labels(self, command):
        return ((command[:, None]-self.action_table[None]).square().sum(-1)).argmin(-1)

    def distribution(self, obs):
        return Categorical(logits=self.actor(obs)/self.temperature)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def sample(self, obs):
        dist = self.distribution(obs); index = dist.sample()
        return self.action_table[index], index, dist.log_prob(index), self.value(obs)

    def deterministic(self, obs):
        return self.action_table[self.actor(obs).argmax(-1)]
