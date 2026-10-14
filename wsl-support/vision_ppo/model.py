import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(hidden, output):
    layers = nn.Sequential(nn.Linear(21, hidden), nn.Tanh(),
                           nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output))
    for module in layers:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, np.sqrt(2))
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(layers[-1].weight, .01 if output == 2 else 1.)
    return layers


class ActorCritic(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.actor = mlp(hidden, 2)
        self.critic = mlp(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), -.5))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-5, 1).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @staticmethod
    def log_prob(dist, latent):
        # Stable log(1-tanh(z)^2), including saturated samples.
        correction = 2 * (np.log(2.) - latent - nn.functional.softplus(-2*latent))
        return (dist.log_prob(latent)-correction).sum(-1)

    def sample(self, obs):
        dist = self.distribution(obs)
        latent = dist.sample()
        return latent.tanh(), latent, self.log_prob(dist, latent), self.value(obs)

    def deterministic(self, obs):
        return self.actor(obs).tanh()
