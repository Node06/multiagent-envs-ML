import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NEIGHBORS = 4
INPUT_DIM = 5+4*NEIGHBORS


def inputs(position, velocity, acceleration, kind, obstacles, radii):
    p = np.atleast_2d(position); v = np.atleast_2d(velocity); a = np.atleast_2d(acceleration)
    delta = p[:, None]-np.asarray(obstacles)[None]
    distance = np.linalg.norm(delta, axis=-1)
    order = np.argsort(distance-np.asarray(radii)[None], axis=-1)[:, :NEIGHBORS]
    distance = np.take_along_axis(distance, order, axis=1)
    direction = np.take_along_axis(delta, order[..., None], axis=1)/np.maximum(distance[..., None], 1e-8)
    r = np.asarray(radii)[order]
    geometry = np.concatenate((distance[..., None], direction, r[..., None]), axis=-1)
    return np.concatenate((v, a, np.full((len(p), 1), kind), geometry.reshape(len(p), -1)), axis=-1).astype(np.float32)


class NeuralDynamics(nn.Module):
    """Learned actuation, inertia and speed limits plus a neural contact term.

    Position integration uses the public time step; no original physics function
    is called. Inputs and supervision are entirely image-derived.
    """
    def __init__(self):
        super().__init__()
        self.retention = nn.Parameter(torch.tensor([.9, .9]))
        self.gain = nn.Parameter(torch.tensor([.04, .04]))
        self.cap = nn.Parameter(torch.tensor([.34, .34]))
        self.body_radius = nn.Parameter(torch.tensor(.02))
        self.log_width = nn.Parameter(torch.tensor(-4.6))
        self.contact = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.contact[-1].weight); nn.init.constant_(self.contact[-1].bias, -2.)

    def forward(self, x):
        kind = x[:, 4].long()
        g = x[:, 5:].reshape(-1, NEIGHBORS, 4)
        gap = g[:, :, 0]-g[:, :, 3]-self.body_radius.clamp(.005, .04)
        radial_v = (x[:, None, :2]*g[:, :, 1:3]).sum(-1)
        features = torch.stack((gap/.03, g[:, :, 3]/.15, radial_v/.3, x[:, None, 4].expand_as(gap)), dim=-1)
        strength = F.softplus(self.contact(features).squeeze(-1))
        gate = torch.exp(-F.relu(gap)/self.log_width.exp().clamp(.001, .03))
        correction = ((strength*gate)[..., None]*g[:, :, 1:3]).sum(1)
        v = x[:, :2]*self.retention[kind].clamp(.5, 1.1)[:, None]+x[:, 2:4]*self.gain[kind].clamp(0, .2)[:, None]+correction
        cap = self.cap[kind].clamp(.15, .4)[:, None]
        return v/torch.clamp(torch.linalg.vector_norm(v, dim=-1, keepdim=True)/cap, min=1.)


class WorldModel:
    def __init__(self, models, calibration=None, dt=.1):
        self.models = [m.eval() for m in models]
        self.parameters = [{key: value.detach().cpu().numpy().copy() for key, value in m.state_dict().items()} for m in models]
        self.calibration = np.array(calibration if calibration is not None else [[0, 0], [0, 0]])
        self.dt = dt
        self.prediction_calls = 0

    def speed_cap(self, kind):
        return float(np.mean([np.clip(p['cap'][kind], .15, .4) for p in self.parameters]))

    @staticmethod
    def numpy_predict(x, parameters):
        """Exact trained-network operations, avoiding tiny Torch-call overhead."""
        k = x[:, 4].astype(int)
        g = x[:, 5:].reshape(-1, NEIGHBORS, 4)
        gap = g[:, :, 0]-g[:, :, 3]-np.clip(parameters['body_radius'], .005, .04)
        rv = (x[:, None, :2]*g[:, :, 1:3]).sum(-1)
        f = np.stack((gap/.03, g[:, :, 3]/.15, rv/.3, np.broadcast_to(x[:, None, 4], gap.shape)), axis=-1)
        f = np.tanh(f@parameters['contact.0.weight'].T+parameters['contact.0.bias'])
        f = np.tanh(f@parameters['contact.2.weight'].T+parameters['contact.2.bias'])
        f = (f@parameters['contact.4.weight'].T+parameters['contact.4.bias'])[..., 0]
        strength = np.logaddexp(np.float32(0), f)
        gate = np.exp(-np.maximum(gap, 0)/np.clip(np.exp(parameters['log_width']), .001, .03))
        correction = ((strength*gate)[..., None]*g[:, :, 1:3]).sum(1)
        v = (x[:, :2]*np.clip(parameters['retention'][k], .5, 1.1)[:, None]
             +x[:, 2:4]*np.clip(parameters['gain'][k], 0, .2)[:, None]+correction)
        cap = np.clip(parameters['cap'][k], .15, .4)[:, None]
        return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True)/cap, 1.)

    def advance(self, p, v, a, kind, obstacles, radii):
        x = inputs(p, v, a, kind, obstacles, radii)
        estimates = np.stack([self.numpy_predict(x, p) for p in self.parameters])
        velocity = estimates.mean(0)
        spread = np.linalg.norm(estimates.std(0), axis=-1)
        near = (x[:, 5:].reshape(-1, NEIGHBORS, 4)[:, :, 0]-x[:, 5:].reshape(-1, NEIGHBORS, 4)[:, :, 3]).min(1) < .06
        uncertainty = spread+self.calibration[kind, near.astype(int)]
        self.prediction_calls += len(p)
        return p+self.dt*velocity, velocity, uncertainty


def load_world(path):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    models = []
    for state in cp['models']:
        model = NeuralDynamics(); model.load_state_dict(state); models.append(model)
    return WorldModel(models, cp['calibration'], cp['dt']), cp
