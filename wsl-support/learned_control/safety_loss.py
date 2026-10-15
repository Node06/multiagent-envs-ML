"""Differentiable training-only risk surrogate; never filters runtime actions."""
import torch
from torch.nn import functional as F


def bounded(vector, maximum):
    return vector/torch.clamp(torch.linalg.vector_norm(vector, dim=-1, keepdim=True)/maximum, min=1.)


def segment_distance(a, b, point):
    delta = b-a
    t = torch.clamp(((point-a)*delta).sum(-1)/torch.clamp((delta*delta).sum(-1), min=1e-8), 0, 1)
    return torch.linalg.vector_norm(point-(a+t[..., None]*delta), dim=-1)


def risk(features, command):
    p = features[:, :2]; v = features[:, 2:4]*.25
    h = p+features[:, 4:6]*2; hv = features[:, 6:8]*.28
    obstacles = p[:, None]+features[:, 10:19].reshape(-1, 3, 3)[:, :, :2]*2
    pending = features[:, 19:21]; q = bounded(command, 1.)
    penalties = []
    for step in range(4):
        oldp, oldh = p, h
        v = bounded(v+.1*(pending if step == 0 else q), .25)
        hv = bounded(hv+.085*(p-h)/torch.clamp(torch.linalg.vector_norm(p-h, dim=-1, keepdim=True), min=1e-6), .28)
        p = p+.1*v; h = h+.1*hv
        if step:  # the queued first step cannot be changed by the proposed action
            od = segment_distance(oldp[:, None], p[:, None], obstacles)
            hd = segment_distance(oldp-oldh, p-h, torch.zeros_like(p))
            penalties.append(F.softplus((.085-od)/.01).mean(-1)
                             +.5*F.softplus((.055-hd)/.01)
                             +F.softplus((torch.max(abs(p), dim=-1).values-.815)/.01))
    return torch.stack(penalties).mean()
