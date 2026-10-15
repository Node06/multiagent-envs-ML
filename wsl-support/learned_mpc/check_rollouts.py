"""Open-loop held-out replay given recorded actions and initial visual geometry."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learned_mpc.model import load_world

HERE = Path(__file__).resolve().parent


def main():
    folder = HERE/'outputs/main_20261014'
    data = np.load(folder/'transitions.npz'); world, _ = load_world(folder/'world_empirical.pt')
    results = {n: [] for n in (1, 5, 10)}
    for episode in np.unique(data['episode'][data['split'] == 'test']):
        for kind in (0, 1):
            indices = np.flatnonzero((data['episode'] == episode) & (data['x'][:, 4] == kind))
            x, y = data['x'][indices], data['y'][indices]
            for horizon in results:
                for start in range(0, len(indices)-horizon+1, horizon):
                    # Missing/filtered frames may not be treated as adjacent.
                    if horizon > 1 and not np.allclose(x[start+1:start+horizon, :2], y[start:start+horizon-1], atol=1e-6): continue
                    g = x[start, 5:].reshape(4, 4)
                    obstacles = -g[:, 0, None]*g[:, 1:3]; radii = g[:, 3]
                    p = np.zeros((1, 2)); v = x[start:start+1, :2]
                    for t in range(horizon):
                        p, v, _ = world.advance(p, v, x[start+t:start+t+1, 2:4], kind, obstacles, radii)
                    target = y[start:start+horizon].sum(0)*.1
                    results[horizon].append(float(np.linalg.norm(p[0]-target)*400))
    summary = {'protocol': 'held-out episodes; recorded future actions supplied; initial four visual neighbors fixed; no future positions fed to the model',
               'pixels_reference': '400x400 visual coordinates',
               'horizons': {n: {'windows': len(v), 'mean_position_error_pixels': float(np.mean(v)),
                               'p95_position_error_pixels': float(np.quantile(v, .95))} for n, v in results.items()}}
    (folder/'open_loop_metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
