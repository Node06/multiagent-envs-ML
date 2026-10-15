"""Compare frozen learned and untrained predictions on held-out image rows."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from learned_mpc.model import NeuralDynamics, WorldModel, load_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=HERE/'outputs/main_20261014')
    args = parser.parse_args()
    folder = args.experiment
    data = np.load(folder/'transitions.npz')
    selected = data['split'] == 'test'
    x, y = data['x'][selected], data['y'][selected]
    trained, _ = load_world(folder/'world_empirical.pt')
    torch.manual_seed(5511)
    untrained = WorldModel([NeuralDynamics() for _ in range(3)])
    results = {}
    for name, world in [('trained', trained), ('untrained', untrained)]:
        prediction = np.stack([world.numpy_predict(x, p) for p in world.parameters]).mean(0)
        error = np.linalg.norm(prediction-y, axis=-1)*.1*400
        results[name] = {'transitions': len(x), 'mean_next_position_error_pixels': float(error.mean()),
                         'p95_next_position_error_pixels': float(np.quantile(error, .95))}
    result = {'coordinate_scale': '400 pixels per world unit; 800x800 RGB image spans two world units',
              'data': 'episode-disjoint held-out image transitions; no fitting or selection in this check', 'results': results}
    (folder/'prediction_ablation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
