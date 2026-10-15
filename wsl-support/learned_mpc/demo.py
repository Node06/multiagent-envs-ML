import argparse
import json
from pathlib import Path
import sys
import torch
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from learned_mpc.runtime import load_controller
from learned_control.evaluate import episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=HERE/'outputs/main_20261014/world_empirical.pt')
    p.add_argument('--output', type=Path, default=HERE/'outputs/main_20261014/demo')
    p.add_argument('--show', action='store_true')
    args = p.parse_args(); torch.set_num_threads(1)
    selection = args.checkpoint.parent/'validation/selection.json'
    config = {'uncertainty_scale': json.loads(selection.read_text())['uncertainty_scale']} if selection.exists() else {'uncertainty_scale': .25}
    model, _ = load_controller(args.checkpoint, config)
    args.output.mkdir(parents=True, exist_ok=True)
    case = dict(id='default_demo', hunter=[0, 0], evader=[.7, .7], goal=[-.5, -.5], group='default_demo')
    result = episode(case, 'learned_mpc', model, None, model.cfg, args.output, video=True, show=args.show)
    result['learned_prediction_calls'] = model.world.prediction_calls
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
