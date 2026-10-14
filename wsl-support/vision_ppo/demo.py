"""Deterministic frozen-policy demonstration; optional image window and MP4."""
import argparse
import json
from pathlib import Path
import sys
import torch
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from vision_ppo.evaluate import episode
from vision_ppo.env_wrapper import VisualEnv
from vision_ppo.model import ActorCritic
from vision_ppo.checkpoint import resolve_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--hunter', nargs=2, type=float, default=[0., 0.])
    p.add_argument('--evader', nargs=2, type=float, default=[.7, .7])
    p.add_argument('--goal', nargs=2, type=float, default=[-.5, -.5])
    p.add_argument('--show', action='store_true')
    p.add_argument('--output', type=Path, default=HERE/'outputs/demo')
    args = p.parse_args()
    if args.checkpoint is None:
        selection = json.loads((HERE/'outputs/formal_20261013/selected_model.json').read_text())
        args.checkpoint = resolve_checkpoint(selection, HERE/'outputs/formal_20261013')
    torch.set_num_threads(1)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = ActorCritic(cp['config']['hidden']); model.load_state_dict(cp['model']); model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    case = {'id': 'demo', 'hunter': args.hunter, 'evader': args.evader, 'goal': args.goal}
    (args.output/'case.json').write_text(json.dumps(case, indent=2))
    env = VisualEnv(cp['config'])
    try:
        result = episode(case, cp['config'], model, f'ppo_seed{cp["seed"]}', {}, env,
                         args.output, record=True, show=args.show)
        (args.output/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
