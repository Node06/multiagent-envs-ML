"""Train: python wsl-support/vision_ppo/train.py --seed 11 --output ..."""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from vision_ppo.env_wrapper import VisualEnv
from vision_ppo.model import ActorCritic
from vision_ppo.buffer import gae
from vision_ppo.ppo import update
from vision_ppo.cases import training_case, generate_cases


def validate(model, env, cases):
    counts, steps = {}, []
    for case in cases:
        obs = env.reset(case)
        for _ in range(env.cfg['max_steps']):
            with torch.no_grad():
                action = model.deterministic(torch.from_numpy(obs)).numpy()
            obs, _, terminal, truncated, info = env.step(action)
            if terminal or truncated:
                reason = info['reason']
                counts[reason] = counts.get(reason, 0)+1
                if reason == 'success':
                    steps.append(env.steps)
                break
    return {'success_rate': counts.get('success', 0)/len(cases), 'outcomes': counts,
            'mean_success_steps': float(np.mean(steps)) if steps else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=HERE/'config.json')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--steps', type=int)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.steps:
        cfg['total_steps'] = args.steps
    torch.set_num_threads(cfg['torch_threads'])
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ActorCritic(cfg['hidden'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=1e-5)
    total, best_rate, best_steps = 0, -1., float('inf')
    if args.resume:
        cp = torch.load(args.resume, map_location='cpu', weights_only=False)
        if cp['seed'] != args.seed:
            raise ValueError('Resume seed must match checkpoint')
        for key in cfg:
            if key != 'total_steps' and cfg[key] != cp['config'][key]:
                raise ValueError(f'Resume configuration differs: {key}')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        total = cp['steps']; best_rate = cp['best_rate']; best_steps = cp['best_steps']
        rng.bit_generator.state = cp['rng']
        np.random.set_state(cp['numpy_rng']); random.setstate(cp['python_rng'])
        torch.set_rng_state(cp['torch_rng'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps({**cfg, 'seed': args.seed}, indent=2))
    validation_cases = generate_cases(cfg['validation_count'], cfg['validation_seed'])
    (args.output/'validation_cases.json').write_text(json.dumps(validation_cases, indent=2))
    env, val_env = VisualEnv(cfg), VisualEnv(cfg)
    started = time.perf_counter()
    episodes = 0; recent = []; last_val = total//cfg['validation_interval']
    def save(name):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'steps': total, 'seed': args.seed, 'config': cfg,
                    'best_rate': best_rate, 'best_steps': best_steps,
                    'rng': rng.bit_generator.state, 'numpy_rng': np.random.get_state(),
                    'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state()}, args.output/name)
    try:
        obs = env.reset(training_case(rng, total/cfg['total_steps'], cfg['curriculum_fraction']))
        while total < cfg['total_steps']:
            n = min(cfg['rollout_steps'], cfg['total_steps']-total)
            observations = []; latents = []; logps = []; rewards = []
            values = []; next_values = []; terminals = []; ends = []
            for _ in range(n):
                with torch.no_grad():
                    action, latent, logp, value = model.sample(torch.from_numpy(obs))
                next_obs, r, terminal, truncated, info = env.step(action.numpy())
                with torch.no_grad():
                    next_value = model.value(torch.from_numpy(next_obs)).item()
                observations.append(obs); latents.append(latent.numpy()); logps.append(logp.item())
                rewards.append(r); values.append(value.item()); next_values.append(next_value)
                terminals.append(terminal); ends.append(terminal or truncated)
                total += 1
                obs = next_obs
                if terminal or truncated:
                    episodes += 1
                    recent.append(info['reason']); recent = recent[-100:]
                    obs = env.reset(training_case(rng, total/cfg['total_steps'], cfg['curriculum_fraction']))
            advantage, returns = gae(np.array(rewards), np.array(values), np.array(next_values),
                                     np.array(terminals), np.array(ends), cfg['gamma'], cfg['gae_lambda'])
            stats = update(model, optimizer,
                           [np.array(observations), np.array(latents), np.array(logps), advantage, returns], cfg)
            row = {'steps': total, 'episodes': episodes, 'train_recent_success': recent.count('success')/max(1, len(recent)),
                   'seconds': round(time.perf_counter()-started, 2), **stats}
            with (args.output/'train.csv').open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            if total//cfg['validation_interval'] > last_val or total == cfg['total_steps']:
                last_val = total//cfg['validation_interval']
                result = validate(model, val_env, validation_cases)
                result.update(steps=total, seed=args.seed)
                with (args.output/'validation.jsonl').open('a') as f:
                    f.write(json.dumps(result)+'\n')
                score_steps = result['mean_success_steps'] or float('inf')
                if (result['success_rate'], -score_steps) > (best_rate, -best_steps):
                    best_rate, best_steps = result['success_rate'], score_steps
                    save('best.pt')
                print('VALIDATION '+json.dumps(result), flush=True)
                save('latest.pt')
            print('TRAIN '+json.dumps(row), flush=True)
        save('final.pt')
        print('COMPLETE '+json.dumps({'steps': total, 'best_validation_rate': best_rate,
                                     'seconds': time.perf_counter()-started}), flush=True)
    except KeyboardInterrupt:
        # Do not claim this partially collected rollout is a resumable checkpoint.
        print('Interrupted; resume from the last completed validation checkpoint.', flush=True)
        raise
    finally:
        env.close(); val_env.close()


if __name__ == '__main__':
    main()
