"""Refine the frozen safety56 actor with training risk and faster value fitting."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from learned_control.optimize import difficult_cases, collect, validate, safety_potential, regularized_update
from learned_control.safety_loss import risk
from learned_control.safety_model import SafetyActorCritic
from learned_control.runtime import LearnedController
from learned_control.perception import PixelNet
from course_audit.audit import original_files, create_case, random_obstacle_cases
from vision_mpc.geometry import action5, limit
from vision_mpc.evaluator import generate_cases, terminal_reason
from vision_mpc.env_adapter import render_rgb, close
from vision_mpc.perception import DetectionError
from vision_ppo.buffer import gae


def fit_risk(actor, x, y, coefficient, rng, epochs):
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    opt = torch.optim.Adam(actor.actor.parameters(), lr=.0001)
    for epoch in range(epochs):
        for idx in np.array_split(rng.permutation(len(tx)), int(np.ceil(len(tx)/256))):
            output = actor.deterministic(tx[idx])
            imitation = (output-ty[idx]).square().mean()
            safety = risk(tx[idx], output)
            loss = imitation+coefficient*safety
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), 1); opt.step()
        if epoch % 10 == 0:
            print('RISK FIT '+json.dumps({'coefficient': coefficient, 'epoch': epoch+1, 'imitation': imitation.item(), 'risk': safety.item()}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'outputs/refined_20261014')
    p.add_argument('--source', type=Path, default=HERE/'outputs/optimized_20261014')
    p.add_argument('--seed', type=int, default=3411)
    p.add_argument('--ppo-steps', type=int, default=8192)
    p.add_argument('--epochs', type=int, default=40)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'training_summary.json').exists(): raise RuntimeError('Use a fresh --output')
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed); before = original_files(); started = time.perf_counter()
    cp = torch.load(args.source/'primary_ppo.pt', map_location='cpu', weights_only=False)
    pixel = PixelNet(); pixel.load_state_dict(cp['perception']); pixel.eval()
    actor = SafetyActorCritic(cp['hidden']); actor.load_state_dict(cp['model'])
    data = np.load(args.source/f"seed{cp['seed']}"/'demonstrations.npz')
    cfg = json.loads((HERE.parent/'vision_mpc/config.json').read_text())
    cases = difficult_cases(60, args.seed+10)+random_obstacle_cases(20, args.seed+11)
    (args.output/'extra_training_cases.json').write_text(json.dumps(cases, indent=2))
    ex, ey, outcomes = collect(pixel, actor, cases, cfg, 1., rng)
    x = np.concatenate((data['features'], ex)); y = np.concatenate((data['actions'], ey))
    np.savez_compressed(args.output/'demonstrations.npz', features=x, actions=y)
    validation = generate_cases(80, 2026101440)
    (args.output/'validation_cases.json').write_text(json.dumps(validation, indent=2))
    rl = dict(cp['config']); rl.update(learning_rate=1e-5, critic_learning_rate=.0003, critic_warm_epochs=5, safety_coef=.1)
    stages = []; best = (-1., -float('inf')); best_ppo = (-1., -float('inf'))
    def assess(stage):
        nonlocal best, best_ppo
        result = validate(pixel, actor, validation, cfg); result['stage'] = stage
        stages.append(result); (args.output/'validation.json').write_text(json.dumps(stages, indent=2))
        saved = dict(cp, model=copy.deepcopy(actor.state_dict()), seed=args.seed, config=dict(rl), stage=stage, validation=result)
        torch.save(saved, args.output/(stage+'.pt'))
        score = (result['success_rate'], -(result['mean_success_steps'] or float('inf')))
        if score > best: best = score; torch.save(saved, args.output/'best.pt')
        if stage.startswith('ppo') and score > best_ppo: best_ppo = score; torch.save(saved, args.output/'primary_ppo.pt')
        print('VALIDATION '+json.dumps(result), flush=True)
    assess('initial')
    for coefficient in (.025, .1, .4):
        actor.load_state_dict(cp['model']); rl['safety_coef'] = coefficient
        fit_risk(actor, x, y, coefficient, np.random.default_rng(args.seed), args.epochs)
        assess('safety_'+str(coefficient).replace('.', '_'))
    selected = torch.load(args.output/'best.pt', weights_only=False)
    actor.load_state_dict(selected['model']); rl['safety_coef'] = selected['config']['safety_coef']
    torch.save(selected, args.output/'imitation_best.pt')
    (args.output/'configuration.json').write_text(json.dumps({'source': str(args.source), 'source_stage': cp['stage'],
                                                             'seed': args.seed, 'coefficients': [.025, .1, .4], 'epochs': args.epochs,
                                                             'rl': rl, 'ppo_steps': args.ppo_steps, 'validation_seed': 2026101440}, indent=2))
    optimizer = torch.optim.Adam([{'params': actor.actor.parameters(), 'lr': rl['learning_rate']},
                                  {'params': actor.critic.parameters(), 'lr': rl['critic_learning_rate']},
                                  {'params': [actor.log_std], 'lr': rl['learning_rate']}], eps=1e-5)
    critic_optimizer = torch.optim.Adam(actor.critic.parameters(), lr=rl['critic_learning_rate'])
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    controller = LearnedController(pixel, actor, feature_version='safety56')
    total = 0; episode_id = 0
    def reset():
        nonlocal episode_id
        seed = args.seed+10000+episode_id
        case = random_obstacle_cases(1, seed)[0] if episode_id % 4 == 0 else difficult_cases(2, seed)[episode_id % 2]
        episode_id += 1; env = create_case(case); controller.reset(); return env, controller.observe(render_rgb(env))
    env, feature = reset(); episode_steps = 0
    try:
        while total < args.ppo_steps:
            lists = [[] for _ in range(8)]
            for _ in range(min(rl['rollout_steps'], args.ppo_steps-total)):
                with torch.no_grad(): command, z, lp, value = actor.sample(torch.from_numpy(feature))
                phi = safety_potential(controller.state)
                _, _, done, _ = env.step(action5(command.numpy())); controller.pending = limit(command.numpy(), 1)
                reason = terminal_reason(env, done); episode_steps += 1
                try: next_feature = controller.observe(render_rgb(env))
                except DetectionError: next_feature = feature.copy(); reason = reason or 'perception_failure'
                terminal = reason is not None; timeout = not terminal and episode_steps >= rl['max_steps']
                r = (20 if reason == 'success' else -20 if terminal else 0)-.01+rl['gamma']*(0 if terminal else safety_potential(controller.state))-phi
                with torch.no_grad(): nv = actor.value(torch.from_numpy(next_feature)).item()
                for row, val in zip(lists, [feature.copy(), z.numpy(), lp.item(), r, value.item(), nv, terminal, terminal or timeout]): row.append(val)
                feature = next_feature; total += 1
                if terminal or timeout: close(env); env, feature = reset(); episode_steps = 0
            ob, z, lp, r, v, nv, terminated, ended = map(np.array, lists)
            advantages, returns = gae(r, v, nv, terminated, ended, rl['gamma'], rl['gae_lambda'])
            ox, target_returns = torch.from_numpy(ob), torch.from_numpy(returns).float()
            for _ in range(rl['critic_warm_epochs']):
                for idx in np.array_split(rng.permutation(len(ox)), int(np.ceil(len(ox)/256))):
                    loss = (actor.value(ox[idx])-target_returns[idx]).square().mean()
                    critic_optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.critic.parameters(), 1); critic_optimizer.step()
            stats = regularized_update(actor, optimizer, [ob, z, lp, advantages, returns], rl, tx, ty, rng)
            print('PPO '+json.dumps({'steps': total, **stats}), flush=True)
            if total % 4096 == 0 or total == args.ppo_steps: assess(f'ppo{total}')
    finally: close(env)
    assert before == original_files()
    final = torch.load(args.output/'primary_ppo.pt', weights_only=False)
    (args.output/'primary_selection.json').write_text(json.dumps({'criterion': '80-case new validation only, PPO stages only', 'stage': final['stage'], **final['validation']}, indent=2))
    (args.output/'training_summary.json').write_text(json.dumps({'seed': args.seed, 'samples': len(x), 'extra_expert_outcomes': outcomes,
                                                               'ppo_steps': total, 'validation_best': best[0], 'ppo_best': best_ppo[0],
                                                               'seconds': time.perf_counter()-started, 'original_integrity': 'passed'}, indent=2))
    print('REFINEMENT TRAINING COMPLETE', flush=True)


if __name__ == '__main__': main()
