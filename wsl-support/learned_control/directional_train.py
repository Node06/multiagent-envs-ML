"""Train a mode-preserving acceleration classifier, DAgger, categorical PPO."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from learned_control.directional_model import DirectionalActorCritic
from learned_control.optimize import validate, collect, difficult_cases, safety_potential
from learned_control.perception import PixelNet
from learned_control.runtime import LearnedController
from course_audit.audit import original_files, create_case, random_obstacle_cases
from vision_mpc.env_adapter import render_rgb, close
from vision_mpc.geometry import action5
from vision_mpc.perception import DetectionError
from vision_mpc.evaluator import generate_cases, terminal_reason
from vision_ppo.buffer import gae


def fit(actor, x, y, epochs, rng):
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    labels = actor.labels(ty)
    opt = torch.optim.Adam(actor.actor.parameters(), lr=.0003)
    for epoch in range(epochs):
        for idx in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x)/256))):
            loss = F.cross_entropy(actor.actor(tx[idx]), labels[idx], label_smoothing=.02)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), 1); opt.step()
        if epoch % 10 == 0: print('CLASSIFIER '+json.dumps({'epoch': epoch+1, 'loss': loss.item(), 'samples': len(x)}), flush=True)


def ppo_update(actor, optimizer, data, tx, labels, cfg, rng):
    ob, index, oldlp, advantages, returns = data
    ob = torch.as_tensor(ob, dtype=torch.float32); index = torch.as_tensor(index, dtype=torch.long)
    oldlp, adv, returns = [torch.as_tensor(a, dtype=torch.float32) for a in (oldlp, advantages, returns)]
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    stats = []; stop = False
    for _ in range(cfg['epochs']):
        for idx in np.array_split(rng.permutation(len(ob)), int(np.ceil(len(ob)/256))):
            dist = actor.distribution(ob[idx]); logratio = dist.log_prob(index[idx])-oldlp[idx]
            ratio = logratio.exp(); kl = ((ratio-1)-logratio).mean()
            if kl.item() > cfg['target_kl']: stop = True; break
            policy = -torch.minimum(ratio*adv[idx], ratio.clamp(.9, 1.1)*adv[idx]).mean()
            value = .5*(actor.value(ob[idx])-returns[idx]).square().mean()
            ti = rng.integers(0, len(tx), len(idx))
            bc = F.cross_entropy(actor.actor(tx[ti]), labels[ti])
            loss = policy+.5*value+cfg['bc_coef']*bc-cfg['entropy_coef']*dist.entropy().mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), .5); optimizer.step()
            stats.append([policy.item(), value.item(), bc.item(), kl.item()])
        if stop: break
    return dict(zip(('policy_loss', 'value_loss', 'bc_loss', 'kl'), np.mean(stats, axis=0).tolist() if stats else [0]*4))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'outputs/directional_20261014')
    p.add_argument('--seed', type=int, default=4411)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--dagger-cases', type=int, default=80)
    p.add_argument('--ppo-steps', type=int, default=8192)
    args = p.parse_args(); torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'training_summary.json').exists(): raise RuntimeError('Choose a new output directory')
    torch.manual_seed(args.seed); np.random.seed(args.seed); rng = np.random.default_rng(args.seed)
    started = time.perf_counter(); before = original_files()
    cp = torch.load(HERE/'outputs/refined_20261014/primary_ppo.pt', map_location='cpu', weights_only=False)
    pixel = PixelNet(); pixel.load_state_dict(cp['perception']); pixel.eval()
    actor = DirectionalActorCritic(256); actor.initialize_backbone(cp['model'])
    data = np.load(HERE/'outputs/refined_20261014/demonstrations.npz'); x, y = data['features'], data['actions']
    cfg = json.loads((HERE.parent/'vision_mpc/config.json').read_text())
    rl = dict(cp['config']); rl.update(bc_coef=.1, entropy_coef=.0001, learning_rate=1e-5, critic_learning_rate=.0003)
    validation = generate_cases(80, 2026101440)  # existing validation, never a published test set
    (args.output/'validation_cases.json').write_text(json.dumps(validation, indent=2))
    (args.output/'configuration.json').write_text(json.dumps({'seed': args.seed, 'epochs': args.epochs,
                                                             'dagger_cases': args.dagger_cases, 'ppo_steps': args.ppo_steps,
                                                             'directions': 32, 'magnitudes': [0, .25, .5, .75, 1], 'temperature': .4,
                                                             'validation_seed': 2026101440, 'rl': rl}, indent=2))
    stages = []; best = (-1., -float('inf')); best_ppo = best
    def assess(stage):
        nonlocal best, best_ppo
        result = validate(pixel, actor, validation, cfg); result['stage'] = stage; stages.append(result)
        (args.output/'validation.json').write_text(json.dumps(stages, indent=2))
        saved = {'perception': pixel.state_dict(), 'model': copy.deepcopy(actor.state_dict()), 'hidden': 256,
                 'config': rl, 'seed': args.seed, 'stage': stage, 'validation': result,
                 'feature_version': 'safety56', 'policy_type': 'categorical129', 'temperature': .4,
                 'input': 'RGB image-derived geometry and own action history; no simulator state'}
        torch.save(saved, args.output/(stage+'.pt'))
        score = (result['success_rate'], -(result['mean_success_steps'] or float('inf')))
        if score > best: best = score; torch.save(saved, args.output/'best.pt')
        if stage.startswith('ppo') and score > best_ppo: best_ppo = score; torch.save(saved, args.output/'primary_ppo.pt')
        print('VALIDATION '+json.dumps(result), flush=True)
    for i in range(0, args.epochs, 20):
        fit(actor, x, y, min(20, args.epochs-i), rng); assess(f'classifier{min(i+20, args.epochs)}')
    actor.load_state_dict(torch.load(args.output/'best.pt', weights_only=False)['model'])
    cases = difficult_cases(args.dagger_cases, args.seed+10)
    (args.output/'dagger_cases.json').write_text(json.dumps(cases, indent=2))
    ex, ey, outcomes = collect(pixel, actor, cases, cfg, .2, rng)
    x = np.concatenate((x, ex)); y = np.concatenate((y, ey))
    np.savez_compressed(args.output/'demonstrations.npz', features=x, actions=y)
    fit(actor, x, y, 60, rng); assess('dagger')
    best_cp = torch.load(args.output/'best.pt', weights_only=False); actor.load_state_dict(best_cp['model'])
    torch.save(best_cp, args.output/'imitation_best.pt')
    tx = torch.from_numpy(x); labels = actor.labels(torch.from_numpy(y))
    opt = torch.optim.Adam([{'params': actor.actor.parameters(), 'lr': 1e-5},
                            {'params': actor.critic.parameters(), 'lr': .0003}], eps=1e-5)
    value_opt = torch.optim.Adam(actor.critic.parameters(), lr=.0003)
    model = LearnedController(pixel, actor, feature_version='safety56'); total = 0; ep = 0
    def reset():
        nonlocal ep
        case = difficult_cases(2, args.seed+10000+ep)[ep % 2]; ep += 1
        env = create_case(case); model.reset(); return env, model.observe(render_rgb(env))
    env, feature = reset(); episode_steps = 0
    try:
        while total < args.ppo_steps:
            lists = [[] for _ in range(8)]
            for _ in range(min(1024, args.ppo_steps-total)):
                with torch.no_grad(): command, index, lp, value = actor.sample(torch.from_numpy(feature))
                phi = safety_potential(model.state)
                _, _, done, _ = env.step(action5(command.numpy())); model.pending = command.numpy().copy()
                reason = terminal_reason(env, done); episode_steps += 1
                try: next_feature = model.observe(render_rgb(env))
                except DetectionError: next_feature = feature.copy(); reason = reason or 'perception_failure'
                terminal = reason is not None; timeout = not terminal and episode_steps >= 500
                r = (20 if reason == 'success' else -20 if terminal else 0)-.01+rl['gamma']*(0 if terminal else safety_potential(model.state))-phi
                with torch.no_grad(): nv = actor.value(torch.from_numpy(next_feature)).item()
                for row, val in zip(lists, [feature.copy(), index.item(), lp.item(), r, value.item(), nv, terminal, terminal or timeout]): row.append(val)
                feature = next_feature; total += 1
                if terminal or timeout: close(env); env, feature = reset(); episode_steps = 0
            ob, index, lp, rewards, values, nv, terminated, ended = map(np.array, lists)
            adv, returns = gae(rewards, values, nv, terminated, ended, rl['gamma'], rl['gae_lambda'])
            ox, rt = torch.from_numpy(ob), torch.as_tensor(returns, dtype=torch.float32)
            for _ in range(3):
                for idx in np.array_split(rng.permutation(len(ob)), 4):
                    loss = (actor.value(ox[idx])-rt[idx]).square().mean()
                    value_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.critic.parameters(), 1); value_opt.step()
            stats = ppo_update(actor, opt, [ob, index, lp, adv, returns], tx, labels, rl, rng)
            print('PPO '+json.dumps({'steps': total, **stats}), flush=True)
            if total % 4096 == 0 or total == args.ppo_steps: assess(f'ppo{total}')
    finally: close(env)
    assert original_files() == before
    selected = torch.load(args.output/'primary_ppo.pt', weights_only=False)
    (args.output/'primary_selection.json').write_text(json.dumps({'criterion': 'validation only, PPO stages only', **selected['validation']}, indent=2))
    (args.output/'training_summary.json').write_text(json.dumps({'seed': args.seed, 'samples': len(x), 'ppo_steps': total,
                                                               'dagger_outcomes': outcomes, 'best_validation': best[0], 'ppo_validation': best_ppo[0],
                                                               'seconds': time.perf_counter()-started, 'original_integrity': 'passed'}, indent=2))
    print('DIRECTIONAL TRAINING COMPLETE', flush=True)


if __name__ == '__main__': main()
