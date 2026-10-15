"""Versioned safety-feature distillation, targeted DAgger and regularized PPO."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE.parent), str(ROOT)]
from course_audit.audit import create_case, original_files
from vision_mpc.env_adapter import render_rgb, close
from vision_mpc.perception import detect, DetectionError
from vision_mpc.state_estimator import Estimator
from vision_mpc.global_planner import Planner
from vision_mpc.controller_v2 import Controller
from vision_mpc.geometry import action5, limit
from vision_mpc.evaluator import generate_cases, terminal_reason
from vision_ppo.buffer import gae
from learned_control.perception import PixelNet
from learned_control.runtime import LearnedController
from learned_control.safety_model import SafetyActorCritic
from learned_control.safety_features import augment


def difficult_cases(count, seed):
    """Fresh legal fixed-obstacle starts; deliberately sample blocked goal rays."""
    rng = np.random.default_rng(seed)
    obstacles = np.array([[-.35, .35], [.35, .35], [0, -.35]])
    cases = []
    while len(cases) < count:
        if len(cases) % 2:
            case = generate_cases(1, int(rng.integers(1, 2**30)))[0]
        else:
            center = obstacles[int(rng.integers(3))]
            angle = rng.uniform(-np.pi, np.pi)
            axis = np.array([np.cos(angle), np.sin(angle)])
            prey = center+axis*rng.uniform(.135, .42)
            goal = center-axis*rng.uniform(.135, .42)+rng.normal(0, .035, 2)
            hunter = rng.uniform(-.72, .72, 2)
            points = np.array([hunter, prey, goal])
            if np.max(abs(points)) > .72 or np.min(np.linalg.norm(points[:, None]-obstacles[None], axis=-1)) < .13:
                continue
            if min(np.linalg.norm(points[i]-points[j]) for i in range(3) for j in range(i)) < .25:
                continue
            case = dict(hunter=hunter.tolist(), evader=prey.tolist(), goal=goal.tolist())
        case['id'] = f'hard_{seed}_{len(cases):03}'
        cases.append(case)
    return cases


def weights(features):
    obs = features[:, 21:48].reshape(-1, 3, 9)
    clearance = obs[:, :, 0].min(axis=1)*.25
    closing = obs[:, :, 3].max(axis=1)*.25
    hunter = features[:, 48]*.5
    return (1+2*(clearance < .10)+1*(closing > .12)+1*(hunter < .22)).astype(np.float32)


def fit(actor, features, targets, epochs, rng, lr=.0002):
    x, y, w = torch.from_numpy(features), torch.from_numpy(targets), torch.from_numpy(weights(features))
    opt = torch.optim.Adam(actor.actor.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        for index in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x)/256))):
            action = actor.deterministic(x[index])
            error = (action-y[index]).square().mean(-1)
            loss = (error*w[index]).sum()/w[index].sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), 1); opt.step()
        if epoch % 20 == 0:
            print(json.dumps({'bc_epoch': epoch+1, 'loss': loss.item(), 'samples': len(x)}), flush=True)
        losses.append(loss.item())
    return float(np.mean(losses))


def validate(pixel, actor, cases, cfg):
    model = LearnedController(pixel, actor, feature_version='safety56')
    counts, lengths = {}, []
    for case in cases:
        env = create_case(case); model.reset(); reason = 'timeout'
        try:
            for step in range(cfg['max_steps']):
                try:
                    command = model.act(render_rgb(env))
                except DetectionError:
                    reason = 'perception_failure'; break
                _, _, done, _ = env.step(action5(command))
                reason = terminal_reason(env, done)
                if reason:
                    break
            reason = reason or 'timeout'
            counts[reason] = counts.get(reason, 0)+1
            if reason == 'success': lengths.append(step+1)
        finally:
            close(env)
    return {'success_rate': counts.get('success', 0)/len(cases), 'outcomes': counts,
            'mean_success_steps': float(np.mean(lengths)) if lengths else None}


def collect(pixel, actor, cases, cfg, probability, rng):
    model = LearnedController(pixel, actor, feature_version='safety56')
    records, targets, outcomes = [], [], {}
    for case in cases:
        env = create_case(case); model.reset(); est = Estimator(.1); expert = None; reason = 'timeout'
        try:
            for step in range(cfg['max_steps']):
                rgb = render_rgb(env)
                try:
                    features = model.observe(rgb); state = est.update(detect(rgb))
                except DetectionError:
                    reason = 'perception_failure'; break
                if expert is None: expert = Controller(cfg, Planner(state.detection, cfg))
                target, _ = expert.choose(state); target = limit(target, 1)
                records.append(features.copy()); targets.append(target.astype(np.float32))
                with torch.no_grad():
                    command = target if rng.random() < probability else limit(actor.deterministic(torch.from_numpy(features)).numpy(), 1)
                _, _, done, _ = env.step(action5(command)); model.pending = command.copy(); expert.pending = command.copy()
                end = terminal_reason(env, done)
                if end: reason = end; break
        finally:
            close(env)
        outcomes[reason] = outcomes.get(reason, 0)+1
        print('DAGGER '+json.dumps({'case': case['id'], 'reason': reason, 'samples': len(records)}), flush=True)
    return np.array(records, np.float32), np.array(targets, np.float32), outcomes


def safety_potential(state):
    d = state.detection
    clearance = np.linalg.norm(d.obstacles-d.evader, axis=1)-.075
    hunter = np.linalg.norm(d.hunter-d.evader)-.05
    # Potential shaping, not altered environment success/collision thresholds.
    return (-5*np.linalg.norm(d.goal-d.evader)
            -.12*np.logaddexp(0, (.10-clearance)/.03).sum()
            -.12*np.logaddexp(0, (.20-hunter)/.05))


def regularized_update(actor, optimizer, data, cfg, teacher_x, teacher_y, rng):
    obs, latent, oldlp, adv, returns = [torch.as_tensor(a, dtype=torch.float32) for a in data]
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    stats = []; stopped = False
    for epoch in range(cfg['epochs']):
        for idx in np.array_split(rng.permutation(len(obs)), int(np.ceil(len(obs)/cfg['batch_size']))):
            dist = actor.distribution(obs[idx]); lp = actor.log_prob(dist, latent[idx]); logratio = lp-oldlp[idx]
            ratio = logratio.exp(); kl = ((ratio-1)-logratio).mean()
            if kl.item() > cfg['target_kl']: stopped = True; break
            policy = -torch.minimum(ratio*adv[idx], ratio.clamp(1-cfg['clip'], 1+cfg['clip'])*adv[idx]).mean()
            value = .5*(actor.value(obs[idx])-returns[idx]).square().mean()
            ti = rng.integers(0, len(teacher_x), len(idx))
            imitation = (actor.deterministic(teacher_x[ti])-teacher_y[ti]).square().mean()
            entropy = -actor.log_prob(dist, dist.rsample()).mean()
            loss = policy+cfg['value_coef']*value+cfg['bc_coef']*imitation-cfg['entropy_coef']*entropy
            if cfg.get('safety_coef', 0):
                from learned_control.safety_loss import risk
                loss = loss+cfg['safety_coef']*risk(obs[idx], actor.deterministic(obs[idx]))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), .5); optimizer.step()
            stats.append([policy.item(), value.item(), imitation.item(), kl.item()])
        if stopped: break
    return dict(zip(('policy_loss', 'value_loss', 'bc_loss', 'kl'), np.mean(stats, axis=0).tolist() if stats else [0]*4))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'outputs/optimized_20261014')
    p.add_argument('--seeds', nargs='+', type=int, default=[2411, 2423, 2437])
    p.add_argument('--bc-epochs', type=int, default=160)
    p.add_argument('--dagger-cases', type=int, default=60)
    p.add_argument('--dagger-rounds', type=int, default=2)
    p.add_argument('--ppo-steps', type=int, default=8192)
    p.add_argument('--validation-count', type=int, default=80)
    args = p.parse_args(); torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'training_summary.json').exists():
        raise RuntimeError('Completed experiment exists; choose a new --output directory')
    protected = original_files(); start = time.perf_counter()
    cfg = json.loads((HERE.parent/'vision_mpc/config.json').read_text())
    rl = json.loads((HERE.parent/'vision_ppo/config.json').read_text())
    rl.update(learning_rate=1e-5, entropy_coef=.0001, epochs=4, rollout_steps=1024,
              clip=.1, target_kl=.01, hidden=256, bc_coef=1.)
    validation = generate_cases(args.validation_count, 2026101416)
    (args.output/'validation_cases.json').write_text(json.dumps(validation, indent=2))
    initial = torch.load(HERE/'outputs/main_20261014/imitation_best.pt', map_location='cpu', weights_only=False)
    pixel = PixelNet(); pixel.load_state_dict(initial['perception']); pixel.eval()
    data = np.load(HERE/'outputs/main_20261014/demonstrations.npz')
    base_x = augment(data['features']); base_y = data['actions']
    configuration = {**vars(args), 'output': str(args.output), 'rl': rl, 'feature_version': 'safety56',
                     'validation_seed': 2026101416, 'training_source': 'main_20261014/demonstrations.npz; no published test scenes'}
    (args.output/'configuration.json').write_text(json.dumps(configuration, indent=2))
    all_results = []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        folder = args.output/f'seed{seed}'; folder.mkdir(exist_ok=True)
        actor = SafetyActorCritic(256); actor.initialize_from(initial['model'])
        stages = []; best = (-1, -float('inf')); best_ppo = (-1, -float('inf'))
        def assess(stage):
            nonlocal best, best_ppo
            result = validate(pixel, actor, validation, cfg); result['stage'] = stage
            stages.append(result); (folder/'validation.json').write_text(json.dumps(stages, indent=2))
            checkpoint = {'perception': pixel.state_dict(), 'model': copy.deepcopy(actor.state_dict()), 'hidden': 256,
                          'config': rl, 'seed': seed, 'stage': stage, 'validation': result, 'feature_version': 'safety56',
                          'input': 'RGB image-derived safety geometry and own action history; no simulator state'}
            torch.save(checkpoint, folder/(stage+'.pt'))
            score = (result['success_rate'], -(result['mean_success_steps'] or float('inf')))
            if score > best: best = score; torch.save(checkpoint, folder/'best.pt')
            if stage.startswith('ppo') and score > best_ppo: best_ppo = score; torch.save(checkpoint, folder/'primary_ppo.pt')
            print('VALIDATION '+json.dumps({'seed': seed, **result}), flush=True)
        x, y = base_x.copy(), base_y.copy()
        # Initial feature extension preserves the old actor exactly before fitting.
        for chunk in range(0, args.bc_epochs, 40):
            fit(actor, x, y, min(40, args.bc_epochs-chunk), rng)
            assess(f'bc{min(chunk+40, args.bc_epochs)}')
        for round_id in range(args.dagger_rounds):
            selected = torch.load(folder/'best.pt', weights_only=False); actor.load_state_dict(selected['model'])
            cases = difficult_cases(args.dagger_cases, seed+100+round_id)
            (folder/f'dagger{round_id+1}_cases.json').write_text(json.dumps(cases, indent=2))
            extra_x, extra_y, outcomes = collect(pixel, actor, cases, cfg, .3/(round_id+1), rng)
            x = np.concatenate((x, extra_x)); y = np.concatenate((y, extra_y))
            np.savez_compressed(folder/'demonstrations.npz', features=x, actions=y)
            fit(actor, x, y, 80, rng, lr=.0001); assess(f'dagger{round_id+1}')
        imitation = torch.load(folder/'best.pt', weights_only=False); actor.load_state_dict(imitation['model'])
        torch.save(imitation, folder/'imitation_best.pt')
        tx, ty = torch.from_numpy(x), torch.from_numpy(y)
        optimizer = torch.optim.Adam(actor.parameters(), lr=rl['learning_rate'], eps=1e-5)
        controller = LearnedController(pixel, actor, feature_version='safety56')
        episode_id = 0; total = 0
        def reset():
            nonlocal episode_id
            case = difficult_cases(2, seed+10000+episode_id)[episode_id % 2]; episode_id += 1
            env = create_case(case); controller.reset(); return env, controller.observe(render_rgb(env))
        env, feature = reset(); episode_steps = 0
        try:
            while total < args.ppo_steps:
                lists = [[] for _ in range(8)]
                for _ in range(min(rl['rollout_steps'], args.ppo_steps-total)):
                    with torch.no_grad(): command, latent, lp, value = actor.sample(torch.from_numpy(feature))
                    phi = safety_potential(controller.state)
                    _, _, done, _ = env.step(action5(command.numpy())); controller.pending = limit(command.numpy(), 1)
                    reason = terminal_reason(env, done); episode_steps += 1
                    try: next_feature = controller.observe(render_rgb(env))
                    except DetectionError: next_feature = feature.copy(); reason = reason or 'perception_failure'
                    terminal = reason is not None; timeout = not terminal and episode_steps >= rl['max_steps']
                    shaped = rl['gamma']*(0 if terminal else safety_potential(controller.state))-phi
                    r = (rl['success_reward'] if reason == 'success' else rl['failure_reward'] if terminal else 0)-rl['step_cost']+shaped
                    with torch.no_grad(): nv = actor.value(torch.from_numpy(next_feature)).item()
                    values = [feature.copy(), latent.numpy(), lp.item(), r, value.item(), nv, terminal, terminal or timeout]
                    for row, val in zip(lists, values): row.append(val)
                    feature = next_feature; total += 1
                    if terminal or timeout: close(env); env, feature = reset(); episode_steps = 0
                ob, z, lp, r, v, nv, terminated, ended = map(np.array, lists)
                advantages, returns = gae(r, v, nv, terminated, ended, rl['gamma'], rl['gae_lambda'])
                stats = regularized_update(actor, optimizer, [ob, z, lp, advantages, returns], rl, tx, ty, rng)
                print('PPO '+json.dumps({'seed': seed, 'steps': total, **stats}), flush=True)
                if total % 4096 == 0 or total == args.ppo_steps: assess(f'ppo{total}')
        finally: close(env)
        all_results.append({'seed': seed, 'best': best[0], 'best_ppo': best_ppo[0], 'samples': len(x), 'ppo_steps': total})
        (args.output/'progress.json').write_text(json.dumps(all_results, indent=2))
    candidates = [(r, torch.load(args.output/f"seed{r['seed']}"/'primary_ppo.pt', weights_only=False)) for r in all_results]
    selected, checkpoint = max(candidates, key=lambda pair: (pair[1]['validation']['success_rate'], -(pair[1]['validation']['mean_success_steps'] or float('inf'))))
    torch.save(checkpoint, args.output/'primary_ppo.pt')
    (args.output/'primary_selection.json').write_text(json.dumps({'criterion': 'validation only across PPO stages/seeds; no test feedback', **selected, 'stage': checkpoint['stage']}, indent=2))
    assert original_files() == protected
    (args.output/'training_summary.json').write_text(json.dumps({'seeds': all_results, 'selected': selected,
                                                               'seconds': time.perf_counter()-start, 'original_integrity': 'passed'}, indent=2))
    print('OPTIMIZATION TRAINING COMPLETE', flush=True)


if __name__ == '__main__': main()
