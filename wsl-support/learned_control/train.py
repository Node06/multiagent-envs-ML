"""Train the image-facing policy: RGB CNN, imitation, DAgger and PPO refinement."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE.parent), str(ROOT)]
from course_audit.audit import create_case, original_files, random_obstacle_cases
from vision_mpc.env_adapter import render_rgb, close
from vision_mpc.perception import detect, DetectionError
from vision_mpc.state_estimator import Estimator
from vision_mpc.global_planner import Planner
from vision_mpc.controller_v2 import Controller
from vision_mpc.evaluator import generate_cases, terminal_reason
from vision_mpc.geometry import action5, limit
from vision_ppo.model import ActorCritic
from vision_ppo.buffer import gae
from vision_ppo.ppo import update
from vision_ppo.rewards import potential, reward
from learned_control.perception import fit_pixel_net
from learned_control.runtime import LearnedController


def collect(cases, cfg, pixel, actor, teacher_probability, rng):
    records, outcomes = [], {}
    model = LearnedController(pixel, actor)
    for case in cases:
        env = create_case(case); model.reset(); estimator = Estimator(.1)
        reason = 'timeout'
        try:
            teacher = None
            for step in range(cfg['max_steps']):
                rgb = render_rgb(env)
                try:
                    feature = model.observe(rgb)
                    state = estimator.update(detect(rgb))
                except DetectionError:
                    reason = 'perception_failure'; break
                if teacher is None:
                    teacher = Controller(cfg, Planner(state.detection, cfg))
                target, _ = teacher.choose(state)
                records.append((feature.copy(), limit(target, 1.).astype(np.float32)))
                if rng.random() < teacher_probability:
                    command = limit(target, 1.)
                else:
                    with torch.no_grad():
                        command = limit(actor.deterministic(torch.from_numpy(feature)).numpy(), 1.)
                _, _, done, _ = env.step(action5(command))
                model.pending = command.copy()
                teacher.pending = command.copy()  # executed queued control, not hypothetical expert action
                reason_now = terminal_reason(env, done)
                if reason_now:
                    reason = reason_now; break
        finally:
            close(env)
        outcomes[reason] = outcomes.get(reason, 0)+1
        print('COLLECT '+json.dumps({'case': case['id'], 'samples': len(records), 'reason': reason}), flush=True)
    return records, outcomes


def fit_actor(actor, records, epochs, rng):
    obs = torch.from_numpy(np.stack([r[0] for r in records])).float()
    targets = torch.from_numpy(np.stack([r[1] for r in records])).float()
    optimizer = torch.optim.Adam(actor.actor.parameters(), lr=.0006)
    for epoch in range(epochs):
        losses = []
        for indices in np.array_split(rng.permutation(len(obs)), max(1, int(np.ceil(len(obs)/256)))):
            predicted = actor.deterministic(obs[indices])
            loss = (predicted-targets[indices]).square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), 1.)
            optimizer.step(); losses.append(loss.item())
        if epoch % 10 == 0 or epoch == epochs-1:
            print('BC '+json.dumps({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'samples': len(obs)}), flush=True)
    return float(np.mean(losses))


def validate(pixel, actor, cases, cfg):
    model = LearnedController(pixel, actor)
    outcomes, lengths = {}, []
    for case in cases:
        env = create_case(case); model.reset(); reason = 'timeout'
        try:
            for step in range(cfg['max_steps']):
                try:
                    command = model.act(render_rgb(env))
                except DetectionError:
                    reason = 'perception_failure'; break
                _, _, done, _ = env.step(action5(command))
                end = terminal_reason(env, done)
                if end:
                    reason = end; break
            if reason == 'success':
                lengths.append(step+1)
        finally:
            close(env)
        outcomes[reason] = outcomes.get(reason, 0)+1
    return {'success_rate': outcomes.get('success', 0)/len(cases), 'outcomes': outcomes,
            'mean_success_steps': float(np.mean(lengths)) if lengths else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'outputs/main_20261014')
    p.add_argument('--seed', type=int, default=1401)
    p.add_argument('--expert-cases', type=int, default=160)
    p.add_argument('--dagger-cases', type=int, default=60)
    p.add_argument('--dagger-rounds', type=int, default=3)
    p.add_argument('--ppo-steps', type=int, default=16384)
    args = p.parse_args()
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); protected = original_files()
    cfg = json.loads((ROOT/'wsl-support/vision_mpc/config.json').read_text())
    rl = json.loads((ROOT/'wsl-support/vision_ppo/config.json').read_text())
    rl.update(learning_rate=.00003, entropy_coef=.0001, epochs=4, rollout_steps=1024, hidden=256)
    validation = generate_cases(40, 2026101403)
    (args.output/'validation_cases.json').write_text(json.dumps(validation, indent=2))
    frames = []
    for case in generate_cases(24, args.seed)+random_obstacle_cases(8, args.seed+1):
        env = create_case(case)
        try:
            frames.append(render_rgb(env))
        finally:
            close(env)
    pixel, perception_history = fit_pixel_net(frames, args.output, args.seed)
    (args.output/'perception_training.json').write_text(json.dumps(perception_history, indent=2))
    print('CNN TRAINED '+json.dumps(perception_history[-1]), flush=True)
    actor = ActorCritic(256)
    actor.log_std.data.fill_(-2.5)
    best_score = (-1., -float('inf')); stages = []
    def checkpoint(name, stage, result):
        torch.save({'perception': pixel.state_dict(), 'model': actor.state_dict(), 'hidden': 256,
                    'config': rl, 'seed': args.seed, 'stage': stage, 'validation': result,
                    'input': 'raw RGB images and own action history; no externally supplied simulator state'}, args.output/name)
    def assess(stage):
        nonlocal best_score
        result = validate(pixel, actor, validation, cfg)
        result.update(stage=stage)
        stages.append(result)
        (args.output/'validation.json').write_text(json.dumps(stages, indent=2))
        checkpoint(stage+'.pt', stage, result)
        score = (result['success_rate'], -(result['mean_success_steps'] or float('inf')))
        if score > best_score:
            best_score = score; checkpoint('best.pt', stage, result)
        print('VALIDATION '+json.dumps(result), flush=True)
    records, expert_outcomes = collect(generate_cases(args.expert_cases, args.seed+10), cfg, pixel, actor, 1., rng)
    fit_actor(actor, records, 100, rng)
    assess('bc')
    for round_id in range(args.dagger_rounds):
        cases = generate_cases(args.dagger_cases, args.seed+100+round_id)
        extra, outcomes = collect(cases, cfg, pixel, actor, max(0., .4-.2*round_id), rng)
        records.extend(extra)
        fit_actor(actor, records, 65, rng)
        assess(f'dagger{round_id+1}')
    # PPO starts from the validation-selected imitation checkpoint.
    initial = torch.load(args.output/'best.pt', weights_only=False)
    actor.load_state_dict(initial['model'])
    torch.save(initial, args.output/'imitation_best.pt')
    optimizer = torch.optim.Adam(actor.parameters(), lr=rl['learning_rate'], eps=1e-5)
    teacher_obs = torch.from_numpy(np.stack([r[0] for r in records])).float()
    teacher_actions = torch.from_numpy(np.stack([r[1] for r in records])).float()
    total = 0; episode_id = 0
    controller = LearnedController(pixel, actor)
    def reset():
        nonlocal episode_id
        case = generate_cases(1, args.seed+1000+episode_id)[0]; episode_id += 1
        env = create_case(case); controller.reset()
        feature = controller.observe(render_rgb(env))
        return env, feature
    env, feature = reset()
    episode_steps = 0
    try:
        while total < args.ppo_steps:
            obs_list, z_list, lp_list, r_list, v_list, nv_list, terminated, ended = [], [], [], [], [], [], [], []
            for _ in range(min(rl['rollout_steps'], args.ppo_steps-total)):
                with torch.no_grad():
                    command, z, lp, v = actor.sample(torch.from_numpy(feature))
                phi = potential(controller.state, rl)
                _, _, done, _ = env.step(action5(command.numpy()))
                controller.pending = limit(command.numpy(), 1.)
                reason = terminal_reason(env, done); episode_steps += 1
                try:
                    next_feature = controller.observe(render_rgb(env))
                except DetectionError:
                    next_feature = feature.copy(); reason = reason or 'perception_failure'
                terminal = reason is not None
                timeout = not terminal and episode_steps >= rl['max_steps']
                value_reward = reward(phi, potential(controller.state, rl), reason or ('timeout' if timeout else None), rl)
                with torch.no_grad():
                    nv = actor.value(torch.from_numpy(next_feature)).item()
                obs_list.append(feature); z_list.append(z.numpy()); lp_list.append(lp.item()); r_list.append(value_reward)
                v_list.append(v.item()); nv_list.append(nv); terminated.append(terminal); ended.append(terminal or timeout)
                feature = next_feature; total += 1
                if terminal or timeout:
                    close(env); env, feature = reset(); episode_steps = 0
            advantages, returns = gae(np.array(r_list), np.array(v_list), np.array(nv_list),
                                      np.array(terminated), np.array(ended), rl['gamma'], rl['gae_lambda'])
            stats = update(actor, optimizer, [np.array(obs_list), np.array(z_list), np.array(lp_list), advantages, returns], rl)
            # Auxiliary teacher regularization prevents catastrophic forgetting.
            for _ in range(4):
                indices = torch.from_numpy(rng.integers(0, len(teacher_obs), 256))
                bc_loss = .5*(actor.deterministic(teacher_obs[indices])-teacher_actions[indices]).square().mean()
                optimizer.zero_grad(); bc_loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), .5); optimizer.step()
            print('PPO '+json.dumps({'steps': total, **stats}), flush=True)
            if total % 4096 == 0 or total == args.ppo_steps:
                assess(f'ppo{total}')
    finally:
        close(env)
    np.savez_compressed(args.output/'demonstrations.npz', features=teacher_obs.numpy(), actions=teacher_actions.numpy())
    assert protected == original_files(), 'Original source changed during training'
    summary = {'seed': args.seed, 'cnn_training': perception_history[-1], 'expert_outcomes': expert_outcomes,
               'demonstration_samples': len(records), 'ppo_steps': total, 'validation_best': best_score[0],
               'seconds': time.perf_counter()-started, 'original_integrity': 'passed'}
    (args.output/'training_summary.json').write_text(json.dumps(summary, indent=2))
    print('TRAINING COMPLETE '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
