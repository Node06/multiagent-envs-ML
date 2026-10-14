"""Frozen paired evaluation. Test cases are never used by train.py."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from vision_ppo.env_wrapper import VisualEnv
from vision_ppo.model import ActorCritic
from vision_mpc.evaluator import generate_cases, wilson
from vision_mpc.global_planner import Planner
from vision_mpc.controller_v1 import Controller as ControllerV1
from vision_mpc.controller_v2 import Controller as ControllerV2
from vision_mpc.env_adapter import render_rgb


def episode(case, cfg, policy, method, mpc_cfg, env, out, record=False, show=False):
    obs = env.reset(case)
    if method in ('astar', 'mpc_v1', 'mpc_v2'):
        controller_cls = ControllerV2 if method == 'mpc_v2' else ControllerV1
        controller = controller_cls(mpc_cfg, Planner(env.state.detection, mpc_cfg),
                                    'astar' if method == 'astar' else 'mpc')
    decisions, ticks, trajectory = [], [], []
    min_hunter, min_obstacle = float('inf'), float('inf')
    writer = None
    started = time.perf_counter()
    try:
        for step in range(cfg['max_steps']):
            d = env.state.detection
            min_hunter = min(min_hunter, float(np.linalg.norm(d.evader-d.hunter)))
            min_obstacle = min(min_obstacle, float(np.min(np.linalg.norm(d.evader-d.obstacles, axis=1))))
            tick = time.perf_counter()
            if method.startswith('ppo_'):
                with torch.no_grad():
                    command = policy.deterministic(torch.from_numpy(obs)).numpy()
            else:
                command, _ = controller.choose(env.state)
            decisions.append(time.perf_counter()-tick)
            trajectory.append({'step': step, 'evader': d.evader.tolist(), 'hunter': d.hunter.tolist(),
                               'command': command.tolist()})
            obs, _, terminal, truncated, info = env.step(command)
            ticks.append(time.perf_counter()-tick)
            if record or show:
                rgb = render_rgb(env.env)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f'{method} step {step+1}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
                if info['reason']:
                    cv2.putText(frame, info['reason'], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 180), 2)
                if record:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out/f'{method}_{case["id"]}.mp4'),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('Video writer unavailable')
                    writer.write(frame)
                if show:
                    cv2.imshow('Vision PPO', frame)
                    if cv2.waitKey(1) == 27:
                        raise KeyboardInterrupt
            if terminal or truncated:
                break
        reason = info['reason']
        d = env.state.detection
        min_hunter = min(min_hunter, float(np.linalg.norm(d.evader-d.hunter)))
        min_obstacle = min(min_obstacle, float(np.min(np.linalg.norm(d.evader-d.obstacles, axis=1))))
        if record:
            (out/f'{method}_{case["id"]}_trajectory.json').write_text(json.dumps(trajectory))
        return {'case': case['id'], 'group': case.get('group', 'ordinary'), 'method': method,
                'reason': reason, 'steps': env.steps, 'seconds': time.perf_counter()-started,
                'decision_p50_ms': float(np.median(decisions))*1000,
                'decision_p95_ms': float(np.percentile(decisions, 95))*1000,
                'step_with_vision_p50_ms': float(np.median(ticks))*1000,
                'min_hunter_center_distance': min_hunter, 'min_obstacle_center_distance': min_obstacle}
    finally:
        if writer:
            writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--cases', type=Path)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--seed', type=int, default=6101399)
    p.add_argument('--baselines', nargs='*', choices=['astar', 'mpc_v1', 'mpc_v2'], default=['astar', 'mpc_v1'])
    p.add_argument('--mpc-config', type=Path, default=HERE.parent/'vision_mpc/config.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--record-count', type=int, default=0)
    p.add_argument('--show', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    checkpoints = [torch.load(path, map_location='cpu', weights_only=False) for path in args.checkpoints]
    cfg = checkpoints[0]['config']
    for cp in checkpoints:
        for key in ('max_steps', 'dt'):
            if cp['config'][key] != cfg[key]:
                raise ValueError(f'Evaluation configuration differs: {key}')
    cases = json.loads(args.cases.read_text(encoding='utf-8-sig')) if args.cases else generate_cases(args.count, args.seed)
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Require nonempty cases with unique identifiers')
    mpc_cfg = json.loads(args.mpc_config.read_text(encoding='utf-8-sig'))
    mpc_cfg['max_steps'] = cfg['max_steps']; mpc_cfg['dt'] = cfg['dt']
    models = {}
    manifest = []
    for path, cp in zip(args.checkpoints, checkpoints):
        name = f'ppo_seed{cp["seed"]}'
        if name in models:
            raise ValueError('Use one checkpoint per seed')
        model = ActorCritic(cp['config']['hidden']); model.load_state_dict(cp['model']); model.eval()
        models[name] = model
        manifest.append({'method': name, 'path': str(path.resolve()), 'steps': cp['steps'],
                         'validation_success': cp['best_rate'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'cases.json').write_text(json.dumps(cases, indent=2))
    (args.output/'manifest.json').write_text(json.dumps({'checkpoints': manifest, 'ppo_config': cfg,
                                                       'mpc_config': mpc_cfg}, indent=2))
    rows = []
    env = VisualEnv(cfg)
    try:
        for method in [*args.baselines, *models]:
            for i, case in enumerate(cases):
                row = episode(case, cfg, models.get(method), method, mpc_cfg, env, args.output,
                              i < args.record_count, args.show)
                rows.append(row)
                with (args.output/'results.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
                print(json.dumps(row), flush=True)
    finally:
        env.close()
        if args.show:
            cv2.destroyAllWindows()
    summary = {}
    for method in [*args.baselines, *models]:
        selected = [r for r in rows if r['method'] == method]
        success = [r for r in selected if r['reason'] == 'success']
        summary[method] = {'episodes': len(selected), 'successes': len(success),
                           'success_rate': len(success)/len(selected), 'success_rate_95ci': wilson(len(success), len(selected)),
                           'outcomes': {reason: sum(r['reason'] == reason for r in selected)
                                        for reason in sorted({r['reason'] for r in selected})},
                           'mean_success_steps': float(np.mean([r['steps'] for r in success])) if success else None,
                           'mean_episode_decision_p50_ms': float(np.mean([r['decision_p50_ms'] for r in selected])),
                           'mean_episode_step_with_vision_p50_ms': float(np.mean([r['step_with_vision_p50_ms'] for r in selected]))}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
