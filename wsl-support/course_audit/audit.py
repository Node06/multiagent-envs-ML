import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE.parent), str(ROOT)]
from vision_mpc.env_adapter import create, close, render_rgb
from vision_mpc.perception import detect
from vision_mpc.evaluator import generate_cases, terminal_reason
from vision_mpc.geometry import action5


def original_files(raw=False):
    files = list((ROOT/'multiagent').rglob('*.py'))+list((ROOT/'agents').rglob('*.py'))
    files += [ROOT/n for n in ('demo.py', 'make_env.py', 'ppo.py', 'ppo_random.py', 'result_plot.py', 'setup.py', 'readme.md')]
    # Git attributes normalize text line endings between Windows and Linux.
    # No other whitespace/content is ignored, and the author's raw hashes remain recorded.
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes() if raw else p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(files)}


def create_case(case):
    # Only initialization differs; the original pursuit/physics code is untouched.
    env = create(case)
    if 'obstacles' in case:
        env.reset(np.array([case['hunter'], case['evader']]), case['goal'], case['obstacles'])
    for agent in env.world.agents:
        agent.action.u = np.zeros(2)
        agent.action.c = np.zeros(2)
    return env


def random_obstacle_cases(count, seed):
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < count:
        p = rng.uniform(-.72, .72, (6, 2))
        distances = np.linalg.norm(p[:, None]-p[None], axis=-1)+np.eye(6)*10
        if distances.min() < .18 or min(np.linalg.norm(p[i]-p[j]) for i in range(3) for j in range(i)) < .25:
            continue
        cases.append({'id': f'random_obstacles_{seed}_{len(cases):03}', 'hunter': p[0].tolist(),
                      'evader': p[1].tolist(), 'goal': p[2].tolist(), 'obstacles': p[3:].tolist(),
                      'group': 'random_obstacles_stress'})
    return cases


def errors(env, detection):
    # Ground truth is diagnostics-only, never supplied to a policy or reward feature.
    actual = [env.world.agents[0].state.p_pos, env.world.agents[1].state.p_pos, env.world.check[0].state.p_pos]
    predicted = [detection.hunter, detection.evader, detection.goal]
    result = {key: float(np.linalg.norm(p-q)*400) for key, p, q in zip(('hunter', 'evader', 'goal'), predicted, actual)}
    cost = np.linalg.norm(detection.obstacles[:, None]-np.array([o.state.p_pos for o in env.world.landmarks])[None], axis=-1)
    i, j = linear_sum_assignment(cost)
    result['obstacles'] = float(cost[i, j].max()*400)
    result['radius_error'] = float(np.max(np.abs(detection.obstacle_radii-.05))*400)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=HERE/'outputs/audit_20261014')
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    baseline = HERE/'original_integrity.json'
    now = original_files()
    if not baseline.exists():
        baseline.write_text(json.dumps({'git_baseline': 'ca0f9ab', 'normalization': 'CRLF to LF only',
                                        'sha256': now, 'raw_sha256': original_files(raw=True)}, indent=2))
    existing = json.loads(baseline.read_text())
    if 'normalization' not in existing:
        assert existing['sha256'] == original_files(raw=True), 'Original bytes changed before canonical hashing'
        baseline.write_text(json.dumps({'git_baseline': 'ca0f9ab', 'normalization': 'CRLF to LF only',
                                        'sha256': now, 'raw_sha256': existing['sha256']}, indent=2))
    recorded = json.loads(baseline.read_text())['sha256']
    if now != recorded:
        raise RuntimeError('Protected original code changed')
    if args.verify_only:
        print('PASS: all protected original source SHA256 unchanged')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    fixed = [{**c, 'group': 'course_fixed_obstacles'} for c in generate_cases(args.count, 2026101401)]
    random = random_obstacle_cases(args.count, 2026101402)
    (args.output/'fixed_cases.json').write_text(json.dumps(fixed, indent=2))
    (args.output/'random_obstacle_cases.json').write_text(json.dumps(random, indent=2))
    rows = []
    for case in fixed+random:
        env = create_case(case)
        try:
            row = {'case': case['id'], 'group': case['group'], **errors(env, detect(render_rgb(env)))}
            rows.append(row)
        finally:
            close(env)
    with (args.output/'perception.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    hunter_rows = []
    case = {'id': 'stationary_capture', 'hunter': [-.4, 0.], 'evader': [.4, 0.], 'goal': [-.6, -.6]}
    env = create_case(case)
    try:
        reason = None
        for step in range(200):
            _, _, done, _ = env.step(action5(np.zeros(2)))
            hunter, evader = env.world.agents
            delta = evader.state.p_pos-hunter.state.p_pos
            command = hunter.action.u
            cosine = float(np.dot(command, delta)/(np.linalg.norm(command)*np.linalg.norm(delta)))
            velocity = hunter.state.p_vel
            hunter_rows.append({'step': step+1, 'distance': float(np.linalg.norm(delta)),
                                'queued_acceleration_towards_prey_cosine': cosine,
                                'velocity_towards_prey_cosine': float(np.dot(velocity, delta)/(np.linalg.norm(velocity)*np.linalg.norm(delta))) if np.linalg.norm(velocity) else 0.})
            reason = terminal_reason(env, done)
            if reason:
                break
    finally:
        close(env)
    with (args.output/'hunter.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hunter_rows[0])); w.writeheader(); w.writerows(hunter_rows)
    summary = {'original_integrity': 'passed', 'scenes': len(rows), 'hunter_stationary_test': {
        'reason': reason, 'steps': len(hunter_rows), 'min_command_alignment_cosine': min(r['queued_acceleration_towards_prey_cosine'] for r in hunter_rows)}}
    for group in ('course_fixed_obstacles', 'random_obstacles_stress'):
        selected = [r for r in rows if r['group'] == group]
        summary[group] = {k: {'mean_pixels': float(np.mean([r[k] for r in selected])),
                             'p95_pixels': float(np.percentile([r[k] for r in selected], 95)),
                             'max_pixels': float(max(r[k] for r in selected))}
                          for k in ('hunter', 'evader', 'goal', 'obstacles', 'radius_error')}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    assert reason == 'caught' and summary['hunter_stationary_test']['min_command_alignment_cosine'] > .99999
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
