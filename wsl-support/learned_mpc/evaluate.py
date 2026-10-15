"""Closed-loop validation and frozen independent learned-model MPC tests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from learned_mpc.runtime import load_controller, AnalyticImageMPC
from learned_mpc.model import NeuralDynamics, WorldModel
from learned_control.evaluate import episode
from learned_control.runtime import load_controller as load_policy
from course_audit.audit import original_files, random_obstacle_cases
from vision_mpc.evaluator import generate_cases, wilson


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, default=HERE/'outputs/main_20261014')
    p.add_argument('--mode', choices=['smoke', 'validation', 'test'], default='smoke')
    p.add_argument('--count', type=int)
    p.add_argument('--seed', type=int, help='Fresh scenario seed; stress scenarios use seed + 1')
    p.add_argument('--resume', action='store_true', help='Continue matching frozen cases and model')
    args = p.parse_args(); torch.set_num_threads(1)
    out = args.experiment/args.mode; out.mkdir(exist_ok=True)
    if (out/'summary.json').exists(): raise RuntimeError('Completed evaluation already frozen')
    checkpoint = args.experiment/'world_empirical.pt'
    cfg = json.loads((HERE.parent/'vision_mpc/config.json').read_text())
    before = original_files(); started = time.perf_counter()
    if args.mode == 'smoke':
        cases = generate_cases(args.count or 2, args.seed if args.seed is not None else 2026101501)
        cases.append(dict(id='default_demo', hunter=[0, 0], evader=[.7, .7], goal=[-.5, -.5]))
        methods = {'learned': .25, 'mpc': None}
    elif args.mode == 'validation':
        cases = generate_cases(args.count or 30, args.seed if args.seed is not None else 2026101502)
        methods = {'learned_u0': 0., 'learned_u025': .25, 'learned_u05': .5}
    else:
        selection = json.loads((args.experiment/'validation/selection.json').read_text())
        methods = {'learned': selection['uncertainty_scale'], 'mpc': None, 'previous_ppo': None, 'untrained_world': selection['uncertainty_scale']}
        seed = args.seed if args.seed is not None else 2026101503
        cases = generate_cases(args.count or 100, seed)
        for c in cases: c['group'] = 'fresh_course_holdout'
        cases += random_obstacle_cases(30, seed + 1)
        cases.append(dict(id='default_demo', group='default_demo', hunter=[0, 0], evader=[.7, .7], goal=[-.5, -.5]))
    models = {}
    for method, scale in methods.items():
        if method.startswith('learned') or method == 'untrained_world':
            models[method], _ = load_controller(checkpoint, {'uncertainty_scale': scale})
            if method == 'untrained_world':
                torch.manual_seed(5511)
                models[method].world = WorldModel([NeuralDynamics() for _ in range(3)])
        elif method == 'previous_ppo':
            models[method], _ = load_policy(HERE.parent/'learned_control/outputs/main_20261014/primary_ppo.pt')
        elif method == 'mpc':
            reference, _ = load_controller(checkpoint)
            models[method] = AnalyticImageMPC(reference.perception, cfg)
    frozen = {'checkpoint': checkpoint.name, 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                                   'methods': methods, 'model_selection': 'validation only; no test feedback',
                                                   'rendering': 'Linux EGL offscreen; same renderer for every method'}
    rows = []
    results_path = out/'results.csv'
    if results_path.exists():
        if not args.resume:
            raise RuntimeError('Partial results exist; use --resume to verify and continue')
        if json.loads((out/'cases.json').read_text()) != cases or json.loads((out/'frozen_model.json').read_text()) != frozen:
            raise RuntimeError('Refusing resume: model, methods or cases changed')
        with results_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in ('steps', 'hunter_obstacle_contact_frames', 'learned_prediction_calls'):
                row[key] = int(row[key])
            for key in ('decision_including_cnn_p50_ms', 'max_center_error_pixels', 'mean_center_error_pixels',
                        'min_hunter_command_alignment_cosine', 'min_hunter_center_distance'):
                row[key] = float(row[key]) if row[key] else None
        allowed = {(method, case['id']) for method in methods for case in cases}
        completed = {(row['method'], row['case']) for row in rows}
        if len(completed) != len(rows) or not completed <= allowed:
            raise RuntimeError('Refusing resume: duplicate or unexpected result rows')
        print('RESUME '+str(len(rows))+' completed episodes', flush=True)
    else:
        (out/'cases.json').write_text(json.dumps(cases, indent=2))
        (out/'frozen_model.json').write_text(json.dumps(frozen, indent=2))
    completed = {(row['method'], row['case']) for row in rows}
    for method in methods:
        for i, case in enumerate(cases):
            if (method, case['id']) in completed:
                continue
            model = models.get(method)
            row = episode(case, method, model, None, cfg, out, video=(i == 0 or case['id'] == 'default_demo'))
            row['learned_prediction_calls'] = model.world.prediction_calls if hasattr(model, 'world') else 0
            rows.append(row)
            temporary = out/'results.pending.csv'
            with temporary.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            temporary.replace(results_path)
            print(json.dumps(row), flush=True)
    summary = {}
    for group in sorted({r['group'] for r in rows}):
        summary[group] = {}
        for method in methods:
            selected = [r for r in rows if r['group'] == group and r['method'] == method]
            success = sum(r['reason'] == 'success' for r in selected)
            summary[group][method] = dict(episodes=len(selected), successes=success, success_rate=success/len(selected),
                                          success_rate_95ci=wilson(success, len(selected)),
                                          outcomes={reason: sum(r['reason'] == reason for r in selected) for reason in sorted({r['reason'] for r in selected})},
                                          median_decision_ms=float(np.median([r['decision_including_cnn_p50_ms'] for r in selected if r['decision_including_cnn_p50_ms'] is not None])))
    if args.mode == 'validation':
        def score(method):
            selected = [r for r in rows if r['method'] == method]
            return (sum(r['reason'] == 'success' for r in selected), -sum(r['steps'] for r in selected))
        best = max(methods, key=score)
        (out/'selection.json').write_text(json.dumps({'method': best, 'uncertainty_scale': methods[best], 'criterion': '30 validation episodes only'}, indent=2))
    assert before == original_files()
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('CLOSED LOOP COMPLETE '+json.dumps({'seconds': time.perf_counter()-started, 'summary': summary}), flush=True)


if __name__ == '__main__': main()
