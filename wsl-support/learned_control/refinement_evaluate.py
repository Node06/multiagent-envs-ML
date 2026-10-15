"""Paired fresh test of the frozen training-risk refinement, without tuning."""
import argparse
import csv
import hashlib
import shutil
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from learned_control.runtime import load_controller
from learned_control.evaluate import episode
from course_audit.audit import original_files, random_obstacle_cases
from vision_mpc.evaluator import generate_cases, wilson


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, default=HERE/'outputs/refined_20261014')
    p.add_argument('--seed', type=int, default=2026101450)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--stress-count', type=int, default=30)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(); args.experiment = args.experiment.resolve(); torch.set_num_threads(1)
    out = args.experiment/'evaluation'; out.mkdir(exist_ok=True)
    if (out/'summary.json').exists(): raise RuntimeError('Completed frozen evaluation exists')
    assert (args.experiment/'training_summary.json').exists()
    before = original_files()
    paths = {'previous_main': HERE/'outputs/main_20261014/primary_ppo.pt',
             'optimized_v2': HERE/'outputs/optimized_20261014/primary_ppo.pt',
             'refined_main': args.experiment/'primary_ppo.pt',
             'refined_imitation': args.experiment/'imitation_best.pt'}
    models, frozen = {}, {}
    for name, path in paths.items():
        model, cp = load_controller(path); models[name] = model
        frozen[name] = {'path': str(path.relative_to(HERE.parent.parent)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'stage': cp['stage'], 'validation': cp['validation'], 'seed': cp['seed']}
    stages = json.loads((args.experiment/'validation.json').read_text())
    initial = next(s for s in stages if s['stage'] == 'initial')
    score = lambda r: (r['success_rate'], -(r['mean_success_steps'] or float('inf')))
    selected = 'refined_main' if score(frozen['refined_main']['validation']) > score(initial) else 'optimized_v2'
    frozen['selected_candidate'] = selected
    frozen['selection_criterion'] = 'same 80-case validation only; compare refinement to its frozen source before test'
    if args.resume and json.loads((out/'frozen_models.json').read_text()) != frozen:
        raise RuntimeError('Resume model hashes or selection differ from the frozen run')
    (out/'frozen_models.json').write_text(json.dumps(frozen, indent=2))
    cases = generate_cases(args.count, args.seed)
    for c in cases: c['group'] = 'fresh_course_holdout'
    cases += random_obstacle_cases(args.stress_count, args.seed+1)
    for c in json.loads((HERE/'outputs/main_20261014/evaluation/cases.json').read_text()):
        if 'obstacles' not in c:
            c = dict(c); c['group'] = 'published_course_regression'; cases.append(c)
    cases.append(dict(id='default_demo', hunter=[0, 0], evader=[.7, .7], goal=[-.5, -.5], group='default_demo'))
    if args.resume and json.loads((out/'cases.json').read_text()) != cases:
        raise RuntimeError('Resume cases differ from the frozen run')
    (out/'cases.json').write_text(json.dumps(cases, indent=2))
    cfg = json.loads((HERE.parent/'vision_mpc/config.json').read_text())
    rows = []
    if args.resume:
        shutil.copyfile(out/'results.csv', out/'before_hidden_resume.csv')
        with (out/'results.csv').open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in ('steps', 'hunter_obstacle_contact_frames'):
                row[key] = int(row[key])
            for key in ('decision_including_cnn_p50_ms', 'max_center_error_pixels', 'mean_center_error_pixels', 'min_hunter_command_alignment_cosine', 'min_hunter_center_distance'):
                row[key] = float(row[key]) if row[key] else None
    completed = {(r['method'], r['case']) for r in rows}
    if len(completed) != len(rows): raise RuntimeError('Duplicate saved episodes')
    print('RESUME completed episodes: '+str(len(rows)), flush=True)
    for method in [*models, 'mpc']:
        for i, case in enumerate(cases):
            if (method, case['id']) in completed:
                continue
            row = episode(case, method, models.get(method), None, cfg, out, video=(i == 0 or case['id'] == 'default_demo'))
            rows.append(row)
            with (out/'results.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            if i % 20 == 0 or case['id'] == 'default_demo': print('EVAL '+json.dumps(row), flush=True)
        print('METHOD COMPLETE '+method, flush=True)
    summary = {}
    for group in sorted({r['group'] for r in rows}):
        summary[group] = {}
        for method in [*models, 'mpc']:
            r = [a for a in rows if a['group'] == group and a['method'] == method]; success = [a for a in r if a['reason'] == 'success']
            summary[group][method] = {'episodes': len(r), 'successes': len(success), 'success_rate': len(success)/len(r),
                                      'success_rate_95ci': wilson(len(success), len(r)),
                                      'outcomes': {reason: sum(a['reason'] == reason for a in r) for reason in sorted({a['reason'] for a in r})},
                                      'mean_success_steps': float(np.mean([a['steps'] for a in success])) if success else None}
    assert before == original_files()
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('REFINEMENT FROZEN EVALUATION COMPLETE', flush=True)


if __name__ == '__main__': main()
