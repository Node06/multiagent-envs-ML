"""Freeze validation-selected optimization models before a new paired holdout."""
import argparse
import csv
import hashlib
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
    p.add_argument('--experiment', type=Path, default=HERE/'outputs/optimized_20261014')
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--stress-count', type=int, default=30)
    p.add_argument('--seed', type=int, default=2026101428)
    args = p.parse_args(); args.experiment = args.experiment.resolve(); torch.set_num_threads(1)
    out = args.experiment/'evaluation'; out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():
        raise RuntimeError('Frozen evaluation already exists; use a new experiment/output for a new test')
    before = original_files()
    train = json.loads((args.experiment/'training_summary.json').read_text())
    selected = train['selected']['seed']
    paths = {'previous_main': HERE/'outputs/main_20261014/primary_ppo.pt'}
    paths.update({f"seed{r['seed']}": args.experiment/f"seed{r['seed']}"/'primary_ppo.pt' for r in train['seeds']})
    paths['optimized_imitation'] = args.experiment/f'seed{selected}'/'imitation_best.pt'
    models, frozen = {}, {}
    for name, path in paths.items():
        model, cp = load_controller(path); models[name] = model
        frozen[name] = {'path': str(path.relative_to(HERE.parent.parent)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'stage': cp['stage'], 'validation': cp['validation'], 'seed': cp['seed']}
    frozen['selected_main'] = f'seed{selected}'
    (out/'frozen_models.json').write_text(json.dumps(frozen, indent=2))
    cases = generate_cases(args.count, args.seed)
    for c in cases: c['group'] = 'fresh_course_holdout'
    cases += random_obstacle_cases(args.stress_count, args.seed+1)
    regression = json.loads((HERE/'outputs/main_20261014/evaluation/cases.json').read_text())
    for c in regression:
        if 'obstacles' not in c:
            c = dict(c); c['group'] = 'published_course_regression'; cases.append(c)
    cases.append(dict(id='default_demo', hunter=[0, 0], evader=[.7, .7], goal=[-.5, -.5], group='default_demo'))
    (out/'cases.json').write_text(json.dumps(cases, indent=2))
    cfg = json.loads((HERE.parent/'vision_mpc/config.json').read_text())
    rows = []
    for method in [*models, 'mpc']:
        for i, case in enumerate(cases):
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
            r = [a for a in rows if a['group'] == group and a['method'] == method]
            success = [a for a in r if a['reason'] == 'success']
            summary[group][method] = {'episodes': len(r), 'successes': len(success), 'success_rate': len(success)/len(r),
                                      'success_rate_95ci': wilson(len(success), len(r)),
                                      'outcomes': {reason: sum(a['reason'] == reason for a in r) for reason in sorted({a['reason'] for a in r})},
                                      'mean_success_steps': float(np.mean([a['steps'] for a in success])) if success else None}
    summary['selected_main'] = f'seed{selected}'
    assert before == original_files()
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('FROZEN OPTIMIZATION EVALUATION COMPLETE', flush=True)


if __name__ == '__main__': main()
