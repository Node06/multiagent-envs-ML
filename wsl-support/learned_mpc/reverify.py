"""Reproducible fresh-scenario verification of the frozen RGB learned MPC."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from collections import Counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from course_audit.audit import original_files, random_obstacle_cases
from learned_control.evaluate import episode
from learned_mpc.runtime import load_controller
from vision_mpc.evaluator import generate_cases, wilson
from vision_mpc.stratified_cases import generate


def main():
    torch.set_num_threads(1)
    seed = 2026101517
    source = HERE/'outputs/main_20261014'
    out = HERE/'outputs/reverify_20261015/fresh_test'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise RuntimeError('Completed verification is frozen')
    cases = [dict(c, group='fresh_course_holdout') for c in generate_cases(100, seed)]
    cases += random_obstacle_cases(30, seed+1)
    cases += [dict(c, id=f'challenge_{seed+2}_{c["id"]}') for c in generate(2, seed+2)]
    cases += [dict(id='default_demo', group='default_demo', hunter=[0, 0],
                   evader=[.7, .7], goal=[-.5, -.5])]
    selection = json.loads((source/'validation/selection.json').read_text())
    controller, _ = load_controller(source/'world_empirical.pt',
                                    {'uncertainty_scale': selection['uncertainty_scale']})
    before = original_files()
    assert before == json.loads((HERE.parent/'course_audit/original_integrity.json').read_text())['sha256']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = dict(seed=seed, platform=platform.platform(), python=sys.version,
                  executable=sys.executable, torch=torch.__version__,
                  checkpoint_sha256=digest(source/'world_empirical.pt'),
                  perception_sha256=digest(source/'perception.pt'),
                  adapter_sha256=digest(HERE.parent/'vision_mpc/env_adapter.py'),
                  config=controller.cfg, selection=selection,
                  protocol='Frozen weights/config; fresh seeds; no training or tuning; all episodes recorded')
    for name, value in [('cases.json', cases), ('frozen.json', frozen)]:
        path = out/name
        if path.exists():
            assert json.loads(path.read_text()) == value, f'Resume mismatch: {name}'
        else:
            path.write_text(json.dumps(value, indent=2))
    rows = []
    if (out/'results.csv').exists():
        with (out/'results.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
    completed = {r['case'] for r in rows}
    assert len(completed) == len(rows)
    assert completed <= {c['id'] for c in cases}
    started = time.perf_counter()
    for case in cases:
        if case['id'] in completed:
            continue
        row = episode(case, 'learned_mpc', controller, None, controller.cfg, out, video=True)
        row['learned_prediction_calls'] = controller.world.prediction_calls
        rows.append(row)
        with (out/'results.pending.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader(); writer.writerows(rows)
        (out/'results.pending.csv').replace(out/'results.csv')
        line = json.dumps(dict(completed=len(rows), total=len(cases), **row))
        with (out/'progress.jsonl').open('a') as f:
            f.write(line+'\n')
        print(line, flush=True)
    summary = {}
    for group in sorted({r['group'] for r in rows}):
        selected = [r for r in rows if r['group'] == group]
        success = sum(r['reason'] == 'success' for r in selected)
        times = [float(r['decision_including_cnn_p50_ms']) for r in selected
                 if r['decision_including_cnn_p50_ms'] not in (None, '')]
        summary[group] = dict(episodes=len(selected), successes=success,
                              success_rate=success/len(selected), ci95=wilson(success, len(selected)),
                              outcomes=dict(Counter(r['reason'] for r in selected)),
                              mean_steps=float(np.mean([int(r['steps']) for r in selected])),
                              median_episode_decision_p50_ms=float(np.median(times)) if times else None)
    assert original_files() == before
    assert digest(source/'world_empirical.pt') == frozen['checkpoint_sha256']
    (out/'summary.json').write_text(json.dumps(dict(groups=summary, episodes=len(rows),
                protected_originals_unchanged=True, weights_unchanged=True,
                elapsed_this_invocation_seconds=time.perf_counter()-started), indent=2))
    print('VERIFICATION COMPLETE '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
