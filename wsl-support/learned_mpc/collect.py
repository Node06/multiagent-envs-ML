"""Collect body transitions exclusively from consecutive RGB images."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from course_audit.audit import create_case, original_files, random_obstacle_cases
from vision_mpc.env_adapter import render_rgb, close
from vision_mpc.evaluator import generate_cases, terminal_reason
from vision_mpc.state_estimator import Estimator
from vision_mpc.geometry import action5, limit, unit
from vision_mpc.dynamics import borders
from vision_mpc.global_planner import Planner
from vision_mpc.controller_v2 import Controller
from vision_mpc.perception import DetectionError
from learned_control.perception import PixelNet
from learned_mpc.model import inputs


def cases(count, seed):
    result = generate_cases(count, seed)
    random = random_obstacle_cases(max(1, count//5), seed+1)
    for i, case in enumerate(random): result[i*5] = case
    # Contact examples are physically non-overlapping, with the hunter close to
    # a fixed obstacle. Only reset positions change; the environment is untouched.
    rng = np.random.default_rng(seed+2)
    obstacles = np.array([[-.35, .35], [.35, .35], [0, -.35]])
    for i in range(2, count, 5):
        for _ in range(1000):
            center = obstacles[int(rng.integers(3))]
            theta = rng.uniform(-np.pi, np.pi); direction = np.array([np.cos(theta), np.sin(theta)])
            hunter = center+direction*rng.uniform(.078, .12)
            prey = center-direction*rng.uniform(.3, .55); goal = rng.uniform(-.7, .7, 2)
            points = np.array([hunter, prey, goal])
            if np.max(abs(points)) > .74 or np.min(np.linalg.norm(points[1:, None]-obstacles[None], axis=-1)) < .13: continue
            if min(np.linalg.norm(points[k]-points[j]) for k in range(3) for j in range(k)) < .25: continue
            result[i] = dict(id=f'contact_{seed}_{i:03}', hunter=hunter.tolist(), evader=prey.tolist(), goal=goal.tolist())
            break
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'outputs/main_20261014')
    p.add_argument('--train-cases', type=int, default=120)
    p.add_argument('--validation-cases', type=int, default=30)
    p.add_argument('--test-cases', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=60)
    args = p.parse_args(); torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'transitions.npz').exists(): raise RuntimeError('Data already frozen; choose a new output')
    before = original_files()
    cp = torch.load(HERE.parent/'learned_control/outputs/main_20261014/primary_ppo.pt', map_location='cpu', weights_only=False)
    pixel = PixelNet(); pixel.load_state_dict(cp['perception']); pixel.eval()
    torch.save(pixel.state_dict(), args.output/'perception.pt')
    cfg = json.loads((HERE.parent/'vision_mpc/config.json').read_text())
    records, targets, episode_ids, splits, manifest = [], [], [], [], []
    episode_id = 0
    for split, count, seed in [('train', args.train_cases, 2026101471), ('validation', args.validation_cases, 2026101474), ('test', args.test_cases, 2026101477)]:
        for case in cases(count, seed):
            manifest.append({'split': split, 'case': case})
            env = create_case(case); estimator = Estimator(.1)
            pending = np.zeros(2); teacher = None; initial = True
            rng = np.random.default_rng(seed+episode_id); reason = 'step_limit'
            try:
                try: state = estimator.update(pixel.detect(render_rgb(env)))
                except DetectionError: reason = 'perception_failure'; continue
                for step in range(args.max_steps):
                    d = state.detection
                    if teacher is None: teacher = Controller(cfg, Planner(d, cfg))
                    # Whole episodes mix expert and perturbed controls. The label
                    # always uses the action that was queued before this step.
                    if episode_id % 3 == 0:
                        command = limit(rng.normal(size=2), 1.)
                    else:
                        command, _ = teacher.choose(state)
                        if episode_id % 3 == 1: command = limit(command+rng.normal(0, .2, 2), 1.)
                    hunter_action = np.zeros(2) if initial else unit(d.evader-d.hunter)*.85
                    obstacles = np.concatenate((d.obstacles, borders()))
                    radii = np.r_[d.obstacle_radii, np.full(80, .15)]
                    rows = [inputs(d.evader, state.evader_velocity, pending, 0, obstacles, radii)[0],
                            inputs(d.hunter, state.hunter_velocity, hunter_action, 1, obstacles, radii)[0]]
                    _, _, done, _ = env.step(action5(command))
                    teacher.pending = command.copy(); pending = command.copy(); initial = False
                    try: next_state = estimator.update(pixel.detect(render_rgb(env)))
                    except DetectionError: reason = 'perception_failure'; break
                    nd = next_state.detection
                    # Exclude colored-object overlap and extreme silhouette
                    # artifacts using only image geometry, never true positions.
                    if min(np.linalg.norm(d.evader-d.hunter), np.linalg.norm(nd.evader-nd.hunter)) > .07 and np.linalg.norm(nd.evader-nd.goal) > .07:
                        velocity = [(nd.evader-d.evader)/.1, (nd.hunter-d.hunter)/.1]
                        for row, target in zip(rows, velocity):
                            if np.linalg.norm(target) < .45:
                                records.append(row); targets.append(target); episode_ids.append(episode_id); splits.append(split)
                    state = next_state
                    ending = terminal_reason(env, done)
                    if ending: reason = ending; break
            finally: close(env)
            episode_id += 1
            print(json.dumps({'split': split, 'case': case['id'], 'body_transitions': len(records), 'reason': reason}), flush=True)
    np.savez_compressed(args.output/'transitions.npz', x=np.array(records, np.float32), y=np.array(targets, np.float32),
                        episode=np.array(episode_ids), split=np.array(splits))
    (args.output/'collection_cases.json').write_text(json.dumps(manifest, indent=2))
    assert original_files() == before
    (args.output/'collection_summary.json').write_text(json.dumps({'episodes': episode_id, 'body_transitions': len(records),
                                                                   'splits': {key: splits.count(key) for key in set(splits)},
                                                                   'labels': 'next position minus current position from RGB /dt; queued actions only',
                                                                   'original_integrity': 'passed'}, indent=2))
    print('IMAGE TRANSITION COLLECTION COMPLETE', flush=True)


if __name__ == '__main__': main()
