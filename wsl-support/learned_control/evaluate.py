"""Frozen paired RGB-model evaluation and optional random-obstacle stress tests."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import shutil
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE.parent), str(ROOT)]
from course_audit.audit import create_case, errors, original_files, random_obstacle_cases
from vision_mpc.env_adapter import render_rgb, close
from vision_mpc.perception import detect, DetectionError
from vision_mpc.state_estimator import Estimator
from vision_mpc.global_planner import Planner
from vision_mpc.controller_v2 import Controller
from vision_mpc.geometry import action5
from vision_mpc.evaluator import generate_cases, terminal_reason, wilson
from vision_ppo.model import ActorCritic
from vision_ppo.observation import encode
from learned_control.runtime import load_controller


def episode(case, method, controller, actor, cfg, output, video=False, show=False):
    env = create_case(case)
    estimator = Estimator(.1); pending = np.zeros(2); teacher = None
    if controller:
        controller.reset()
    samples, timings, trajectory = [], [], []
    reason = 'timeout'; writer = None
    hunter_contacts = 0
    hunter_alignment = []
    min_hunter_distance = float('inf')
    try:
        for step in range(cfg['max_steps']):
            rgb = render_rgb(env)
            started = time.perf_counter()
            try:
                if controller:
                    command = controller.act(rgb)
                    state = controller.state
                else:
                    state = estimator.update(detect(rgb))
                    if actor:
                        with torch.no_grad():
                            command = actor.deterministic(torch.from_numpy(encode(state, pending))).numpy()
                    else:
                        if teacher is None:
                            teacher = Controller(cfg, Planner(state.detection, cfg), method)
                        command, _ = teacher.choose(state)
            except DetectionError:
                reason = 'perception_failure'; break
            timings.append(time.perf_counter()-started)
            samples.append(errors(env, state.detection))  # diagnostic after decision
            min_hunter_distance = min(min_hunter_distance, float(np.linalg.norm(state.detection.hunter-state.detection.evader)))
            trajectory.append({'step': step, 'hunter': state.detection.hunter.tolist(),
                               'evader': state.detection.evader.tolist(), 'command': command.tolist()})
            _, _, done, _ = env.step(action5(command)); pending = command.copy()
            # Post-action metrics only. They are never read by a controller.
            hunter, prey = env.world.agents
            delta = prey.state.p_pos-hunter.state.p_pos
            denominator = np.linalg.norm(hunter.action.u)*np.linalg.norm(delta)
            if denominator:
                hunter_alignment.append(float(np.dot(hunter.action.u, delta)/denominator))
            hunter_contacts += int(any(np.linalg.norm(hunter.state.p_pos-o.state.p_pos) <= .075 for o in env.world.landmarks))
            ending = terminal_reason(env, done)
            if video or show:
                frame = cv2.cvtColor(render_rgb(env), cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f'{method} step {step+1} {ending or ""}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
                if video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(output/f'{method}_{case["id"]}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (800, 800))
                        if not writer.isOpened():
                            raise RuntimeError('Video writer unavailable')
                    writer.write(frame)
                if show:
                    cv2.imshow('Trained image policy', frame)
                    if cv2.waitKey(1) == 27:
                        raise KeyboardInterrupt
            if ending:
                reason = ending; break
        if video:
            (output/f'{method}_{case["id"]}_trajectory.json').write_text(json.dumps(trajectory))
        return {'case': case['id'], 'group': case.get('group', 'course_fixed_obstacles'), 'method': method,
                'reason': reason, 'steps': len(trajectory), 'decision_including_cnn_p50_ms': float(np.median(timings))*1000 if timings else None,
                'max_center_error_pixels': max(max(r[k] for k in ('hunter', 'evader', 'goal', 'obstacles')) for r in samples) if samples else None,
                'mean_center_error_pixels': float(np.mean([r[k] for r in samples for k in ('hunter', 'evader', 'goal', 'obstacles')])) if samples else None,
                'hunter_obstacle_contact_frames': hunter_contacts,
                'min_hunter_command_alignment_cosine': min(hunter_alignment) if hunter_alignment else None,
                'min_hunter_center_distance': min_hunter_distance if np.isfinite(min_hunter_distance) else None}
    finally:
        if writer:
            writer.release()
        close(env)
        if show:
            cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, default=HERE/'outputs/main_20261014')
    p.add_argument('--output', type=Path)
    p.add_argument('--count', type=int, default=60)
    p.add_argument('--stress-count', type=int, default=30)
    p.add_argument('--methods', nargs='+', default=['astar', 'mpc', 'old_ppo', 'imitation', 'learned'])
    p.add_argument('--cases', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--video-count', type=int, default=1)
    p.add_argument('--show', action='store_true')
    p.add_argument('--seed', type=int, default=2026101404)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output = args.output or args.experiment/'evaluation'
    args.output.mkdir(parents=True, exist_ok=True)
    before = original_files()
    # The requested main route is PPO. Select only among completed PPO stages,
    # by validation, and keep the best imitation checkpoint as an honest ablation.
    if 'learned' in args.methods and args.checkpoint is None:
        stages = json.loads((args.experiment/'validation.json').read_text())
        candidates = [stage for stage in stages if stage['stage'].startswith('ppo')]
        if not candidates:
            raise RuntimeError('PPO refinement must finish before publishing the PPO main model')
        selected = max(candidates, key=lambda stage: (stage['success_rate'], -(stage['mean_success_steps'] or float('inf'))))
        shutil.copyfile(args.experiment/(selected['stage']+'.pt'), args.experiment/'primary_ppo.pt')
        (args.experiment/'primary_selection.json').write_text(json.dumps({'criterion': 'PPO-stage validation only; no test selection',
                                                                       **selected}, indent=2))
    cfg = json.loads((ROOT/'wsl-support/vision_mpc/config.json').read_text())
    models = {}
    frozen = {}
    for method, name in [('learned', 'primary_ppo.pt'), ('imitation', 'imitation_best.pt')]:
        if method in args.methods:
            path = args.checkpoint if method == 'learned' and args.checkpoint else args.experiment/name
            controller, cp = load_controller(path); models[method] = controller
            frozen[method] = {'path': str(path), 'stage': cp['stage'], 'validation': cp['validation']}
    old = None
    if 'old_ppo' in args.methods:
        cp = torch.load(ROOT/'wsl-support/vision_ppo/outputs/formal_20261013/seed11/best.pt', weights_only=False)
        old = ActorCritic(cp['config']['hidden']); old.load_state_dict(cp['model']); old.eval()
    cases = json.loads(args.cases.read_text()) if args.cases else generate_cases(args.count, args.seed)
    if not args.cases:
        cases += random_obstacle_cases(args.stress_count, args.seed+1)
    (args.output/'cases.json').write_text(json.dumps(cases, indent=2))
    (args.output/'frozen_models.json').write_text(json.dumps(frozen, indent=2))
    rows = []
    for method in args.methods:
        for i, case in enumerate(cases):
            row = episode(case, method, models.get(method), old if method == 'old_ppo' else None,
                          cfg, args.output, i < args.video_count, args.show)
            rows.append(row)
            with (args.output/'results.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
            print(json.dumps(row), flush=True)
    summary = {}
    for group in sorted({r['group'] for r in rows}):
        summary[group] = {}
        for method in args.methods:
            selected = [r for r in rows if r['method'] == method and r['group'] == group]
            success = [r for r in selected if r['reason'] == 'success']
            summary[group][method] = {'episodes': len(selected), 'successes': len(success), 'success_rate': len(success)/len(selected),
                                      'success_rate_95ci': wilson(len(success), len(selected)),
                                      'outcomes': {reason: sum(r['reason'] == reason for r in selected) for reason in sorted({r['reason'] for r in selected})},
                                      'mean_success_steps': float(np.mean([r['steps'] for r in success])) if success else None,
                                      'max_center_error_pixels': max(r['max_center_error_pixels'] or 0 for r in selected)}
    assert before == original_files()
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('EVALUATION COMPLETE '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
