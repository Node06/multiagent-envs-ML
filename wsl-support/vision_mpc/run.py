"""CLI: python wsl-support/vision_mpc/run.py --mode demo/evaluate/compare."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from vision_mpc.perception import detect, DetectionError
from vision_mpc.state_estimator import Estimator
from vision_mpc.global_planner import Planner
from vision_mpc.controller_v2 import Controller
from vision_mpc.controller_v1 import Controller as OriginalController
from vision_mpc.geometry import action5
from vision_mpc.env_adapter import create, close, render_rgb
from vision_mpc.evaluator import terminal_reason, generate_cases, perception_error_pixels, wilson, hunter_obstacle_distance
from vision_mpc.visualize import overlay, Display


def episode(case, cfg, method, out, video=False, realtime=False, show=False):
    env = create(case)
    estimator = Estimator(cfg['dt'])
    times = []; trajectory = []; reason = 'timeout'; unsafe_count = 0
    controller = None; writer = None; display = None
    vision_errors = []; min_hunter = np.inf; min_obstacle = np.inf
    prediction_errors = []; last_prediction = None
    hunter_contact_frames = 0; hunter_obstacle_min = np.inf; recovery_steps = 0
    started = time.perf_counter()
    try:
        for step in range(cfg['max_steps']):
            tick = time.perf_counter()
            rgb = render_rgb(env)
            if env._vision_stopped:
                reason = 'interrupted'
                break
            try:
                state = estimator.update(detect(rgb))
            except DetectionError as exc:
                reason = 'perception_failure'
                print(f'{case["id"]}: {exc}', flush=True)
                break
            if controller is None:
                cls = Controller if method == 'mpc' and cfg.get('controller_version', 'v2') == 'v2' else OriginalController
                controller = cls(cfg, Planner(state.detection, cfg), 'mpc' if method == 'mpc_v1' else method)
            command, debug = controller.choose(state)
            recovery_steps += int(debug.get('recovery', False))
            vision_errors.append(perception_error_pixels(env, state.detection))
            min_hunter = min(min_hunter, float(np.linalg.norm(state.detection.evader-state.detection.hunter)))
            min_obstacle = min(min_obstacle, float(np.linalg.norm(state.detection.evader-state.detection.obstacles, axis=1).min()))
            if last_prediction is not None:
                prediction_errors.append(float(np.linalg.norm(state.detection.evader-last_prediction)))
            prediction = np.asarray(debug['predicted'])
            last_prediction = prediction[0].copy() if len(prediction) else None
            unsafe_count += int(debug['unsafe'])
            trajectory.append({'step': step, 'evader': state.detection.evader.tolist(),
                               'hunter': state.detection.hunter.tolist(), 'command': command.tolist()})
            if video or show or step == 0:
                frame = overlay(rgb, state, debug, step, method)
                if step == 0:
                    cv2.imwrite(str(out/f'{method}_{case["id"]}_preview.png'), frame)
                if video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out/f'{method}_{case["id"]}.mp4'),
                                                cv2.VideoWriter_fourcc(*'mp4v'), 10, (800, 800))
                        if not writer.isOpened():
                            raise RuntimeError('Video writer unavailable')
                    writer.write(frame)
                if show:
                    if display is None:
                        display = Display()
                    if not display.update(frame):
                        reason = 'interrupted'
                        break
            _, _, done, _ = env.step(action5(command))
            contact_distance = hunter_obstacle_distance(env)
            hunter_obstacle_min = min(hunter_obstacle_min, contact_distance)
            hunter_contact_frames += int(contact_distance < .075)
            reason_now = terminal_reason(env, done)
            times.append(time.perf_counter()-tick)
            if reason_now:
                reason = reason_now
                break
            if realtime:
                time.sleep(max(0., cfg['dt']-(time.perf_counter()-tick)))
        final_rgb = render_rgb(env)
        cv2.imwrite(str(out/f'{method}_{case["id"]}_final.png'), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
        if writer:
            for _ in range(10):
                writer.write(cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
    except KeyboardInterrupt:
        reason = 'interrupted'
    finally:
        if writer:
            writer.release()
        close(env)
        if display:
            display.close()
    (out/f'{method}_{case["id"]}_trajectory.json').write_text(json.dumps(trajectory), encoding='utf-8')
    return {'case': case['id'], 'method': method, 'reason': reason, 'steps': len(trajectory),
            'group': case.get('group', 'unstratified'),
            'seconds': round(time.perf_counter()-started, 3),
            'decision_p50_ms': round(float(np.median(times))*1000, 2) if times else 0,
            'decision_p95_ms': round(float(np.percentile(times, 95))*1000, 2) if times else 0,
            'unsafe_predictions': unsafe_count,
            'min_hunter_center_distance': round(min_hunter, 5) if np.isfinite(min_hunter) else None,
            'min_obstacle_center_distance': round(min_obstacle, 5) if np.isfinite(min_obstacle) else None,
            'max_center_error_pixels': round(max(vision_errors), 3) if vision_errors else None,
            'mean_one_step_prediction_error': round(float(np.mean(prediction_errors)), 5) if prediction_errors else None,
            'hunter_obstacle_contact_frames': hunter_contact_frames,
            'recovery_steps': recovery_steps,
            'hunter_obstacle_min_distance': round(hunter_obstacle_min, 5) if np.isfinite(hunter_obstacle_min) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['demo', 'evaluate', 'compare'], default='demo')
    p.add_argument('--method', choices=['mpc', 'mpc_v1', 'astar', 'direct', 'no_hunter', 'no_delay'], default='mpc')
    p.add_argument('--methods', nargs='+', choices=['mpc', 'mpc_v1', 'astar', 'direct', 'no_hunter', 'no_delay'],
                   help='Explicit methods for a paired evaluation')
    p.add_argument('--config', type=Path, default=HERE/'config.json')
    p.add_argument('--controller-version', choices=['v1', 'v2'],
                   help='Use v1 to reproduce historical mpc-labelled experiments')
    p.add_argument('--cases', type=Path)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--video', action='store_true')
    p.add_argument('--realtime', action='store_true')
    p.add_argument('--show', action='store_true', help='Display annotated RGB perception and predicted path')
    p.add_argument('--output', type=Path)
    p.add_argument('--hunter', nargs=2, type=float, default=[0., 0.])
    p.add_argument('--evader', nargs=2, type=float, default=[.7, .7])
    p.add_argument('--goal', nargs=2, type=float, default=[-.5, -.5])
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    cfg['controller_version'] = args.controller_version or cfg.get('controller_version', 'v2')
    cfg.setdefault('recovery_horizon', 50)
    if args.max_steps:
        cfg['max_steps'] = args.max_steps
    out = args.output or HERE/'outputs'/f'{args.mode}_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=True)
    if args.cases:
        cases = json.loads(args.cases.read_text(encoding='utf-8-sig'))
    elif args.mode == 'demo':
        cases = [{'id': 'demo', 'hunter': args.hunter, 'evader': args.evader, 'goal': args.goal}]
    else:
        cases = generate_cases(args.count, args.seed)
    (out/'cases.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    (out/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    methods = args.methods or (['direct', 'astar', 'mpc', 'no_hunter'] if args.mode == 'compare' else [args.method])
    rows = []
    for method in methods:
        for case in cases:
            row = episode(case, cfg, method, out, args.video, args.realtime, args.show)
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (out/'results.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader(); writer.writerows(rows)
    summary = {}
    for method in methods:
        selected = [r for r in rows if r['method'] == method]
        successes = [r for r in selected if r['reason'] == 'success']
        summary[method] = {'episodes': len(selected), 'successes': len(successes),
                           'success_rate': len(successes)/len(selected),
                           'success_rate_95ci': wilson(len(successes), len(selected)),
                           'outcomes': {reason: sum(r['reason'] == reason for r in selected)
                                        for reason in sorted({r['reason'] for r in selected})},
                           'mean_success_steps': float(np.mean([r['steps'] for r in successes])) if successes else None}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'OUTPUT: {out}', flush=True)


if __name__ == '__main__':
    main()
