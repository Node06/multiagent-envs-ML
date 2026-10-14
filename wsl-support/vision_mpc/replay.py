"""Replay recorded commands in a fresh original environment and export video."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
from vision_mpc.env_adapter import create, close, render_rgb
from vision_mpc.perception import detect
from vision_mpc.state_estimator import Estimator
from vision_mpc.geometry import action5
from vision_mpc.visualize import overlay


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path); p.add_argument('--case', required=True)
    p.add_argument('--method', default='mpc')
    args = p.parse_args()
    case = next(c for c in json.loads((args.results/'cases.json').read_text()) if c['id'] == args.case)
    recorded = json.loads((args.results/f'{args.method}_{args.case}_trajectory.json').read_text())
    env = create(case); est = Estimator(.1)
    target = args.results/f'{args.method}_{args.case}_replay.mp4'
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), 10, (800, 800))
    if not writer.isOpened():
        raise RuntimeError('Video writer unavailable')
    try:
        for row in recorded:
            rgb = render_rgb(env)
            s = est.update(detect(rgb))
            frame = overlay(rgb, s, {'route': [r['evader'] for r in recorded[:row['step']+1]]},
                            row['step'], args.method+' recorded trail')
            writer.write(frame)
            env.step(action5(row['command']))
        final = cv2.cvtColor(render_rgb(env), cv2.COLOR_RGB2BGR)
        for _ in range(20):
            writer.write(final)
    finally:
        writer.release(); close(env)
    print(target)


if __name__ == '__main__':
    main()
