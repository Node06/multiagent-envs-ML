"""Replay selected frozen-model successes/failures without retraining."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from checkpoint import resolve_checkpoint

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', type=Path)
    args = p.parse_args()
    root = args.experiment.resolve()
    selection = json.loads((root/'selected_model.json').read_text())
    cases = {c['id']: c for c in json.loads((root/'holdout/cases.json').read_text())}
    with (root/'holdout/results.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['method'] == selection['method']]
    chosen = []
    for reason in ('success', 'caught', 'obstacle_collision', 'perception_failure', 'timeout'):
        row = next((r for r in rows if r['reason'] == reason), None)
        if row:
            chosen.append(row)
    out = root/'representative_replays'
    out.mkdir(exist_ok=True)
    (out/'cases.json').write_text(json.dumps([cases[r['case']] for r in chosen], indent=2))
    (out/'original_outcomes.json').write_text(json.dumps(chosen, indent=2))
    subprocess.run([sys.executable, str(HERE/'evaluate.py'), '--checkpoints', str(resolve_checkpoint(selection, root)),
                    '--cases', str(out/'cases.json'), '--baselines', '--record-count', str(len(chosen)),
                    '--output', str(out)], check=True)
    with (out/'results.csv').open() as f:
        replays = {r['case']: r for r in csv.DictReader(f)}
    for r in chosen:
        replay = replays[r['case']]
        if (r['reason'], r['steps']) != (replay['reason'], replay['steps']):
            raise RuntimeError(f'Replay differs from frozen test: {r["case"]}')
    strips = []
    for r in chosen:
        video = out/f'{selection["method"]}_{r["case"]}.mp4'
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise RuntimeError(f'Empty video: {video}')
        frames = []
        for position in (0, count//2, count-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode video: {video}')
            frames.append(cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA))
        cap.release()
        label = np.full((35, 960, 3), 255, dtype=np.uint8)
        cv2.putText(label, f'{r["reason"]}: {r["case"]} | first / middle / final',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
        strips.append(np.vstack([label, np.hstack(frames)]))
    cv2.imwrite(str(out/'replay_contact_sheet.png'), np.vstack(strips))
    print('Representative replays match frozen outcomes and steps.', flush=True)


if __name__ == '__main__':
    main()
