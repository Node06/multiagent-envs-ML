"""Finish media and audit after the active formal experiment exits successfully."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from checkpoint import resolve_checkpoint

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', type=Path)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    root = args.experiment.resolve()
    deadline = time.monotonic()+7200
    while True:
        try:
            finished = 'finished_unix' in json.loads((root/'experiment.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            finished = False
        if finished:
            break
        if not args.wait or time.monotonic() > deadline:
            raise RuntimeError('Formal experiment has not finished successfully')
        time.sleep(5)
    subprocess.run([sys.executable, str(HERE/'media.py'), str(root)], check=True)
    selected = json.loads((root/'selected_model.json').read_text())
    subprocess.run([sys.executable, str(HERE/'demo.py'), '--checkpoint', str(resolve_checkpoint(selected, root)),
                    '--output', str(root/'default_demo')], check=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE/'tests'), '-v'], check=True)
    subprocess.run([sys.executable, '-m', 'compileall', '-q', str(HERE)], check=True)
    (root/'delivery_checks.json').write_text(json.dumps({'representative_replays': 'matched frozen outcomes and steps',
                                                       'default_demo': 'recorded', 'core_tests': 'passed',
                                                       'compileall': 'passed'}, indent=2))
    print('DELIVERY CHECKS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
