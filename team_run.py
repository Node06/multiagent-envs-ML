"""Portable entry point. Run with the Python of your project environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'mpc', 'ppo', 'train', 'evaluate', 'tests'])
    args, extra = p.parse_known_args()
    scripts = {'check': 'team-share/check_environment.py', 'mpc': 'vision_mpc/run.py',
               'ppo': 'vision_ppo/demo.py', 'train': 'vision_ppo/train.py', 'evaluate': 'vision_ppo/evaluate.py'}
    if args.mode == 'tests':
        for folder in ('vision_mpc/tests', 'vision_ppo/tests'):
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT/'wsl-support'/folder), '-v'],
                           cwd=ROOT, check=True)
    else:
        defaults = ['--mode', 'demo'] if args.mode == 'mpc' and '--mode' not in extra else []
        subprocess.run([sys.executable, str(ROOT/'wsl-support'/scripts[args.mode]), *defaults, *extra],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
