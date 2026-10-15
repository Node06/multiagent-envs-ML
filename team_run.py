"""Portable entry point. Run with the Python of your project environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'mpc', 'ppo', 'train', 'evaluate', 'tests', 'learned', 'learned-train', 'learned-evaluate', 'learned-optimize', 'learned-optimization-evaluate', 'learned-refine', 'learned-refinement-evaluate', 'model-mpc', 'model-collect', 'model-train', 'model-evaluate'])
    args, extra = p.parse_known_args()
    scripts = {'check': 'team-share/check_environment.py', 'mpc': 'vision_mpc/run.py',
               'ppo': 'vision_ppo/demo.py', 'train': 'vision_ppo/train.py', 'evaluate': 'vision_ppo/evaluate.py',
               'learned': 'learned_control/demo.py', 'learned-train': 'learned_control/train.py',
               'learned-evaluate': 'learned_control/evaluate.py', 'learned-optimize': 'learned_control/optimize.py',
               'learned-optimization-evaluate': 'learned_control/optimization_evaluate.py',
               'learned-refine': 'learned_control/refine.py', 'learned-refinement-evaluate': 'learned_control/refinement_evaluate.py',
               'model-mpc': 'learned_mpc/demo.py', 'model-collect': 'learned_mpc/collect.py',
               'model-train': 'learned_mpc/train.py', 'model-evaluate': 'learned_mpc/evaluate.py'}
    if args.mode == 'tests':
        for folder in ('vision_mpc/tests', 'vision_ppo/tests', 'learned_control/tests', 'learned_mpc/tests'):
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT/'wsl-support'/folder), '-v'],
                           cwd=ROOT, check=True)
    else:
        defaults = ['--mode', 'demo'] if args.mode == 'mpc' and '--mode' not in extra else []
        subprocess.run([sys.executable, str(ROOT/'wsl-support'/scripts[args.mode]), *defaults, *extra],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
