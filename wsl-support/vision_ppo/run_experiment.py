"""Run independent CPU training seeds, freeze models, then paired holdout tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'outputs/formal_20261013')
    p.add_argument('--steps', type=int, default=131072)
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37])
    p.add_argument('--test-count', type=int, default=60)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'experiment.json').exists():
        raise RuntimeError('Use a new output directory; never overwrite a finished experiment')
    spec = {'seeds': args.seeds, 'steps_per_seed': args.steps, 'test_count': args.test_count,
            'test_seed': 6101399, 'started_unix': time.time(),
            'training_scheme': 'independent on-policy PPO; no MPC demonstrations',
            'selection': 'highest validation success, then shortest successful episodes',
            'source_sha256': {str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in HERE.rglob('*.py') if 'outputs' not in f.parts}}
    (args.output/'experiment.json').write_text(json.dumps(spec, indent=2))
    processes, handles = [], []
    runtime_env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        for seed in args.seeds:
            handle = (args.output/f'seed{seed}.log').open('w')
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, str(HERE/'train.py'), '--seed', str(seed),
                                                '--steps', str(args.steps), '--output', str(args.output/f'seed{seed}')],
                                               stdout=handle, stderr=subprocess.STDOUT, env=runtime_env))
        remaining = set(range(len(processes)))
        while remaining:
            for i in list(remaining):
                code = processes[i].poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'Training seed {args.seeds[i]} failed ({code}); inspect its log')
                    remaining.remove(i)
                    print(f'TRAINING COMPLETE seed={args.seeds[i]}', flush=True)
            if remaining:
                time.sleep(5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for handle in handles:
            handle.close()
    # Generate holdout only after all selected checkpoints have been frozen.
    checkpoints = [str(args.output/f'seed{seed}/best.pt') for seed in args.seeds]
    with (args.output/'evaluation.log').open('w') as handle:
        subprocess.run([sys.executable, str(HERE/'evaluate.py'), '--checkpoints', *checkpoints,
                        '--count', str(args.test_count), '--seed', str(spec['test_seed']),
                        '--baselines', 'astar', 'mpc_v1', '--record-count', '1',
                        '--output', str(args.output/'holdout')], check=True,
                       stdout=handle, stderr=subprocess.STDOUT, env=runtime_env)
    subprocess.run([sys.executable, str(HERE/'report.py'), str(args.output)], check=True)
    spec['finished_unix'] = time.time()
    (args.output/'experiment.json').write_text(json.dumps(spec, indent=2))
    print('EXPERIMENT COMPLETE '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
