"""Build a portable MPC+PPO handoff archive without local environments."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
routes = parser.add_mutually_exclusive_group()
routes.add_argument('--optimized', action='store_true')
routes.add_argument('--model-mpc', action='store_true', help='Package the trained dynamics main route after frozen testing')
args = parser.parse_args()
DEST = Path(__file__).resolve().parent/('multiagent-envs-ML_Optimized_PPO_team_20261014.zip' if args.optimized else 'multiagent-envs-ML_Learned_PPO_team_20261014.zip')
if args.model_mpc:
    DEST = Path(__file__).resolve().parent/'multiagent-envs-ML_LearnedDynamics_team_20261015.zip'
    dynamics_results = json.loads((ROOT/'wsl-support/learned_mpc/outputs/main_20261014/test/summary.json').read_text())
if args.optimized:
    optimized = json.loads((ROOT/'wsl-support/learned_control/outputs/refined_20261014/release_decision.json').read_text())
MPC_OUTPUTS = {'final_test', 'validation', 'stress_test', 'delay_ablation', 'demo_final',
               'stratified_20261013', 'optimization_dev', 'v2_regression_stratified',
               'v2_regression_ordinary', 'v2_independent', 'v2_optimized_demos'}
EXCLUDED = {'__pycache__', '.git', '.venv', '.pytest_cache', 'package-validation', '.team-validation-venv'}
LOCAL_FILES = {'EnterLinux.cmd', 'RunVisionMPC.cmd', 'RunPPO.cmd', 'configure_wsl.sh', 'shell_wsl.sh',
               'requirements-wsl.lock.txt', 'verify_wsl.py', 'platform_check.png', 'wsl_render_check.png',
               'profile.txt', 'outputs_smoke.log', 'formal_run.log', 'finalize.log'}
files = []
for directory, directories, names in os.walk(ROOT):
    directories[:] = sorted(d for d in directories if d not in EXCLUDED and not d.endswith('.egg-info') and not d.startswith('multiagent-envs-ML_'))
    for name in sorted(names):
        source = Path(directory)/name
        relative = source.relative_to(ROOT)
        if source.suffix.lower() in {'.pyc', '.pyo', '.zip'} or name in LOCAL_FILES:
            continue
        if 'outputs' in relative.parts:
            index = relative.parts.index('outputs')
            module = relative.parts[index-1]
            experiment = relative.parts[index+1] if index+1 < len(relative.parts) else ''
            allowed = (MPC_OUTPUTS if module == 'vision_mpc' else {'formal_20261013'} if module == 'vision_ppo'
                       else ({'main_20261014', 'optimized_20261014', 'refined_20261014'} if args.optimized else {'main_20261014'}) if module == 'learned_control'
                       else {'main_20261014'} if module == 'learned_mpc' and args.model_mpc
                       else {'audit_20261014'} if module == 'course_audit' else set())
            if experiment not in allowed:
                continue
        if 'invalid_unmapped_window_attempt' in relative.parts:
            continue
        files.append(source)
files.sort()
guide = (ROOT/'wsl-support/team-share/团队交接指南.md').read_bytes()
first_readme = 'multiagent-envs-ML/README_先读.md'
manifest = {'snapshot': 'Trained RGB CNN + imitation/DAgger + PPO primary model 2026-10-14; historical baselines retained',
            'completed': '10833 demonstration samples, 16384 PPO refinement steps, 450 new paired evaluations, 200 localization audits',
            'windows_validation': 'Native Python3.10 portable-path tests and CLI help passed; native OpenGL run NOT verified',
            'linux_validation': 'WSL2 Ubuntu-22.04 complete original experiments; moved-package self-check and PPO replay',
            'excluded': 'Environments, caches, author-only launchers, local editable lock and old archives',
            'sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
manifest['sha256']['README_先读.md'] = hashlib.sha256(guide).hexdigest()
if args.optimized:
    manifest['snapshot'] = 'Safety56: three-seed DAgger/PPO; one-seed training-risk refinement; two paired evaluations; 2026-10-14'
    manifest['completed'] = optimized
if args.model_mpc:
    manifest['snapshot'] = 'Trained image perception + three learned dynamics models + MPC; 2026-10-15'
    manifest['completed'] = dynamics_results
    manifest['linux_validation'] = 'WSL2 EGL offscreen rendering; 37 tests; frozen paired closed-loop evaluation'
with zipfile.ZipFile(DEST, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for source in files:
        archive.write(source, 'multiagent-envs-ML/'+source.relative_to(ROOT).as_posix())
    archive.writestr(first_readme, guide)
    archive.writestr('multiagent-envs-ML/共享包清单.json', json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(DEST) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    required = ['setup.py', 'team_run.py', 'README_先读.md', 'Run_PPO_Windows.cmd', 'Run_MPC_Windows.cmd',
                'multiagent/environment.py', 'wsl-support/vision_mpc/controller_v2.py',
                'wsl-support/vision_ppo/train.py', 'wsl-support/vision_ppo/checkpoint.py',
                'wsl-support/team-share/requirements-ppo.txt',
                'wsl-support/vision_ppo/outputs/formal_20261013/holdout/cases.json',
                'wsl-support/vision_ppo/outputs/formal_20261013/实验报告.md']
    required += [f'wsl-support/vision_ppo/outputs/formal_20261013/seed{s}/best.pt' for s in (11, 23, 37)]
    required += ['PROJECT_GUIDE.md', 'Run_Learned_Windows.cmd',
                 'wsl-support/learned_control/outputs/main_20261014/primary_ppo.pt',
                 'wsl-support/learned_control/outputs/main_20261014/perception.pt',
                 'wsl-support/learned_control/outputs/main_20261014/主模型实验报告.md',
                 'wsl-support/course_audit/original_integrity.json']
    if args.optimized:
        required += ['wsl-support/learned_control/outputs/optimized_20261014/primary_ppo.pt',
                     'wsl-support/learned_control/outputs/optimized_20261014/优化实验报告.md',
                     'wsl-support/learned_control/outputs/refined_20261014/primary_ppo.pt',
                     'wsl-support/learned_control/outputs/refined_20261014/最终优化报告.md']
    if args.model_mpc:
        required += ['Run_LearnedDynamics_Windows.cmd', 'wsl-support/learned_mpc/README.md',
                     'wsl-support/learned_mpc/outputs/main_20261014/world_empirical.pt',
                     'wsl-support/learned_mpc/outputs/main_20261014/perception.pt',
                     'wsl-support/learned_mpc/outputs/main_20261014/transitions.npz',
                     'wsl-support/learned_mpc/outputs/main_20261014/validation/selection.json',
                     'wsl-support/learned_mpc/outputs/main_20261014/test/results.csv',
                     'wsl-support/learned_mpc/outputs/main_20261014/test/summary.json']
    for name in required:
        assert 'multiagent-envs-ML/'+name in names, name
    assert not any('/.venv/' in n or '__pycache__/' in n or '.egg-info/' in n or n.endswith('.zip') for n in names)
    for relative, digest in manifest['sha256'].items():
        assert hashlib.sha256(archive.read('multiagent-envs-ML/'+relative)).hexdigest() == digest
print(f'PACKAGE: {DEST}')
print(f'FILES: {len(names)}; SIZE: {DEST.stat().st_size/1024/1024:.2f} MiB; CRC AND SHA256 VERIFIED')
