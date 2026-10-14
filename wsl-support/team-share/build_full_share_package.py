"""Build a portable MPC+PPO handoff archive without local environments."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent/'multiagent-envs-ML_MPC_PPO_team_20261013.zip'
MPC_OUTPUTS = {'final_test', 'validation', 'stress_test', 'delay_ablation', 'demo_final',
               'stratified_20261013', 'optimization_dev', 'v2_regression_stratified',
               'v2_regression_ordinary', 'v2_independent', 'v2_optimized_demos'}
EXCLUDED = {'__pycache__', '.git', '.venv', '.pytest_cache', 'package-validation', '.team-validation-venv'}
LOCAL_FILES = {'EnterLinux.cmd', 'RunVisionMPC.cmd', 'RunPPO.cmd', 'configure_wsl.sh', 'shell_wsl.sh',
               'requirements-wsl.lock.txt', 'verify_wsl.py', 'platform_check.png', 'wsl_render_check.png',
               'profile.txt', 'outputs_smoke.log', 'formal_run.log', 'finalize.log'}
files = []
for directory, directories, names in os.walk(ROOT):
    directories[:] = sorted(d for d in directories if d not in EXCLUDED and not d.endswith('.egg-info'))
    for name in sorted(names):
        source = Path(directory)/name
        relative = source.relative_to(ROOT)
        if source.suffix.lower() in {'.pyc', '.pyo', '.zip'} or name in LOCAL_FILES:
            continue
        if 'outputs' in relative.parts:
            index = relative.parts.index('outputs')
            module = relative.parts[index-1]
            experiment = relative.parts[index+1] if index+1 < len(relative.parts) else ''
            allowed = MPC_OUTPUTS if module == 'vision_mpc' else {'formal_20261013'} if module == 'vision_ppo' else set()
            if experiment not in allowed:
                continue
        files.append(source)
files.sort()
guide = (ROOT/'wsl-support/team-share/团队交接指南.md').read_bytes()
first_readme = 'multiagent-envs-ML/README_先读.md'
manifest = {'snapshot': 'MPC V2 + visual-state PPO team handoff 2026-10-13',
            'completed': '393216 PPO training steps; 300 paired holdout episodes; MPC V2 original reports retained separately',
            'windows_validation': 'Native Python3.10 portable-path tests and CLI help passed; native OpenGL run NOT verified',
            'linux_validation': 'WSL2 Ubuntu-22.04 complete original experiments; moved-package self-check and PPO replay',
            'excluded': 'Environments, caches, author-only launchers, local editable lock and old archives',
            'sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
manifest['sha256']['README_先读.md'] = hashlib.sha256(guide).hexdigest()
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
    for name in required:
        assert 'multiagent-envs-ML/'+name in names, name
    assert not any('/.venv/' in n or '__pycache__/' in n or '.egg-info/' in n or n.endswith('.zip') for n in names)
    for relative, digest in manifest['sha256'].items():
        assert hashlib.sha256(archive.read('multiagent-envs-ML/'+relative)).hexdigest() == digest
print(f'PACKAGE: {DEST}')
print(f'FILES: {len(names)}; SIZE: {DEST.stat().st_size/1024/1024:.2f} MiB; CRC AND SHA256 VERIFIED')
