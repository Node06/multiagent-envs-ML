"""Package source, reproducible results and guides; never package an environment."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent/'multiagent-envs-ML_团队交接_V2_20261013.zip'
OUTPUTS = {'final_test', 'validation', 'stress_test', 'delay_ablation', 'demo_final',
           'stratified_20261013', 'optimization_dev', 'v2_regression_stratified',
           'v2_regression_ordinary', 'v2_independent', 'v2_optimized_demos'}
EXCLUDED = {'__pycache__', '.git', '.venv', '.pytest_cache', 'package-validation'}
LOCAL_FILES = {'EnterLinux.cmd', 'RunVisionMPC.cmd', 'configure_wsl.sh', 'shell_wsl.sh',
               'requirements-wsl.lock.txt', 'verify_wsl.py', 'platform_check.png'}
files = []
for source in sorted(ROOT.rglob('*')):
    if not source.is_file():
        continue
    relative = source.relative_to(ROOT)
    if any(part in EXCLUDED or part.endswith('.egg-info') for part in relative.parts):
        continue
    if source.suffix.lower() in {'.pyc', '.pyo', '.zip'} or source.name in LOCAL_FILES:
        continue
    if 'outputs' in relative.parts:
        pos = relative.parts.index('outputs')
        if pos+1 >= len(relative.parts) or relative.parts[pos+1] not in OUTPUTS:
            continue
    files.append(source)
manifest = {'snapshot': 'MPC V2 team handoff 2026-10-13',
            'windows_validation': 'Native Windows rendering NOT verified; run check_environment.py before use',
            'linux_validation': 'WSL2 Ubuntu-22.04, Python 3.10; original evaluation and V2 regression/independent tests completed',
            'excluded': 'Virtual environments, cache, editable installation metadata, author-specific launchers and local lock paths',
            'sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
with zipfile.ZipFile(DEST, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for source in files:
        archive.write(source, 'multiagent-envs-ML/'+source.relative_to(ROOT).as_posix())
    guide = (ROOT/'wsl-support'/'team-share'/'团队交接指南.md').read_text(encoding='utf-8')
    archive.writestr('multiagent-envs-ML/README_先读.md', guide)
    archive.writestr('multiagent-envs-ML/共享包清单.json', json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(DEST) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    for required in ['setup.py', 'multiagent/environment.py', 'wsl-support/vision_mpc/controller_v2.py',
                     'wsl-support/vision_mpc/RunWindows.cmd', 'wsl-support/vision_mpc/run_linux.sh',
                     'wsl-support/team-share/requirements-team.txt', 'README_先读.md']:
        assert 'multiagent-envs-ML/'+required in names, required
    assert not any('.venv/' in n or '__pycache__/' in n or '.egg-info/' in n for n in names)
print(f'PACKAGE: {DEST}')
print(f'FILES: {len(files)+2}; SIZE: {DEST.stat().st_size/1024/1024:.2f} MiB; CRC VERIFIED')
