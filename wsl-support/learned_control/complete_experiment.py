"""Wait for training, freeze models, run independent cases, diagnostics and demo."""
from pathlib import Path
import subprocess
import sys
import time
import json

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE/'outputs/main_20261014'
deadline = time.monotonic()+10800
while not (EXPERIMENT/'training_summary.json').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('Training did not finish within three hours; inspect train.log')
    time.sleep(5)
subprocess.run([sys.executable, str(HERE/'evaluate.py')], check=True)
subprocess.run([sys.executable, str(HERE/'demo.py'), '--output', str(EXPERIMENT/'default_demo')], check=True)
subprocess.run([sys.executable, str(HERE.parent/'course_audit/audit.py'), '--verify-only'], check=True)
(EXPERIMENT/'complete.json').write_text(json.dumps({'evaluation': 'completed', 'default_demo': 'recorded', 'original_integrity': 'passed'}, indent=2))
print('FULL EXPERIMENT COMPLETE', flush=True)
