"""Record reproducibility metadata; no credentials or unrelated settings."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import argparse

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=root/'outputs'/'final_test')
args = parser.parse_args()
output_dir = args.output_dir.resolve()
relative_output = output_dir.relative_to(root)
files = sorted(list(root.glob('*.py')) + list((root/'tests').glob('*.py')) + [root/'config.json'])
manifest = {
    'python': platform.python_version(),
    'platform': platform.system(),
    'packages': {name: metadata.version(name) for name in ['numpy', 'scipy', 'opencv-python', 'pyglet', 'gymnasium', 'matplotlib']},
    'source_sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    'official_test': str(relative_output/'cases.json'),
    'test_config': str(relative_output/'config.json'),
    'notes': 'Controller parameters frozen before final test. Later changes concern display context, close handling and reporting only.'
}
previous_path = root/'outputs'/'final_test'/'manifest.json'
if previous_path.exists() and output_dir != previous_path.parent:
    previous = json.loads(previous_path.read_text())
    manifest['matches_original_controller'] = manifest['source_sha256']['controller.py'] == previous['source_sha256']['controller.py']
    manifest['matches_original_dynamics'] = manifest['source_sha256']['dynamics.py'] == previous['source_sha256']['dynamics.py']
    manifest['matches_original_config'] = json.loads((output_dir/'config.json').read_text()) == json.loads((previous_path.parent/'config.json').read_text())
    manifest['case_count'] = len(json.loads((output_dir/'cases.json').read_text()))
(output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(output_dir/'manifest.json')
