"""Export selected failures and paired examples using the saved command traces."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
args = parser.parse_args()
selections = json.loads((args.results/'representative_replays.json').read_text())
for item in selections:
    subprocess.run([sys.executable, str(Path(__file__).with_name('replay.py')),
                    str(args.results), '--case', item['case'], '--method', item['method']], check=True)
