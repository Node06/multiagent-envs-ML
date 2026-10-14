#!/usr/bin/env bash
set -euo pipefail
team_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$team_root"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -z "${CONDA_PREFIX:-}" && -f "$HOME/.venvs/multiagent-ml/bin/activate" ]]; then
    source "$HOME/.venvs/multiagent-ml/bin/activate"
fi
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Activate Python 3.10 first"'
python wsl-support/vision_mpc/run.py --mode demo --show --video --realtime
