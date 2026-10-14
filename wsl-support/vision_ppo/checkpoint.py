"""Locate bundled checkpoints after moving a project between operating systems."""
import re
from pathlib import Path


def resolve_checkpoint(selection, directory):
    directory = Path(directory).resolve()
    relative = selection.get('relative_path')
    if relative:
        target = (directory/relative).resolve()
        if not target.is_relative_to(directory):
            raise ValueError('Checkpoint relative_path must stay inside its experiment directory')
        if target.is_file():
            return target
    match = re.fullmatch(r'ppo_seed(\d+)', selection.get('method', ''))
    if match:
        target = directory/f'seed{match[1]}'/'best.pt'
        if target.is_file():
            return target
    original = Path(selection.get('path', ''))
    if original.is_file():
        return original
    raise FileNotFoundError('Bundled checkpoint missing; include seed*/best.pt or pass --checkpoint explicitly')
