import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('checkpoint', Path(__file__).resolve().parents[1]/'checkpoint.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CheckpointTests(unittest.TestCase):
    def test_moved_project_uses_bundled_model_before_old_absolute_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root/'seed11/best.pt'
            target.parent.mkdir(); target.write_bytes(b'local model')
            selection = {'method': 'ppo_seed11', 'path': '/mnt/e/author-only/seed11/best.pt'}
            self.assertEqual(module.resolve_checkpoint(selection, root), target.resolve())

    def test_explicit_relative_model_and_escape_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root/'model.pt'; target.write_bytes(b'model')
            self.assertEqual(module.resolve_checkpoint({'relative_path': 'model.pt'}, root), target.resolve())
            with self.assertRaises(ValueError):
                module.resolve_checkpoint({'relative_path': '../outside.pt'}, root)

    def test_missing_model_reports_a_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                module.resolve_checkpoint({'method': 'ppo_seed11'}, temporary)
