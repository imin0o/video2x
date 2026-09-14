"""M2 CLI argument boundaries, replay script scope and child tile environment."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import art_process
from art_probe_support import TILE_VARIABLE, sampled_run
from art_processing import processing_scripts
from art_processing_config import configuration


class CliTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.source = self.root / 'source.json'
        self.source.write_text('{}', encoding='utf-8')

    def main(self, *arguments):
        with patch.object(sys, 'argv', ['art_process.py', *map(str, arguments)]), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            try:
                art_process.main()
            except SystemExit as stop:
                return stop.code, errors.getvalue()
        return 0, errors.getvalue()

    def test_invalid_combinations_are_usage_errors(self):
        output, saved = self.root / 'out', self.root / 'saved.json'
        for arguments in (('--replay', 'run.json', '--output-dir', output, '--seed', '1'),
                          ('--replay', 'run.json', '--output-dir', output, '--baseline-dir', 'm0'),
                          ('--recipe', self.source, '--save-recipe', saved, '--output-dir', output),
                          ('--recipe', self.source, '--save-recipe', saved, '--baseline-dir', 'm0'),
                          ('--recipe', self.source, '--output-dir', output),
                          ('--recipe', self.source, '--baseline-dir', 'm0', '--output-dir', self.root)):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.main(*arguments)[0], 2)
        self.assertFalse(output.exists() or saved.exists())

    def test_validation_errors_do_not_raise_tracebacks(self):
        saved = self.root / 'saved.json'
        code, errors = self.main('--recipe', self.source, '--save-recipe', saved, '--input-noise', '-1')
        self.assertEqual(code, 2)
        self.assertIn('input_noise', errors)
        self.assertFalse(saved.exists())
        self.assertEqual(self.main('--recipe', self.source, '--save-recipe', self.source)[0], 2)
        self.assertEqual(self.main('--replay', self.root / 'missing.json', '--output-dir', self.root / 'o')[0], 2)

    def test_save_recipe_applies_typed_overrides_and_trims_layers(self):
        saved = self.root / 'saved.json'
        self.assertEqual(self.main('--recipe', self.source, '--save-recipe', saved, '--input-noise', '3',
                                   '--seed', '19', '--weight-layers', 'Conv_16, Conv_18')[0], 0)
        config = json.loads(saved.read_text(encoding='utf-8'))
        self.assertEqual(configuration(config), config)
        self.assertEqual((config['settings']['input_noise'], config['settings']['seed']), (3.0, 19))
        self.assertEqual(config['settings']['weight_layers'], ['Conv_16', 'Conv_18'])

    def test_replay_hashes_processing_modules_only(self):
        names = {path.name for path in processing_scripts()}
        self.assertTrue({'art_processing.py', 'art_experiment_run.py', 'art_experiment_frames.py',
                         'art_experiment_color.py', 'art_processing_rng.py'} <= names)
        self.assertFalse({'art_process.py', 'art_processing_verify.py', 'art_probe.py'} & names)

    def test_child_tile_is_set_only_when_requested(self):
        environments = []
        with patch.dict(os.environ, {TILE_VARIABLE: '32'}), patch('art_probe_support.shutil.which', return_value=None), \
                patch('art_probe_support.command', side_effect=lambda *a, **k: environments.append(k['env'])):
            sampled_run(['video2x'], None, self.root / 'a.csv')
            sampled_run(['video2x'], None, self.root / 'b.csv', tile=64)
        self.assertNotIn(TILE_VARIABLE, environments[0])
        self.assertEqual(environments[1][TILE_VARIABLE], '64')


if __name__ == '__main__':
    unittest.main()
