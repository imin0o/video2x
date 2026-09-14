"""M2 serialization, stream isolation, process cancellation and execution lifecycle."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from art_experiment_model import recipe
from art_processing import load_run, run
from art_processing_config import configuration
from art_processing_rng import DOMAINS, rng
from art_processing_session import Cancelled, Session, checkpoint, run_process


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_recipe_roundtrip_and_m1_migration_are_lossless_and_detached(self):
        legacy = recipe(dict(seed=19, weight_strength=0.6, source_color=1, input_blur=16))
        config = configuration(legacy)
        self.assertEqual(configuration(json.loads(json.dumps(config))), config)
        self.assertEqual(config['settings'], legacy)
        config['settings']['weight_layers'].append('Conv_0')
        self.assertEqual(legacy['weight_layers'], ['Conv_16'])
        legacy['weight_layers'].clear()
        self.assertEqual(recipe({})['weight_layers'], ['Conv_16'])

    def test_unknown_versions_models_rng_and_invalid_values_are_rejected(self):
        config = configuration({})
        for update in ({'schema_version': True}, {'schema_version': 3}, {'method': 'other'},
                       {'rng': 'default_rng'}, {'model': {'id': 'other'}}, {'extra': 1},
                       {'settings': {'input_noise': -1}}, {'settings': {'weight_strength': float('inf')}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                configuration(config | update)

    def test_all_domains_are_independent_and_preserve_m1_stream_bytes(self):
        for domain in DOMAINS:
            purpose = domain + ':0'
            digest = hashlib.sha256(f'm1-pcg64-v1:19:{purpose}'.encode()).digest()
            expected = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], 'little'))).random(10)
            for other in DOMAINS:
                rng(19, other + ':0').random(1000)
            np.testing.assert_array_equal(expected, rng(19, purpose).random(10))
        self.assertEqual(len({rng(19, d + ':0').bytes(16) for d in DOMAINS}), 5)

    def test_streams_match_in_a_new_python_process(self):
        code = 'from art_processing_rng import rng; print(rng(19, \'weight:Conv_16\').bytes(32).hex())'
        actual = subprocess.check_output([sys.executable, '-B', '-c', code],
                                         cwd=Path(__file__).resolve().parents[1], text=True).strip()
        self.assertEqual(actual, rng(19, 'weight:Conv_16').bytes(32).hex())

    def test_cancel_waits_for_child_and_releases_context(self):
        session, processes = Session(), []
        original = subprocess.Popen

        def capture(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            session.cancel()
            return process

        with patch('art_processing_session.subprocess.Popen', side_effect=capture):
            with self.assertRaises(Cancelled), session:
                run_process([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=subprocess.PIPE)
        self.assertIsNotNone(processes[0].poll())
        self.assertFalse(session.active)
        with Session():
            checkpoint()

    def test_same_session_cannot_be_reentered_and_other_thread_cannot_take_it(self):
        session, failures = Session(), []

        def enter():
            try:
                with session:
                    pass
            except RuntimeError:
                failures.append(True)

        with session:
            worker = threading.Thread(target=enter)
            worker.start()
            worker.join()
            with self.assertRaises(RuntimeError):
                with Session():
                    pass
        self.assertEqual(failures, [True])

    def test_failure_and_cancel_are_recorded_and_next_run_can_start(self):
        baseline = dict(runs=[dict(effective=dict(tile=200, scale=2, prepadding=10, tta=False))])
        for index, error in enumerate((RuntimeError('inference failure'), Cancelled('cancelled'))):
            session, folder = Session(), self.root / str(index)
            with patch('art_processing.validate_baseline'), patch('art_processing.environment', return_value={}):
                with patch('art_processing.execute', side_effect=error), self.assertRaises(type(error)):
                    run(baseline, folder, {}, self.root / 'cli.exe', session=session)
            record = json.loads((folder / 'run.json').read_text())
            self.assertEqual(record['status'], 'cancelled' if index else 'failed')
            self.assertIsNone(record['result'])
            self.assertFalse(session.active)
            with self.assertRaises(ValueError):
                load_run(folder / 'run.json')

    def test_replay_environment_mismatch_fails_before_creating_output(self):
        baseline = dict(runs=[dict(effective=dict(tile=200, scale=2, prepadding=10, tta=False))])
        saved = dict(schema_version=2, status='completed', configuration=configuration({}),
                     baseline=copy.deepcopy(baseline), environment={'changed': True})
        with patch('art_processing.validate_baseline'), patch('art_processing.environment', return_value={}):
            with self.assertRaises(ValueError):
                run(baseline, self.root / 'missing', {}, 'cli.exe', replay=saved)
        self.assertFalse((self.root / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
