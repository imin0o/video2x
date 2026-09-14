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
from art_experiment_frames import InputNoise
from art_experiment_model import model_copy, recipe
from art_model_inspect import MODEL, inspect_model
from art_processing import completed_run, load_run, run
from art_processing_config import configuration
from art_processing_rng import DOMAINS, manifest, rng
from art_processing_session import Cancelled, Session, checkpoint, is_cancellation, run_process

BASELINE = dict(runs=[dict(effective=dict(tile=200, scale=2, prepadding=10, tta=False))])
HAS_MODEL = MODEL.with_suffix('.param').is_file() and MODEL.with_suffix('.bin').is_file()


def saved_run(**changes):
    return dict(schema_version=2, status='completed', run_id='parent', configuration=configuration({}),
                baseline=copy.deepcopy(BASELINE), environment={}, final_frames=[],
                engine=dict(runs=[dict(pre_encode_frames=[])])) | changes


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

    def test_all_domains_preserve_m1_stream_bytes(self):
        for domain in DOMAINS:
            purpose = domain + ':0'
            digest = hashlib.sha256(f'm1-pcg64-v1:19:{purpose}'.encode()).digest()
            expected = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], 'little'))).random(10)
            np.testing.assert_array_equal(expected, rng(19, purpose).random(10))
        self.assertEqual(len({rng(19, d + ':0').bytes(16) for d in DOMAINS}), 5)

    @unittest.skipUnless(HAS_MODEL, 'original model is not available')
    def test_model_draws_do_not_depend_on_other_selected_layers(self):
        common = dict(seed=19, weight_strength=0.5, feature_mode='noise', feature_strength=0.5)
        _, first = model_copy(self.root / 'a', recipe(common | dict(weight_layers=['Conv_16'])))
        _, second = model_copy(self.root / 'b', recipe(common | dict(weight_layers=['Conv_18', 'Conv_16'])))
        a, b = (np.fromfile(self.root / name / 'realesr-animevideov3-x2.bin', np.uint8) for name in 'ab')
        tensor = next(t for t in inspect_model(MODEL.with_suffix('.param'), MODEL.with_suffix('.bin'))['tensors']
                      if t['layer'] == 'Conv_18' and t['role'] == 'weight')
        start = tensor['offset'] + 64 * 4 * 2  # Feature scales and biases are inserted before Conv_18.
        changed = np.flatnonzero(a != b)
        self.assertTrue(changed.size)
        self.assertTrue(((changed >= start) & (changed < start + tensor['bytes'])).all())
        self.assertEqual(first['feature'], second['feature'])
        self.assertEqual(len(first['feature']['biases']), 64)

    @unittest.skipUnless(HAS_MODEL, 'original model is not available')
    def test_rng_manifest_lists_exactly_the_consumed_streams(self):
        settings = recipe(dict(seed=19, weight_strength=0.5, weight_layers=['Conv_16', 'Conv_18'],
                               feature_mode='mask', feature_strength=0.5, input_noise=3, time_mode='smooth'))
        used = []

        def recording(seed, purpose):
            used.append(purpose)
            return rng(seed, purpose)

        with patch('art_experiment_model.rng', side_effect=recording), \
                patch('art_experiment_frames.rng', side_effect=recording):
            model_copy(self.root / 'model', settings)
            InputNoise((2, 2, 3), settings['seed'])
        streams = manifest(settings)
        self.assertEqual(set(used), set(streams['streams']) - set(streams['reserved']))

    def test_streams_match_in_a_new_python_process(self):
        code = 'from art_processing_rng import rng; print(rng(19, \'weight:Conv_16\').bytes(32).hex())'
        actual = subprocess.check_output([sys.executable, '-B', '-c', code],
                                         cwd=Path(__file__).resolve().parents[1], text=True).strip()
        self.assertEqual(actual, rng(19, 'weight:Conv_16').bytes(32).hex())

    def test_cancel_waits_for_child_and_releases_context(self):
        session, processes, flags = Session(), [], []
        original = subprocess.Popen

        def capture(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            flags.append(kwargs.get('creationflags'))
            session.cancel()
            return process

        with patch('art_processing_session.subprocess.Popen', side_effect=capture):
            with self.assertRaises(Cancelled), session:
                run_process([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=subprocess.PIPE)
        self.assertIsNotNone(processes[0].poll())
        # A pythonw GUI must not open a console window for each child process.
        self.assertEqual(flags, [subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0])
        self.assertFalse(session.active)
        self.assertFalse(issubclass(Cancelled, KeyboardInterrupt))
        self.assertTrue(is_cancellation(Cancelled()) and is_cancellation(KeyboardInterrupt()))
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
        for index, error in enumerate((RuntimeError('inference failure'), Cancelled('cancelled'))):
            session, folder = Session(), self.root / str(index)
            with patch('art_processing.validate_baseline'), patch('art_processing.environment', return_value={}):
                with patch('art_processing.execute', side_effect=error), self.assertRaises(type(error)):
                    run(BASELINE, folder, {}, self.root / 'cli.exe', session=session)
            record = json.loads((folder / 'run.json').read_text())
            self.assertEqual(record['status'], 'cancelled' if index else 'failed')
            self.assertIsNone(record['result'])
            self.assertFalse(session.active)
            with self.assertRaises(ValueError):
                load_run(folder / 'run.json')

    def test_replay_environment_mismatch_fails_before_creating_output(self):
        completed_run(saved_run())
        with patch('art_processing.validate_baseline'), patch('art_processing.environment', return_value={}):
            with self.assertRaisesRegex(ValueError, 'environment differs'):
                run(BASELINE, self.root / 'missing', {}, 'cli.exe', replay=saved_run(environment={'changed': True}))
        self.assertFalse((self.root / 'missing').exists())

    def test_incomplete_replay_records_are_rejected_before_processing(self):
        for changes in (dict(final_frames=None), dict(engine=dict(runs=[])), dict(engine=None),
                        dict(engine=dict(runs=[{}])), dict(status='failed')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                completed_run(saved_run(**changes))
        self.assertRaises(ValueError, completed_run, {k: v for k, v in saved_run().items() if k != 'baseline'})

    def test_replay_frame_mismatch_is_recorded_as_failed(self):
        video = self.root / 'result.mkv'
        video.write_bytes(b'video')
        engine = dict(result=str(video), model={}, baseline=BASELINE,
                      runs=[dict(model_files=[], pre_encode_frames=[])])
        with patch('art_processing.validate_baseline'), patch('art_processing.environment', return_value={}), \
                patch('art_processing.execute', return_value=engine), \
                patch('art_processing.video_signature', return_value=['new']):
            with self.assertRaisesRegex(ValueError, 'A04'):
                run(BASELINE, self.root / 'replay', {}, 'cli.exe', replay=saved_run(final_frames=['old']))
        record = json.loads((self.root / 'replay/run.json').read_text())
        self.assertEqual((record['status'], record['replay']['exact_frames_match']), ('failed', False))
        self.assertNotIn('baseline', record['engine'])


if __name__ == '__main__':
    unittest.main()
