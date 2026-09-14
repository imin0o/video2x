"""Qt integration tests exercise controls, thread delivery and paired VFR playback."""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from art_gui import ArtWindow
from art_gui_player import ComparisonPlayer, comparison_frames
from art_gui_settings import document, load_document, save_document
from art_gui_worker import RenderController
from art_processing_config import configuration
from art_processing_session import Cancelled

APP = QApplication.instance() or QApplication([])


def pump(until, seconds=5):
    deadline = time.monotonic() + seconds
    while not until():
        APP.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError('Qt wait timed out')
        time.sleep(.005)
    APP.processEvents()


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.window = ArtWindow(argparse.Namespace(input=None, recipe=None, baseline_dir=self.root, cli=self.root/'cli.exe'))

    def tearDown(self):
        self.window.close()
        APP.processEvents()
        self.temp.cleanup()

    def test_recipe_roundtrip_seed_precision_and_reset(self):
        config = configuration(dict(seed=2**64-1, weight_strength=1.23456789123456, input_noise=6,
                                    time_mode='smooth', retain=.3, passes=2))
        self.window.form.set_config(config)
        self.assertEqual(config, self.window.form.config())
        self.window.paths.widgets['end'].setText('')
        saved = self.root / 'settings.json'
        save_document(saved, config, self.window.paths.values())
        loaded, controls = load_document(saved)
        self.assertEqual(loaded, config)
        self.assertEqual(controls['end'], '')
        relative = self.window.paths.values() | dict(source='test/test.mp4', output_root='build/art/gui')
        portable = document(config, relative)['controls']
        self.assertTrue(Path(portable['source']).is_absolute())
        self.assertTrue(Path(portable['output_root']).is_absolute())
        with self.assertRaises(FileExistsError):
            save_document(saved, config, controls)
        self.window.form.widgets['input_noise'].setValue(12)
        self.assertEqual(self.window.form.config()['settings']['seed'], 2**64-1)
        self.window.form.reset_effects()
        result = self.window.form.config()['settings']
        self.assertEqual(result['passes'], 1)
        self.assertTrue(all(result[k] == 0 for k in ('weight_strength', 'feature_strength', 'input_noise',
                                                   'input_blur', 'output_blur', 'retain', 'source_color', 'luma_change')))

    def test_unsupported_and_invalid_documents_are_atomic(self):
        value = self.window.snapshot()
        value['configuration']['model']['id'] = 'unsupported'
        path = self.root / 'invalid.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        before = self.window.snapshot()
        with patch.object(self.window, 'show_error') as error:
            self.window.load_settings(path)
            error.assert_called_once()
        self.assertEqual(before, self.window.snapshot())
        values = self.window.paths.values() | dict(start='nan')
        with self.assertRaises(ValueError):
            document(configuration({}), values)

    def test_changed_settings_do_not_relabel_existing_result(self):
        self.window.result_snapshot = self.window.snapshot()
        self.window.paths.widgets['output_root'].setText(str(self.root / 'elsewhere'))
        self.assertIn('一致', self.window.dirty.text())
        self.window.form.widgets['input_noise'].setValue(3)
        self.assertIn('未反映', self.window.dirty.text())
        self.assertEqual(self.window.result_snapshot['configuration']['settings']['input_noise'], 0)

    def test_wheel_over_unfocused_controls_does_not_change_values(self):
        before = self.window.snapshot()
        for widget in (self.window.form.widgets['input_noise'], self.window.form.widgets['feature_mode'],
                       self.window.paths.widgets['audio']):
            point = QPointF(widget.rect().center())
            APP.sendEvent(widget, QWheelEvent(point, widget.mapToGlobal(point), QPoint(), QPoint(0, 120),
                                              Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
        self.assertEqual(before, self.window.snapshot())

    def test_progress_and_outcome_update_window(self):
        window, done = self.window, str(self.root / 'done')
        window.on_progress(dict(stage='inference', completed=3, total=10, total_exact=True, elapsed_seconds=1.0))
        self.assertEqual((window.progress.maximum(), window.progress.value()), (10, 3))
        window.on_progress(dict(stage='inference', completed=12, total=10, total_exact=False, elapsed_seconds=2.0))
        self.assertEqual(window.progress.maximum(), 0)
        window.on_progress(dict(stage='verify-audio', completed=None, total=None, total_exact=None, elapsed_seconds=3.0))
        with patch.object(window.player, 'load') as load:
            window.on_outcome(dict(state='completed', directory=done, record={'result': done + '/result.mkv'},
                                   snapshot=window.snapshot()))
        load.assert_called_once()
        window.on_outcome(dict(state='failed', directory=str(self.root / 'broken'), error='ValueError: bad'))
        self.assertIn('bad', window.status.text())
        # The folder button and comparison stay on the displayed success.
        self.assertEqual(window.last_directory, done)
        self.assertTrue(window.buttons['folder'].isEnabled())
        self.assertIn('一致', window.dirty.text())

    def test_cancel_race_stale_notification_and_new_request(self):
        controller = RenderController()
        outcomes, threads = [], []
        controller.outcome.connect(lambda value: (outcomes.append(value), threads.append(threading.get_ident())))
        started = threading.Event()

        def work(*args, session, **kwargs):
            started.set()
            while not session.cancelled.wait(.01):
                pass
            raise Cancelled('cancel test')

        with patch('art_gui_worker.request', return_value={}), patch('art_gui_worker.baseline_context', return_value={}):
            with patch('art_gui_worker.run', side_effect=work):
                controller.start(self.window.snapshot(), self.root/'cancel')
                pump(started.is_set)
                old_token = controller.token
                controller.cancel()
                pump(lambda: controller.worker is None)
                self.assertEqual(outcomes[-1]['state'], 'cancelled')
            with patch('art_gui_worker.run', return_value={'result': 'new'}):
                controller.start(self.window.snapshot(), self.root/'next')
                controller.accept_outcome(old_token, {'state': 'completed', 'directory': 'old'})
                pump(lambda: controller.worker is None)
        self.assertEqual([o['state'] for o in outcomes], ['cancelled', 'completed'])
        self.assertEqual(set(threads), {threading.get_ident()})

    def test_late_cancel_reports_saved_result_and_keeps_failure_reason(self):
        controller, outcomes = RenderController(), []
        controller.outcome.connect(outcomes.append)
        started, release = threading.Event(), threading.Event()

        def finish(result):
            def work(*args, session, **kwargs):
                started.set()
                release.wait(5)
                if isinstance(result, Exception):
                    raise result
                return result
            return work

        with patch('art_gui_worker.request', return_value={}), patch('art_gui_worker.baseline_context', return_value={}):
            for result in ({'result': 'saved.mkv'}, ValueError('disk full')):
                started.clear()
                release.clear()
                with patch('art_gui_worker.run', side_effect=finish(result)):
                    controller.start(self.window.snapshot(), self.root/'late')
                    pump(started.is_set)
                    controller.cancel()  # After the last checkpoint: the work still ends on its own.
                    release.set()
                    pump(lambda: controller.worker is None)
        self.assertEqual([o['state'] for o in outcomes], ['cancelled', 'failed'])
        self.assertTrue('保存済み' in outcomes[0]['error'] and 'saved.mkv' in outcomes[0]['error'])
        self.assertIn('disk full', outcomes[1]['error'])

    def test_failure_then_success_and_close_waits(self):
        controller = self.window.controller
        outcomes = []
        controller.outcome.disconnect(self.window.on_outcome)
        controller.outcome.connect(outcomes.append)
        with patch('art_gui_worker.request', side_effect=ValueError('bad source')):
            controller.start(self.window.snapshot(), self.root/'failure')
            pump(lambda: controller.worker is None)
        self.assertEqual(outcomes[0]['state'], 'failed')
        self.assertIn('bad source', outcomes[0]['error'])
        with patch('art_gui_worker.request', return_value={}), patch('art_gui_worker.baseline_context', return_value={}):
            with patch('art_gui_worker.run', return_value={'result': 'ok'}):
                controller.start(self.window.snapshot(), self.root/'success')
                pump(lambda: controller.worker is None)
            self.assertEqual(outcomes[1]['state'], 'completed')
            with patch('art_gui_worker.run', side_effect=lambda *a, session, **k: session.cancelled.wait(2)):
                controller.start(self.window.snapshot(), self.root/'close')
                self.window.close()
                self.assertTrue(self.window.closing)
                pump(lambda: controller.worker is None)
        self.assertEqual(outcomes[-1]['state'], 'cancelled')

    def test_comparison_pairs_vfr_frames_and_source_times(self):
        for name, multiplier in [('input.mkv', 1), ('result.mkv', 2)]:
            with av.open(str(self.root/name), 'w') as output:
                stream = output.add_stream('ffv1', rate=10)
                stream.width = stream.height = 16 * multiplier
                stream.pix_fmt = 'bgr0'
                stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
                for i, stamp in enumerate([0, 40, 160]):
                    frame = av.VideoFrame.from_ndarray(np.full((16*multiplier, 16*multiplier, 3), i*50, np.uint8), format='rgb24')
                    frame.pts, frame.time_base = stamp, Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
        record = dict(result=str(self.root/'result.mkv'), source_timeline=[
            dict(display_start=a, display_end=b) for a, b in [('2.017', '2.04'), ('2.04', '2.16'), ('2.16', '2.2')]])
        frames = list(comparison_frames(record, threading.Event()))
        self.assertEqual([f[1] for f in frames], [2.017, 2.04, 2.16])
        for i, (pictures, _, _) in enumerate(frames):
            self.assertEqual([p.pixelColor(0, 0).red() for p in pictures], [i*50, i*50])
        self.window.player.load(record)
        pump(lambda: self.window.player.position is not None)
        self.window.player.toggle()
        pump(lambda: not self.window.player.play_button.isEnabled())
        self.assertIn('終了', self.window.player.stamp.text())
        player, sizes = ComparisonPlayer(), []
        try:
            player.resize(600, 400)
            player.show()
            player.load(record)
            pump(lambda: player.position is not None)
            for width, height in [(900, 600), (450, 320)]:  # Paused: only a resize redraws the frame.
                player.resize(width, height)
                APP.processEvents()
                label, pixmap = player.labels[1], player.labels[1].pixmap()
                sizes.append(pixmap.width())
                self.assertTrue(pixmap.width() <= label.width() and pixmap.height() <= label.height())
            self.assertGreater(sizes[0], sizes[1])
        finally:
            player.shutdown()


if __name__ == '__main__':
    unittest.main()
