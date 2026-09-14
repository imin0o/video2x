"""M3 interval, modulation, delivery and failure contracts on real FFV1/VFR media."""
import copy
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from art_experiment_frames import InputNoise, transform, video_signature
from art_experiment_run import FFMPEG
from art_probe_support import command
from art_processing import run
from art_processing_config import configuration, recipe
from art_processing_session import Cancelled, Session
from art_video_output import export_video
from art_video_run import remove_intermediates
from art_video_source import prepare, request

BASELINE = dict(runs=[dict(effective=dict(tile=200, scale=2, prepadding=10, tta=False))])


def fixture(path, stamps=(0, 33, 71, 119, 153, 199), offset=0, rate=30):
    with av.open(str(path), 'w') as writer:
        stream = writer.add_stream('ffv1', rate=rate)
        stream.width, stream.height, stream.pix_fmt = 64, 32, 'bgr0'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        random = np.random.default_rng(5)
        for pts in stamps:
            frame = av.VideoFrame.from_ndarray(random.integers(0, 256, (32, 64, 3), dtype=np.uint8), format='bgr24')
            frame.pts, frame.time_base = pts + offset, Fraction(1, 1000)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)


class VideoTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source.mkv'
        fixture(self.source)

    def segment(self, name, start=0, end=None):
        folder = self.root / name
        folder.mkdir()
        spec = request(self.source, start, end)
        context, timeline, tail = prepare(spec, folder, BASELINE)
        return folder, spec, context, timeline, tail

    def test_partial_modulation_matches_full_at_non_frame_boundary(self):
        full = self.segment('full')
        part = self.segment('part', .055, .154)
        settings = recipe(dict(input_noise=17, time_mode='smooth', period=.3, phase=.17, modulation_depth=.6))
        outputs = []
        for folder, spec, context, timeline, tail in (full, part):
            transform(context['prepared_input']['path'], folder / 'noise.mkv', settings,
                      source_times=context['prepared_input']['source_times'])
            outputs.append({f['time']: s['sha256'] for f, s in zip(timeline, video_signature(folder / 'noise.mkv'), strict=True)})
        self.assertEqual(outputs[1], {k: v for k, v in outputs[0].items() if k in outputs[1]})
        self.assertEqual(list(outputs[1]), ['33/1000', '71/1000', '119/1000', '153/1000'])
        self.assertEqual((part[3][0]['display_start'], part[3][0]['display_end']), ('11/200', '71/1000'))

    def test_adjacent_intervals_partition_full_display_time(self):
        full = self.segment('whole', 0, .2)[3]
        parts = self.segment('first', 0, .1)[3] + self.segment('second', .1, .2)[3]
        merged = []
        for frame in parts:
            if merged and merged[-1]['time'] == frame['time'] and merged[-1]['display_end'] == frame['display_start']:
                merged[-1]['display_end'] = frame['display_end']
            else:
                merged.append(dict(frame))
        self.assertEqual(merged, full)
        self.assertEqual([f['display_start'] for f in parts[:4]], ['0', '33/1000', '71/1000', '1/10'])

    def test_sub_frame_interval_keeps_source_time_for_modulation(self):
        _, _, context, timeline, tail = self.segment('short', .04, .06)
        self.assertEqual([(f['time'], f['display_start'], f['display_end']) for f in timeline],
                         [('33/1000', '1/25', '3/50')])
        self.assertEqual((context['prepared_input']['source_times'], tail), (['33/1000'], Fraction(3, 50)))

    def test_depth_zero_is_fixed_and_legacy_depth_is_preserved(self):
        pixels = np.full((32, 64, 3), 128, dtype=np.uint8)
        noise = InputNoise(pixels.shape, 19)
        np.testing.assert_array_equal(noise.apply(pixels, 17, .125, 'smooth', depth=0),
                                      noise.apply(pixels, 17, 0, 'fixed'))
        np.testing.assert_array_equal(noise.apply(pixels, 17, .125, 'smooth'),
                                      noise.apply(pixels, 17, .125, 'smooth', depth=1))
        for values in ({'modulation_depth': -1}, {'modulation_depth': float('nan')}, {'modulation_target': 'weight'}):
            self.assertRaises(ValueError, configuration, values)

    def test_interval_preserves_vfr_hold_until_the_requested_end(self):
        fixture(self.root / 'held.mkv', stamps=(0, 200, 450))
        self.source = self.root / 'held.mkv'
        _, _, _, timeline, tail = self.segment('hold', 0, .15)
        self.assertEqual(tail, Fraction(15, 100))
        self.assertEqual(timeline[0]['display_end'], '3/20')

    def test_nonzero_container_origin_is_subtracted_once(self):
        fixture(self.root / 'offset.mkv', offset=2000)
        self.source = self.root / 'offset.mkv'
        _, spec, _, timeline, _ = self.segment('offset', .055, .154)
        self.assertEqual(spec['info']['origin'], '2')
        self.assertEqual([f['time'] for f in timeline], ['33/1000', '71/1000', '119/1000', '153/1000'])

    @unittest.skipUnless(FFMPEG.is_file(), 'bundled FFmpeg required')
    def test_equal_size_retain_one_recovers_exact_original_pixels(self):
        folder, spec, context, timeline, tail = self.segment('retain', 0, .2)
        clip = context['prepared_input']['path']
        damaged = folder / 'damaged.mkv'
        transform(clip, damaged, recipe(dict(input_noise=30)), output_size=(128, 64))
        spec['output_scale'] = 1
        result = export_video(damaged, spec, timeline, tail, folder, settings=recipe(dict(retain=1)), original=clip)
        self.assertEqual([f['sha256'] for f in result['pre_encode_frames']],
                         [f['sha256'] for f in video_signature(clip)])

    @unittest.skipUnless(FFMPEG.is_file(), 'bundled FFmpeg required')
    def test_sixty_fps_default_duration_has_at_most_one_ms_rounding(self):
        fixture(self.root / 'sixty.mkv', stamps=(0, 17, 33, 50, 67), rate=60)
        self.source = self.root / 'sixty.mkv'
        folder, spec, context, timeline, tail = self.segment('sixty', 0, .084)
        result = export_video(context['prepared_input']['path'], spec, timeline, tail, folder)
        error = Fraction(result['decoded_video_end_seconds']) - Fraction(result['video_end_seconds'])
        self.assertLessEqual(abs(error), Fraction(1, 1000))

    def test_invalid_interval_rejected(self):
        for start, end in ((-1, 1), (1, 0), (0, float('nan')), (float('inf'), None), (2, None)):
            self.assertRaises(ValueError, request, self.source, start, end)

    @unittest.skipUnless(FFMPEG.is_file(), 'bundled FFmpeg required')
    def test_vfr_one_frame_tail_dimensions_pixels_and_audio_delivery(self):
        audio = self.root / 'audio.mkv'
        command([FFMPEG, '-v', 'error', '-n', '-i', self.source, '-f', 'lavfi', '-i',
                 'sine=frequency=1000:sample_rate=48000:duration=0.3', '-map', '0:v', '-map', '1:a',
                 '-c:v', 'copy', '-c:a', 'pcm_f32le', audio])
        self.source = audio
        for name, start, end in (('vfr', .055, .2), ('tail', .19, .205)):
            folder, spec, context, timeline, tail = self.segment(name, start, end)
            spec['output_scale'] = 1
            delivery = export_video(context['prepared_input']['path'], spec, timeline, tail, folder)
            actual = video_signature(delivery['pending'])
            self.assertEqual(len(actual), len(timeline))
            self.assertEqual((actual[0]['width'], actual[0]['height']), (64, 32))
            self.assertEqual([f['time'] for f in actual],
                             [str(Fraction(f['display_start']) - Fraction(str(start))) for f in timeline])
            self.assertEqual(actual[0]['time'], '0')  # The straddling frame starts at the requested start.
            self.assertEqual(delivery['audio_streams'][0]['codec'], 'pcm_f32le')
            with av.open(delivery['pending']) as reader:
                decoded = list(reader.decode(audio=0))
            self.assertTrue(decoded)
            self.assertLessEqual(abs(float(decoded[0].pts * decoded[0].time_base)), 1 / 1000)
            samples = sum(f.samples for f in decoded)
            self.assertLessEqual(abs(samples / 48000 - float(tail - Fraction(str(start)))), 1 / 48000)
        with self.assertRaisesRegex(ValueError, 'no video frame'):  # Audio-only time is not filled with video.
            self.segment('empty', .25, .3)

    def test_m3_cancellation_and_failure_leave_records_and_release_session(self):
        for index, error in enumerate((Cancelled('stop'), MemoryError('simulated GPU allocation failure'))):
            folder, events = self.root / str(index), []
            session = Session(on_progress=events.append)
            with patch('art_video_run.validate_baseline'), patch('art_processing.environment', return_value={}), \
                    patch('art_video_run.prepare', side_effect=error), self.assertRaises(type(error)):
                run(copy.deepcopy(BASELINE), folder, {}, 'cli.exe', video=request(self.source), session=session)
            record = json.loads((folder / 'run.json').read_text())
            self.assertEqual(record['status'], 'failed' if index else 'cancelled')
            self.assertIsNone(record['result'])
            self.assertTrue(record['source_preserved'])
            self.assertFalse((folder / 'result.mkv').exists() or session.active)
            self.assertEqual([(e['stage'], e['state']) for e in events], [('prepare', 'running'), ('prepare', record['status'])])
            self.assertEqual((events[-1]['run_id'], events[-1]['pass_count']), (record['run_id'], 1))
            self.assertRaises(RuntimeError, session.__enter__)

    def test_success_removes_only_generated_intermediate_videos(self):
        folder = self.root / 'done'
        for name in ('work/pass-1.mkv', 'work/injected.mkv', 'video.partial.mkv', 'input.mkv', 'work/run.json'):
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(b'x')
        outcome = remove_intermediates(folder)
        self.assertEqual(sorted(Path(r['path']).name for r in outcome['removed']),
                         ['injected.mkv', 'pass-1.mkv', 'video.partial.mkv'])
        self.assertTrue((folder / 'input.mkv').is_file() and (folder / 'work/run.json').is_file())
        self.assertEqual(outcome['failures'], {})
        with self.assertRaisesRegex(ValueError, 'M3 video runs'):
            run(copy.deepcopy(BASELINE), self.root / 'm2', {}, 'cli.exe', keep_intermediates=True)
        self.assertFalse((self.root / 'm2').exists())


if __name__ == '__main__':
    unittest.main()
