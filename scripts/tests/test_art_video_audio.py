"""Audio offset, compressed-source trimming, multiple tracks and timestamp precision."""
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from art_experiment_run import FFMPEG
from art_probe_support import command
from art_video_output import export_video
from art_video_source import prepare, request
from test_art_video import BASELINE, fixture


@unittest.skipUnless(FFMPEG.is_file(), 'bundled FFmpeg required')
class AudioTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        fixture(self.root / 'video.mkv')

    def source(self, name, offset=0, codec='pcm_f32le', multiple=False, origin=0):
        path = self.root / f'{name}.mkv'
        args = [FFMPEG, '-v', 'error', '-n', '-itsoffset', str(origin), '-i', self.root / 'video.mkv',
                '-itsoffset', str(offset + origin),
                '-f', 'lavfi', '-i', 'sine=frequency=1000:sample_rate=48000:duration=0.3',
                '-map', '0:v:0', '-map', '1:a:0']
        if multiple:
            args += ['-map', '1:a:0']
        command(args + ['-c:v', 'copy', '-c:a', codec, path])
        return path

    def export(self, path, name, start, end):
        spec = request(path, start, end, 1)
        folder = self.root / name
        folder.mkdir()
        context, timeline, tail = prepare(spec, folder, BASELINE)
        return export_video(context['prepared_input']['path'], spec, timeline, tail, folder)

    def test_delayed_audio_preserves_offset_and_interval_with_no_audio(self):
        source = self.source('delayed', .08)
        result = self.export(source, 'overlap', .019, .2)
        self.assertAlmostEqual(float(Fraction(result['audio_verification'][0]['first_time'])), .061, places=3)
        early = self.export(source, 'early', 0, .03)
        self.assertEqual(sum(a['samples'] for a in early['audio_verification']), 0)

    def test_aac_non_sample_aligned_cut_and_multiple_tracks(self):
        for codec in ('aac', 'pcm_f32le'):
            source = self.source(codec, codec=codec, multiple=True)
            result = self.export(source, f'cut-{codec}', .055123, .200432)
            self.assertEqual(len(result['audio_verification']), 2)
            self.assertTrue(all(a['synchronized'] for a in result['audio_verification']))
            with av.open(result['pending']) as reader:
                self.assertEqual(reader.streams.video[0].average_rate, Fraction(30))

    def test_nonzero_container_start_preserves_audio_offset(self):
        source = self.source('shifted', .08, origin=2)
        result = self.export(source, 'shifted-result', .019, .2)
        self.assertAlmostEqual(float(Fraction(result['audio_verification'][0]['first_time'])), .061, places=3)

    def test_pcm_delivery_preserves_the_actual_source_samples(self):
        source = self.source('samples')
        result = self.export(source, 'samples-result', .055, .2)
        with av.open(str(source)) as reader:
            original = np.concatenate([f.to_ndarray().ravel() for f in reader.decode(audio=0)])
        with av.open(result['pending']) as reader:
            actual = np.concatenate([f.to_ndarray().ravel() for f in reader.decode(audio=0)])
        np.testing.assert_array_equal(actual, original[2640:9600])


if __name__ == '__main__':
    unittest.main()
