"""M1 output color transform, model resolution, parent re-verification and acceptance record tests."""
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from art_acceptance import deferred_runs
from art_color_finish import verify_parent
from art_destruction_probe import candidates
from art_experiment_color import restore_source_color
from art_experiment_frames import blend, blur, frames, resize, transform, video_signature
from art_experiment_model import recipe
from art_experiment_run import FFMPEG, resolved_model
from art_model_inspect import sha256


def write_video(path, images):
    with av.open(str(path), 'w') as writer:
        stream = writer.add_stream('ffv1', rate=30)
        stream.height, stream.width = images[0].shape[:2]
        stream.pix_fmt, stream.time_base = 'bgr0', Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for pts, image in zip((0, 33, 71), images):
            frame = av.VideoFrame.from_ndarray(image, format='bgr24')
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.random = np.random.default_rng(13)

    def images(self, height, width):
        return [self.random.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(3)]

    def test_output_transform_uses_blurred_upscaled_color_reference_then_retain(self):
        source, inferred = self.images(16, 24), self.images(32, 48)
        write_video(self.root / 'source.mkv', source)
        write_video(self.root / 'inferred.mkv', inferred)
        settings = recipe(dict(source_color=1, luma_change=0.5, input_blur=2, retain=0.25))
        transform(self.root / 'inferred.mkv', self.root / 'result.mkv', settings,
                  original=self.root / 'source.mkv', stage='output')
        for original, damaged, frame in zip(source, inferred, frames(self.root / 'result.mkv'), strict=True):
            colored = restore_source_color(damaged, resize(blur(original, 2), 48, 32), 1, 0.5)
            expected = blend(colored, resize(original, 48, 32), 0.25)
            np.testing.assert_array_equal(frame.to_ndarray(format='bgr24'), expected)

    def test_model_must_resolve_from_working_directory(self):
        folder = self.root / 'models/realesrgan'
        folder.mkdir(parents=True)
        files = [folder / f'realesr-animevideov3-x2{suffix}' for suffix in ('.param', '.bin')]
        for path in files:
            path.write_bytes(path.suffix.encode())
        base = Path('unused/realesr-animevideov3')
        expected = tuple(sha256(path) for path in files)
        self.assertEqual(resolved_model(self.root, base, expected), [str(path) for path in files])
        files[1].write_bytes(b'changed')
        with self.assertRaises(ValueError):
            resolved_model(self.root, base, expected)
        files[1].unlink()
        with self.assertRaises(ValueError):
            resolved_model(self.root, base, expected)

    def test_color_finish_rejects_unpreserved_parent_before_decoding(self):
        with self.assertRaises(ValueError):
            verify_parent(dict(original_hashes_preserved=False), self.root)

    @unittest.skipUnless(FFMPEG.is_file(), 'bundled FFmpeg is required')
    def test_color_finish_redecodes_parent_pixels(self):
        write_video(self.root / 'raw.mkv', self.images(16, 24))
        recorded = [dict(sha256=frame['sha256']) for frame in video_signature(self.root / 'raw.mkv')]
        parent = dict(original_hashes_preserved=True, result=str(self.root / 'raw.mkv'),
                      runs=[dict(pre_encode_frames=recorded)])
        (self.root / 'ok').mkdir()
        self.assertEqual(verify_parent(parent, self.root / 'ok')['decoded_frames'], 3)
        recorded[1]['sha256'] = '0' * 64
        (self.root / 'tampered').mkdir()
        with self.assertRaises(ValueError):
            verify_parent(parent, self.root / 'tampered')

    def test_acceptance_requires_reasons_and_disjoint_decisions(self):
        (self.root / 'suite/a').mkdir(parents=True)
        (self.root / 'suite/a/run.json').write_text('{}', encoding='utf-8')
        (self.root / 'suite/review.json').write_text(json.dumps(dict(candidates=[dict(name='a')])), encoding='utf-8')
        group = dict(directory=str(self.root / 'suite'), reason='too weak', reason_source='creator')
        decision = dict(adopted=[dict(run='build/adopted/run.json')], deferred=[group])
        self.assertEqual([x['reason'] for x in deferred_runs(decision)], ['too weak'])
        for change in (dict(reason=' '), dict(reason_source='guess'), dict(candidates=['missing'])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                deferred_runs(decision | dict(deferred=[group | change]))
        overlap = (self.root / 'suite/a/run.json').as_posix()
        with self.assertRaises(ValueError):
            deferred_runs(decision | dict(adopted=[dict(run=overlap)]))

    def test_weight_free_controls_match_adopted_recipes_except_weights(self):
        table = dict(candidates())
        for name in ('melt-16', 'melt-24'):
            control = dict(table[f'{name}-weights-0'], weight_layers=table[name]['weight_layers'],
                           weight_strength=table[name]['weight_strength'])
            self.assertEqual(recipe(control), recipe(table[name]))
            self.assertEqual(recipe(table[f'{name}-weights-0'])['weight_strength'], 0)


if __name__ == '__main__':
    unittest.main()
