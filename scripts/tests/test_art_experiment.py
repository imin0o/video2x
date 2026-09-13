"""M1 numeric, serialization, timeline and source-preservation regression tests."""
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from art_experiment_frames import InputNoise, blend, transform, video_signature
from art_experiment import time_suite
from art_experiment_model import MODEL_HASHES, model_copy, recipe
from art_model_inspect import MODEL, inspect_model, sha256


class ModelTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def generate(self, name, values):
        base, hashes = model_copy(self.root / name, recipe(values))
        param, binary = base.parent / 'realesr-animevideov3-x2.param', base.parent / 'realesr-animevideov3-x2.bin'
        return inspect_model(param, binary), binary.read_bytes(), hashes

    def test_zero_is_byte_identical_and_returning_strength_recreates_weights(self):
        _, _, zero = self.generate('zero', {})
        self.assertEqual((zero['param_sha256'], zero['binary_sha256']), MODEL_HASHES)
        _, a, _ = self.generate('a', dict(seed=19, weight_strength=0.5))
        self.generate('b', dict(seed=19, weight_strength=1.0))
        _, again, _ = self.generate('again', dict(seed=19, weight_strength=0.5))
        self.assertEqual(a, again)
        self.assertNotEqual(a, MODEL.with_suffix('.bin').read_bytes())
        self.assertEqual(sha256(MODEL.with_suffix('.bin')), MODEL_HASHES[1])

    def test_only_selected_weight_bytes_change_and_decay_zeroes_weights(self):
        original = MODEL.with_suffix('.bin').read_bytes()
        report, changed, _ = self.generate('decay', dict(weight_mode='decay', weight_strength=1))
        target = next(x for x in report['tensors'] if x['layer'] == 'Conv_16' and x['role'] == 'weight')
        begin, end = target['offset'], target['offset'] + target['bytes']
        self.assertEqual(original[:begin], changed[:begin])
        self.assertEqual(original[end:], changed[end:])
        self.assertEqual(np.count_nonzero(np.frombuffer(changed[begin:end], dtype='<f2')), 0)

    def test_feature_layer_inserted_at_correct_binary_position_and_consumer(self):
        for target in ('PRelu_1', 'PRelu_17', 'PRelu_33'):
            report, data, _ = self.generate(target, dict(feature_layer=target, feature_strength=1,
                                                        feature_mode='mask'))
            self.assertEqual(report['layer_count'], 42)
            self.assertEqual(report['blob_count'], 43)
            self.assertEqual(report['bytes_consumed'], len(data))
            layer = next(i for i, x in enumerate(report['layers']) if x['name'] == target)
            self.assertEqual(report['layers'][layer]['outputs'], ['m1_feature_input'])
            scale = report['layers'][layer + 1]
            self.assertEqual(scale['type'], 'Scale')
            self.assertEqual(report['layers'][layer + 2]['inputs'], scale['outputs'])
            weights = next(x for x in report['tensors'] if x['role'] == 'scale')
            self.assertEqual(weights['maximum'], 0)

    def test_feature_rng_does_not_change_weight_rng(self):
        a, _, _ = self.generate('a', dict(seed=9, weight_strength=0.5))
        b, _, _ = self.generate('b', dict(seed=9, weight_strength=0.5, feature_strength=0.4,
                                        feature_mode='noise'))
        self.assertEqual([x['sha256'] for x in a['tensors']],
                         [x['sha256'] for x in b['tensors'] if x['layer'] != 'm1_feature'])

    def test_rejects_invalid_external_recipe(self):
        for values in [[], {'seed': -1}, {'seed': True}, {'weight_strength': float('nan')},
                       {'input_noise': '12'}, {'passes': 3}, {'time_mode': 'random'},
                       {'version': 2}, {'weight_layers': ['Conv_16', 'Conv_16']},
                       {'weight_layers': [{}]}, {'feature_layer': 'Conv_0'},
                       {'weight_mode': 'decay', 'weight_strength': 1.1}, {'extra': 1}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                recipe(values)


class FrameTests(unittest.TestCase):
    def test_time_suite_rejects_incomplete_cycle_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'uncreated'
            baseline = {'prepared_input': {'probe': {'format': {'duration': '1.0'}}}}
            with self.assertRaises(ValueError):
                time_suite(baseline, output, 0, Path('unused.exe'))
            self.assertFalse(output.exists())

    def test_noise_zero_seed_domains_absolute_time_and_smooth_endpoints(self):
        pixels = np.full((16, 20, 3), 128, dtype=np.uint8)
        noise = InputNoise(pixels.shape, 7)
        self.assertIs(noise.apply(pixels, 0, 0), pixels)
        np.testing.assert_array_equal(noise.apply(pixels, 12, 9),
                                      InputNoise(pixels.shape, 7).apply(pixels, 12, 1))
        np.testing.assert_array_equal(noise.apply(pixels, 12, 0, 'smooth'),
                                      noise.apply(pixels, 12, 0))
        np.testing.assert_array_equal(noise.apply(pixels, 12, 4, 'smooth'),
                                      noise.apply(pixels, 12, 0))
        # Two independently decoded segments evaluated at the same source timestamp.
        np.testing.assert_array_equal(noise.apply(pixels, 12, 2.25, 'smooth'),
                                      InputNoise(pixels.shape, 7).apply(pixels, 12, 2 + 0.25, 'smooth'))
        self.assertFalse(np.array_equal(noise.apply(pixels, 12, 0),
                                         noise.apply(pixels, 12, 2, 'smooth')))

    def test_smooth_constant_keeps_noise_variance_at_the_blend_midpoint(self):
        pixels = np.full((96, 96, 3), 128, dtype=np.uint8)
        noise = InputNoise(pixels.shape, 3)
        spread = {mode: float(np.std(noise.apply(pixels, 10, 1, mode).astype(np.float32) - 128))
                  for mode in ('fixed', 'smooth', 'smooth-constant')}
        self.assertAlmostEqual(spread['smooth-constant'] / spread['fixed'], 1, delta=0.03)
        self.assertAlmostEqual(spread['smooth'] / spread['fixed'], 0.5 ** 0.5, delta=0.03)
        np.testing.assert_array_equal(noise.apply(pixels, 10, 0, 'smooth-constant'), noise.apply(pixels, 10, 0))

    def test_retain_endpoints_and_rounding(self):
        a = np.full((2, 3, 3), 20, np.uint8)
        b = np.full_like(a, 100)
        self.assertIs(blend(a, b, 0), a)
        self.assertIs(blend(a, b, 1), b)
        np.testing.assert_array_equal(blend(a, b, 0.5), np.full_like(a, 60))

    def test_lossless_transform_preserves_vfr_tail_and_zero_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output, mixed = [Path(folder) / name for name in ('in.mkv', 'out.mkv', 'mixed.mkv')]
            with av.open(str(source), 'w') as writer:
                stream = writer.add_stream('ffv1', rate=30)
                stream.width, stream.height, stream.pix_fmt = 32, 16, 'bgr0'
                stream.time_base = Fraction(1, 1000)
                stream.codec_context.time_base = Fraction(1, 1000)
                for i, pts in enumerate((0, 33, 71)):
                    frame = av.VideoFrame.from_ndarray(np.full((16, 32, 3), 40 + i, np.uint8), format='bgr24')
                    frame.pts, frame.time_base = pts, Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        writer.mux(packet)
                for packet in stream.encode():
                    writer.mux(packet)
            before = video_signature(source)
            result = transform(source, output, recipe({}))
            self.assertEqual(before, video_signature(output))
            self.assertEqual(result['frames'], 3)
            self.assertEqual(before[-1]['time'], '71/1000')
            transform(output, mixed, recipe(dict(retain=1)), original=source, stage='output')
            self.assertEqual(video_signature(mixed), before)
            with self.assertRaises(FileExistsError):
                transform(source, output, recipe({}))


if __name__ == '__main__':
    unittest.main()
