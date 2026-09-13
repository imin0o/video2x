"""Boundary tests for M0 model inspection and reproducibility evidence."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from art_model_inspect import inspect_model
from art_probe_support import parse_diagnostics


class ModelInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.param = self.root / 'test.param'
        self.binary = self.root / 'test.bin'
        self.param.write_text('7767517\n2 2\n'
                              'Convolution conv 1 1 in mid 0=1 1=1 5=1 6=1\n'
                              'PReLU prelu 1 1 mid out 0=1\n')

    def test_fp16_alignment_bias_and_prelu(self):
        original = struct.pack('<Ie2xff', 0x01306B47, 1.25, -0.5, 0.25)
        self.binary.write_bytes(original)
        report = inspect_model(self.param, self.binary)
        self.assertEqual(report['layers'][0]['weight_shape'], [1, 1, 1, 1])
        self.assertEqual([x['offset'] for x in report['tensors']], [4, 8, 12])
        self.assertEqual([x['minimum'] for x in report['tensors']], [1.25, -0.5, 0.25])
        self.assertEqual(report['bytes_consumed'], len(original))
        self.assertEqual(self.binary.read_bytes(), original)

    def test_fp32_zero_tag(self):
        self.binary.write_bytes(struct.pack('<Ifff', 0, 1.25, -0.5, 0.25))
        self.assertEqual(inspect_model(self.param, self.binary)['tensor_encodings'], {'fp32': 3})

    def test_rejects_nonfinite_and_unsupported_encoding(self):
        for data in (struct.pack('<Ifff', 0, float('nan'), 0, 1),
                     struct.pack('<Ifff', 0x000D4B38, 1, 0, 1)):
            with self.subTest(data=data):
                self.binary.write_bytes(data)
                with self.assertRaises(ValueError):
                    inspect_model(self.param, self.binary)

    def test_rejects_truncated_and_trailing_data(self):
        valid = struct.pack('<Ifff', 0, 1, 0, 1)
        for data in (valid[:-1], valid + b'extra'):
            with self.subTest(data=data):
                self.binary.write_bytes(data)
                with self.assertRaises(ValueError):
                    inspect_model(self.param, self.binary)

    def test_rejects_unknown_layer_and_bad_count(self):
        self.binary.write_bytes(struct.pack('<Ifff', 0, 1, 0, 1))
        original = self.param.read_text()
        for text in (original.replace('PReLU', 'Unknown'), original.replace('2 2', '3 2')):
            with self.subTest(text=text):
                self.param.write_text(text)
                with self.assertRaises(ValueError):
                    inspect_model(self.param, self.binary)


class DiagnosticTests(unittest.TestCase):
    def test_requires_pixel_evidence_and_successful_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run.log'
            text = ('[art-baseline] model_load_ms=12.5\n'
                    '[art-baseline] gpu=0 heap_budget_mb=10000 tile=200 scale=2 prepadding=10 tta=false\n'
                    '[art-baseline] pts=0 inference_ms=3.5 result=0\n'
                    '[art-baseline] pts=0 pixel_sha256=' + 'a' * 64 + '\n'
                    '[art-baseline] pts=17 inference_ms=2.5 result=0\n'
                    '[art-baseline] pts=17 pixel_sha256=' + 'b' * 64 + '\n')
            path.write_text(text)
            result = parse_diagnostics(path)
            self.assertEqual(result['reused_frame_mean_ms'], 2.5)
            self.assertEqual(result['pre_encode_frames'][1]['pts'], 17)
            for bad in (text.replace('pixel_sha256', 'unavailable'), text.replace('result=0', 'result=-1')):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    parse_diagnostics(path)


if __name__ == '__main__':
    unittest.main()
