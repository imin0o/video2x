"""Source color, black-bar and zero-strength invariants for destructive M1 output."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from art_experiment_color import BAND_ROWS, restore_source_color
from art_experiment_model import recipe


def reference_restore(pixels, reference, amount, luma_change):
    """The original unbanded formula that produced the accepted melt-16/melt-24 outputs."""
    candidate, source = pixels.astype(np.float32), reference.astype(np.float32)
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    damaged_y, source_y = np.sum(candidate * weights, axis=-1), np.sum(source * weights, axis=-1)
    deviation = float(damaged_y.std())
    if deviation > 1e-6:
        mapped_y = (damaged_y - damaged_y.mean()) * (float(source_y.std()) / deviation) + source_y.mean()
    else:
        mapped_y = np.full_like(source_y, source_y.mean())
    protection = np.clip(source_y / 16, 0, 1)
    luminance = np.clip(source_y + luma_change * protection * (mapped_y - source_y), 0, 255)
    chroma = source - source_y[..., None]
    positive = np.maximum(chroma.max(axis=-1), 1e-6)
    negative = np.maximum(-chroma.min(axis=-1), 1e-6)
    gamut_scale = np.minimum(1, np.minimum((255 - luminance) / positive, luminance / negative))
    restored = luminance[..., None] + gamut_scale[..., None] * chroma
    return np.clip(np.rint((1 - amount) * candidate + amount * restored), 0, 255).astype(np.uint8)


class ColorTests(unittest.TestCase):
    def test_banded_implementation_is_bit_identical_to_reference_formula(self):
        random = np.random.default_rng(47)
        for height in (1, BAND_ROWS - 1, BAND_ROWS * 2 + 3):
            for amount, luma_change in ((1, 0.5), (1.0, 0.7), (0.35, 1), (1, 0)):
                with self.subTest(height=height, amount=amount, luma_change=luma_change):
                    source = random.integers(0, 256, (height, 23, 3), dtype=np.uint8)
                    damaged = random.integers(0, 256, source.shape, dtype=np.uint8)
                    for candidate in (damaged, np.full_like(damaged, 90)):
                        np.testing.assert_array_equal(restore_source_color(candidate, source, amount, luma_change),
                                                      reference_restore(candidate, source, amount, luma_change))

    def test_zero_color_amount_is_exact_identity(self):
        pixels = np.array([[[0, 255, 0], [255, 0, 255]]], dtype=np.uint8)
        self.assertIs(restore_source_color(pixels, np.zeros_like(pixels), 0, 1), pixels)

    def test_zero_luma_change_recovers_source_including_gamut_boundaries(self):
        random = np.random.default_rng(31)
        source = random.integers(0, 256, (12, 17, 3), dtype=np.uint8)
        damaged = random.integers(0, 256, source.shape, dtype=np.uint8)
        np.testing.assert_array_equal(restore_source_color(damaged, source, 1, 0), source)

    def test_source_chroma_direction_and_black_bars_survive_neon_input(self):
        source = np.array([[[0, 0, 0], [20, 60, 100], [50, 90, 130], [80, 120, 160]]], dtype=np.uint8)
        damaged = np.array([[[255, 0, 255], [0, 255, 0], [0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        result = restore_source_color(damaged, source, 1, 0.7)
        np.testing.assert_array_equal(result[0, 0], source[0, 0])
        # Orange reference remains R > G > B, including darkened/brightened regions.
        self.assertTrue(np.all(result[0, 1:, 2] > result[0, 1:, 1]))
        self.assertTrue(np.all(result[0, 1:, 1] > result[0, 1:, 0]))
        self.assertFalse(np.array_equal(result, source))

    def test_flat_damaged_image_stays_finite_and_neutral_reference_stays_neutral(self):
        source = np.repeat(np.array([[0, 64, 128, 255]], np.uint8)[..., None], 3, axis=-1)
        result = restore_source_color(np.full_like(source, 255), source, 1, 1)
        np.testing.assert_array_equal(result[..., 0], result[..., 1])
        np.testing.assert_array_equal(result[..., 1], result[..., 2])
        self.assertEqual(result.dtype, np.uint8)

    def test_external_controls_are_bounded_and_old_recipe_is_unchanged(self):
        self.assertEqual(recipe({})['source_color'], 0)
        for values in ({'source_color': -0.1}, {'source_color': float('nan')}, {'luma_change': 1.1}):
            with self.assertRaises(ValueError):
                recipe(values)


if __name__ == '__main__':
    unittest.main()
