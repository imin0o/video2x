"""Keep source chroma while borrowing damaged luminance without neon color shifts."""
import numpy as np


def restore_source_color(pixels, reference, amount, luma_change):
    if amount == 0:
        return pixels
    candidate = pixels.astype(np.float32)
    source = reference.astype(np.float32)
    # BGR coefficients, applied to the same 8-bit SDR representation as M1.
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    damaged_y = np.sum(candidate * weights, axis=-1)
    source_y = np.sum(source * weights, axis=-1)
    deviation = float(damaged_y.std())
    if deviation > 1e-6:
        mapped_y = ((damaged_y - damaged_y.mean()) * (float(source_y.std()) / deviation)
                    + source_y.mean())
    else:
        mapped_y = np.full_like(source_y, source_y.mean())
    # Keep original black bars/dark regions from turning into gray or colored panels.
    protection = np.clip(source_y / 16, 0, 1)
    luminance = np.clip(source_y + luma_change * protection * (mapped_y - source_y), 0, 255)
    chroma = source - source_y[..., None]
    positive = np.maximum(chroma.max(axis=-1), 1e-6)
    negative = np.maximum(-chroma.min(axis=-1), 1e-6)
    # Reduce chroma uniformly when needed to fit the gamut; avoid per-channel clipping.
    gamut_scale = np.minimum(1, np.minimum((255 - luminance) / positive, luminance / negative))
    restored = luminance[..., None] + gamut_scale[..., None] * chroma
    result = (1 - amount) * candidate + amount * restored
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
