"""Keep source chroma while borrowing damaged luminance without neon color shifts."""
import numpy as np

# BGR coefficients, applied to the same 8-bit SDR representation as M1.
WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)
# Row bands keep float32 temporaries in cache; per-pixel results do not depend on the band size.
BAND_ROWS = 64


def luma(pixels):
    b, g, r = (pixels[..., i].astype(np.float32) for i in range(3))
    return b * WEIGHTS[0] + g * WEIGHTS[1] + r * WEIGHTS[2]


def restore_band(candidate, source, damaged_y, source_y, amount, luma_change, stats):
    damaged_mean, source_mean, ratio = stats
    if ratio is None:
        mapped_y = np.full_like(source_y, source_mean)
    else:
        mapped_y = damaged_y - damaged_mean
        mapped_y *= ratio
        mapped_y += source_mean
    # Keep original black bars/dark regions from turning into gray or colored panels.
    change = np.clip(source_y / 16, 0, 1)
    change *= luma_change
    mapped_y -= source_y
    mapped_y *= change
    luminance = np.clip(source_y + mapped_y, 0, 255)
    chroma = [source[..., i].astype(np.float32) - source_y for i in range(3)]
    positive = np.maximum(np.maximum(np.maximum(chroma[0], chroma[1]), chroma[2]), 1e-6)
    negative = np.maximum(-np.minimum(np.minimum(chroma[0], chroma[1]), chroma[2]), 1e-6)
    # Reduce chroma uniformly when needed to fit the gamut; avoid per-channel clipping.
    np.divide(255 - luminance, positive, out=positive)
    np.divide(luminance, negative, out=negative)
    gamut_scale = np.minimum(1, np.minimum(positive, negative))
    output = np.empty(candidate.shape, np.uint8)
    for i, channel in enumerate(chroma):
        channel *= gamut_scale
        channel += luminance
        if amount != 1:
            channel = (1 - amount) * candidate[..., i].astype(np.float32) + amount * channel
        np.rint(channel, out=channel)
        np.clip(channel, 0, 255, out=channel)
        output[..., i] = channel
    return output


def restore_source_color(pixels, reference, amount, luma_change):
    if amount == 0:
        return pixels
    damaged_y, source_y = luma(pixels), luma(reference)
    deviation = float(damaged_y.std())  # Statistics stay whole-frame.
    ratio = float(source_y.std()) / deviation if deviation > 1e-6 else None
    stats = damaged_y.mean(), source_y.mean(), ratio
    bands = [slice(row, row + BAND_ROWS) for row in range(0, pixels.shape[0], BAND_ROWS)]
    return np.concatenate([restore_band(pixels[band], reference[band], damaged_y[band], source_y[band],
                                        amount, luma_change, stats) for band in bands])
