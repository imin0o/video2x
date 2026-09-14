"""Lossless BGR experiment transforms with original frame times (PyAV + NumPy)."""
import hashlib
import math
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageFilter

from art_experiment_model import rng
from art_experiment_color import restore_source_color
from art_processing_session import checkpoint


class InputNoise:
    def __init__(self, shape, seed):
        # Independent B/G/R samples, in source pixel coordinates, before tiling.
        self.first = rng(seed, 'input:0').standard_normal(shape).astype(np.float32)
        self.second = rng(seed, 'input:1').standard_normal(shape).astype(np.float32)

    def apply(self, pixels, strength, seconds, mode='fixed', period=4, phase=0):
        if strength == 0:
            return pixels
        field = self.first
        if mode in ('smooth', 'smooth-constant'):
            alpha = (1 - math.cos(2 * math.pi * (seconds / period + phase))) / 2
            field = (1 - alpha) * field + alpha * self.second
            if mode == 'smooth-constant':
                # Independent unit fields: keep the noise variance constant across the blend.
                field /= math.sqrt((1 - alpha) ** 2 + alpha ** 2)
        return np.clip(np.rint(pixels.astype(np.float32) + strength * field), 0, 255).astype(np.uint8)


def resize(pixels, width, height):
    if pixels.shape[:2] == (height, width):
        return pixels
    return np.array(Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS))


def blur(pixels, radius):
    return np.array(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(radius))) if radius else pixels


def blend(pixels, original, retain):
    if retain == 0:
        return pixels
    if retain == 1:
        return original
    return np.clip(np.rint((1 - retain) * pixels.astype(np.float32)
                          + retain * original.astype(np.float32)), 0, 255).astype(np.uint8)


def frames(path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            checkpoint()  # Full-clip signatures stay cancellable between frames.
            yield frame


def signature(frame):
    return dict(time=str(frame.pts * frame.time_base), width=frame.width, height=frame.height,
                sha256=hashlib.sha256(frame.to_ndarray(format='bgr24').tobytes()).hexdigest())


def video_signature(path):
    return [signature(frame) for frame in frames(path)]


def transform(source, destination, settings, start=0, original=None, output_size=None, stage='input'):
    """Preserve prepared-clip timestamps; time modulation adds the source start offset."""
    started = time.perf_counter()
    pending = Path(destination).with_suffix('.partial.mkv')
    if pending.exists() or Path(destination).exists():
        raise FileExistsError(destination)
    reference = frames(original) if original else None
    count, updates, clipped, values, evidence = 0, [], 0, 0, []
    try:
        with av.open(str(source)) as reader, av.open(str(pending), 'w') as writer:
            source_stream = reader.streams.video[0]
            stream = writer.add_stream('ffv1', rate=source_stream.average_rate or Fraction(30))
            stream.width, stream.height = output_size or (source_stream.width, source_stream.height)
            stream.pix_fmt = 'bgr0'
            stream.time_base = source_stream.time_base
            stream.codec_context.time_base = source_stream.time_base
            noise = None
            for frame in reader.decode(video=0):
                checkpoint()
                pixels = frame.to_ndarray(format='bgr24')
                update = time.perf_counter()
                if stage == 'input':
                    pixels = blur(pixels, settings['input_blur'])
                    if settings['input_noise']:
                        if noise is None:
                            noise = InputNoise(pixels.shape, settings['seed'])
                        pixels = noise.apply(pixels, settings['input_noise'],
                                             start + float(frame.pts * frame.time_base),
                                             settings['time_mode'], settings['period'], settings['phase'])
                pixels = resize(pixels, stream.width, stream.height)
                if stage == 'output':
                    pixels = blur(pixels, settings['output_blur'])
                if reference is not None:
                    ref = next(reference, None)
                    if ref is None or ref.pts * ref.time_base != frame.pts * frame.time_base:
                        raise ValueError('Original/result frame timestamps differ')
                    raw_reference = ref.to_ndarray(format='bgr24')
                    if stage == 'output' and settings.get('source_color', 0):
                        color_reference = resize(blur(raw_reference, settings['input_blur']), stream.width, stream.height)
                        pixels = restore_source_color(pixels, color_reference,
                                                      settings['source_color'], settings['luma_change'])
                    if settings['retain']:
                        pixels = blend(pixels, resize(raw_reference, stream.width, stream.height), settings['retain'])
                updates.append((time.perf_counter() - update) * 1000)
                clipped += int(np.count_nonzero((pixels == 0) | (pixels == 255)))
                values += pixels.size
                out = av.VideoFrame.from_ndarray(pixels, format='bgr24')
                out.pts, out.time_base = frame.pts, frame.time_base
                evidence.append(signature(out))
                for packet in stream.encode(out):
                    writer.mux(packet)
                count += 1
            if reference is not None and next(reference, None) is not None:
                raise ValueError('Original/result frame counts differ')
            for packet in stream.encode():
                writer.mux(packet)
        if not count or video_signature(pending) != evidence:
            raise ValueError('Lossless transform changed pixels, dimensions or timestamps')
        pending.rename(destination)
    finally:
        if reference is not None:
            reference.close()
    return dict(frames=count, wall_seconds=time.perf_counter() - started,
                first_update_ms=updates[0], reused_update_mean_ms=float(np.mean(updates[1:])) if count > 1 else None,
                saturated_channel_fraction=clipped / values, pre_encode_frames=evidence,
                decoded_pixels_and_times_match=True)
