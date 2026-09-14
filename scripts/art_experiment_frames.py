"""Lossless BGR experiment transforms with original frame times (PyAV + NumPy)."""
import hashlib
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageFilter

from art_experiment_model import rng
from art_experiment_color import restore_source_color
from art_processing_session import checkpoint, progress, timed, timed_iter

STAGES = dict(input='input-transform', output='output-transform', resize='reinput-resize')
# Frames in flight while NumPy/Pillow release the GIL; about 150 MB of 4K temporaries per frame.
WORKERS = max(1, min(8, os.cpu_count() or 1))


class InputNoise:
    def __init__(self, shape, seed):
        # Independent B/G/R samples, in source pixel coordinates, before tiling.
        self.first = rng(seed, 'input:0').standard_normal(shape).astype(np.float32)
        self.second = rng(seed, 'input:1').standard_normal(shape).astype(np.float32)

    def apply(self, pixels, strength, seconds, mode='fixed', period=4, phase=0, depth=1):
        if strength == 0:
            return pixels
        field = self.first
        if mode in ('smooth', 'smooth-constant'):
            alpha = depth * (1 - math.cos(2 * math.pi * (seconds / period + phase))) / 2
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


def threaded(stream):
    """FFV1 is intra-only and lossless: frame threads decode identical pixels about 3x faster."""
    if stream.codec_context.name == 'ffv1':
        stream.thread_type = 'AUTO'
    return stream


def frames(path, operation='decode'):
    with av.open(str(path)) as container:
        for frame in timed_iter(container.decode(threaded(container.streams.video[0])), operation):
            checkpoint()  # Full-clip signatures stay cancellable between frames.
            yield frame


def signature(frame, operation='pixel_hash'):
    with timed(operation):
        return dict(time=str(frame.pts * frame.time_base), width=frame.width, height=frame.height,
                    sha256=hashlib.sha256(frame.to_ndarray(format='bgr24').tobytes()).hexdigest())


def video_signature(path):
    """Decode-back verification; timed separately from the decode that feeds processing."""
    return [signature(frame, 'verify_hash') for frame in frames(path, 'verify_decode')]


def transform(source, destination, settings, start=0, original=None, output_size=None, stage='input',
              source_times=None, total=None, pass_index=None):
    """Preserve prepared-clip timestamps; time modulation adds the source start offset."""
    started = time.perf_counter()
    pending = Path(destination).with_suffix('.partial.mkv')
    if pending.exists() or Path(destination).exists():
        raise FileExistsError(destination)
    reference = frames(original) if original else None
    count, updates, clipped, values, evidence, noise = 0, [], 0, 0, [], None
    total = len(source_times) if source_times is not None else total

    def update(pixels, raw_reference, seconds, pts, time_base, size):
        """Pure per-frame work; runs on pool threads, so frame results do not depend on scheduling."""
        began = time.perf_counter()
        if stage == 'input':
            pixels = blur(pixels, settings['input_blur'])
            if settings['input_noise']:
                pixels = noise.apply(pixels, settings['input_noise'], seconds, settings['time_mode'], settings['period'],
                                     settings['phase'], settings.get('modulation_depth', 1))
        pixels = resize(pixels, *size)
        if stage == 'output':
            pixels = blur(pixels, settings['output_blur'])
        if raw_reference is not None:
            if stage == 'output' and settings.get('source_color', 0):
                color_reference = resize(blur(raw_reference, settings['input_blur']), *size)
                pixels = restore_source_color(pixels, color_reference, settings['source_color'], settings['luma_change'])
            if settings['retain']:
                pixels = blend(pixels, resize(raw_reference, *size), settings['retain'])
        elapsed = (time.perf_counter() - began) * 1000
        out = av.VideoFrame.from_ndarray(pixels, format='bgr24')
        out.pts, out.time_base = pts, time_base
        return out, elapsed, int(np.count_nonzero((pixels == 0) | (pixels == 255))), pixels.size, signature(out)

    try:
        with av.open(str(source)) as reader, av.open(str(pending), 'w') as writer, ThreadPoolExecutor(WORKERS) as pool:
            source_stream = threaded(reader.streams.video[0])
            stream = writer.add_stream('ffv1', rate=source_stream.average_rate or Fraction(30))
            stream.width, stream.height = size = output_size or (source_stream.width, source_stream.height)
            stream.pix_fmt = 'bgr0'
            stream.time_base = source_stream.time_base
            stream.codec_context.time_base = source_stream.time_base

            def finish(future):
                nonlocal count, clipped, values
                with timed('pixel'):  # Wall time waiting for pool updates, saturation statistics and hashes.
                    out, elapsed, saturated, channels, digest = future.result()
                updates.append(elapsed)
                clipped, values = clipped + saturated, values + channels
                evidence.append(digest)
                with timed('encode'):
                    for packet in stream.encode(out):
                        writer.mux(packet)
                count += 1
                progress(STAGES.get(stage, stage), count, total, pass_index)

            window = deque()
            for index, frame in enumerate(timed_iter(reader.decode(source_stream), 'decode')):
                checkpoint()
                raw_reference = None
                with timed('decode'):
                    pixels = frame.to_ndarray(format='bgr24')
                if reference is not None:
                    ref = next(reference, None)
                    if ref is None or ref.pts * ref.time_base != frame.pts * frame.time_base:
                        raise ValueError('Original/result frame timestamps differ')
                    with timed('decode'):
                        raw_reference = ref.to_ndarray(format='bgr24')
                if stage == 'input' and settings['input_noise'] and noise is None:
                    noise = InputNoise(pixels.shape, settings['seed'])
                seconds = (float(Fraction(source_times[index])) if source_times is not None
                           else start + float(frame.pts * frame.time_base))
                window.append(pool.submit(update, pixels, raw_reference, seconds, frame.pts, frame.time_base, size))
                if len(window) >= WORKERS:
                    finish(window.popleft())
            while window:
                finish(window.popleft())
            if reference is not None and next(reference, None) is not None:
                raise ValueError('Original/result frame counts differ')
            with timed('encode'):
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
