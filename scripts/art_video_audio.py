"""Independent decoded audio timing checks and replay evidence for every source stream."""
import hashlib
import math
from fractions import Fraction

import av

from art_processing_session import checkpoint, progress, timed_iter


def verify_audio(source, output, start, tail, origin, expected_streams):
    evidence = []
    if expected_streams:
        progress('verify-audio', 0, expected_streams, unit='streams')
    for index in range(expected_streams):
        count, first, last, digest, previous = 0, None, None, hashlib.sha256(), None
        with av.open(str(output)) as reader:
            for frame in timed_iter(reader.decode(audio=index), 'verify_decode'):
                checkpoint()
                stamp = frame.pts * frame.time_base
                if previous is not None and stamp < previous - Fraction(1, 1000):
                    raise ValueError('Audio timestamps overlap or regress')
                first = stamp if first is None else first
                last = stamp + Fraction(frame.samples, frame.sample_rate)
                previous = last
                count += frame.samples
                digest.update(frame.to_ndarray().tobytes())
        expected_count, expected_first, expected_last, source_end = 0, None, None, None
        with av.open(str(source)) as reader:
            sample_rate = reader.streams.audio[index].codec_context.sample_rate
            for frame in timed_iter(reader.decode(audio=index), 'verify_decode'):
                checkpoint()
                stamp = frame.pts * frame.time_base - origin
                # Matroska rounds audio packet PTS to ms; continuous samples retain their clock.
                if source_end is not None and abs(stamp - source_end) <= Fraction(1, 1000):
                    stamp = source_end
                source_end = stamp + Fraction(frame.samples, frame.sample_rate)
                if stamp >= tail:
                    break
                low = max(0, math.ceil((start - stamp) * frame.sample_rate))
                high = min(frame.samples, math.ceil((tail - stamp) * frame.sample_rate))
                if high <= low:
                    continue
                expected_count += high - low
                expected_first = stamp + Fraction(low, frame.sample_rate) - start if expected_first is None else expected_first
                expected_last = stamp + Fraction(high, frame.sample_rate) - start
        tolerance = Fraction(1, 1000) + Fraction(1, sample_rate)
        if (abs(count - expected_count) > 1 or (count and
                (abs(first - expected_first) > tolerance or abs(last - expected_last) > tolerance))):
            raise ValueError('Decoded audio samples or synchronization differ from the source interval')
        evidence.append(dict(stream=index, samples=count, first_time=str(first), end_time=str(last),
                             pcm_sha256=digest.hexdigest(), source_samples=expected_count,
                             synchronized=True))
        progress('verify-audio', index + 1, expected_streams, unit='streams')
    return evidence
