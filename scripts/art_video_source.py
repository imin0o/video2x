"""Source-time selection; ordinal intermediate timestamps never drive modulation."""
import copy
import math
from fractions import Fraction
from pathlib import Path

import av

from art_experiment_frames import signature, video_signature
from art_model_inspect import sha256
from art_processing_session import progress, timed, timed_iter

PIXELS = {'yuv420p', 'yuv422p', 'yuv444p', 'yuvj420p', 'yuvj422p', 'yuvj444p',
          'rgb24', 'bgr24', 'bgr0', 'rgb0', 'bgra', 'rgba', 'nv12'}


def request(source, start=0, end=None, output_scale=2, audio='pcm'):
    for name, value in (('start', start), ('end', end)):
        if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    if start is None or (end is not None and end <= start):
        raise ValueError('Require 0 <= start < end')
    if type(output_scale) is not int or output_scale not in (1, 2) or audio not in ('pcm', 'omit'):
        raise ValueError('Output scale must be 1 or 2; audio must be pcm or omit')
    path = Path(source).resolve()
    with av.open(str(path)) as reader:
        if not reader.streams.video:
            raise ValueError('No video stream')
        video = reader.streams.video[0]
        ctx = video.codec_context
        if (ctx.format is None or ctx.format.name not in PIXELS or ctx.color_trc in (16, 18)
                or ctx.colorspace not in (0, 1, 2, 5, 6) or ctx.sample_aspect_ratio not in (None, Fraction(1))):
            raise ValueError('Require square-pixel 8-bit SDR RGB/BT.601/BT.709 video')
        duration = reader.duration / av.time_base if reader.duration is not None else None
        if duration is not None and start >= duration:
            raise ValueError('Start is beyond the source duration')
        origin = Fraction(reader.start_time or 0, av.time_base)
        info = dict(stream_index=video.index, width=ctx.width, height=ctx.height,
                    average_rate=str(video.average_rate or 30),
                    pixel_format=ctx.format.name, colorspace=ctx.colorspace, color_range=ctx.color_range,
                    color_transfer=ctx.color_trc, origin=str(origin), duration=duration,
                    audio_streams=[s.index for s in reader.streams.audio])
    return dict(source=str(path), start=start, end=end, output_scale=output_scale, audio=audio, info=info)


def validate_request(value):
    if (not isinstance(value, dict) or not isinstance(value.get('source'), (str, Path)) or
            value.keys() - {'source', 'start', 'end', 'output_scale', 'audio', 'info'}):
        raise ValueError('Invalid source interval/output settings')
    return request(**{k: v for k, v in value.items() if k != 'info'})


def prepare(spec, directory, baseline, source_hash=None):
    """Decode from the start so GOP seek/decoder history cannot change overlapping frames.

    A frame is selected when its display interval [PTS, next PTS) overlaps [start, end). `time` stays the
    source PTS that drives modulation; display_start/display_end are that interval clipped to the request.
    Source time without video (before the first frame or after the last) is never filled.
    """
    clip, pending, info = directory / 'input.mkv', directory / 'input.partial.mkv', spec['info']
    origin, start = Fraction(info['origin']), Fraction(str(spec['start']))
    end = Fraction(str(spec['end'])) if spec['end'] is not None else None
    limit = spec['end'] if spec['end'] is not None else info['duration']
    estimate = math.ceil(limit * float(Fraction(info['average_rate']))) if limit is not None else None
    selected, evidence, held, decoded = [], [], None, 0
    with av.open(spec['source']) as reader, av.open(str(pending), 'w') as writer:
        source = reader.streams.video[0]
        stream = writer.add_stream('ffv1', rate=1)
        stream.width, stream.height = info['width'], info['height']
        stream.pix_fmt, stream.time_base = 'bgr0', Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)

        def write(frame, stamp, stop):
            display_start, stop = max(stamp, start), min(stop, end) if end is not None else stop
            if stop <= display_start:
                raise ValueError('Invalid source frame duration')
            if (frame.width, frame.height) != (stream.width, stream.height):
                raise ValueError('Changing source dimensions are unsupported')
            if frame.format.name not in PIXELS or frame.colorspace not in (0, 1, 2, 5, 6):
                raise ValueError('Source pixel format or color matrix changed to an unsupported value')
            with timed('decode'):
                # Swscale matrix/range are explicit; unspecified SDR YUV uses BT.601.
                rgb = frame.reformat(format='bgr24', src_colorspace=1 if frame.colorspace == 1 else 5,
                                     src_color_range=frame.color_range or 1, dst_color_range=2)
                out = av.VideoFrame.from_ndarray(rgb.to_ndarray(), format='bgr24')
            out.pts, out.time_base = len(selected) * 1000, Fraction(1, 1000)
            selected.append(dict(time=str(stamp), display_start=str(display_start), display_end=str(stop),
                                 source_pts=frame.pts, source_time_base=str(frame.time_base)))
            evidence.append(signature(out))
            with timed('encode'):
                for packet in stream.encode(out):
                    writer.mux(packet)

        for frame in timed_iter(reader.decode(source), 'decode'):
            if frame.pts is None:
                raise ValueError('Source frame has no timestamp')
            stamp = frame.pts * frame.time_base - origin
            if held is not None and stamp <= held[1]:
                raise ValueError('Source timestamps must be strictly increasing')
            decoded += 1
            progress('decode-source', decoded, estimate, total_exact=False)
            if held is not None and stamp > start:
                write(*held, stamp)
            if end is not None and stamp >= end:
                held = None
                break
            held = frame, stamp
        if held is not None:
            frame, stamp = held
            last = stamp + (frame.duration * frame.time_base if frame.duration else 1 / Fraction(info['average_rate']))
            if last > start:
                write(frame, stamp, last)
        with timed('encode'):
            for packet in stream.encode():
                writer.mux(packet)
    if not selected:
        raise ValueError('The selected interval contains no video frame')
    tail = Fraction(selected[-1]['display_end'])
    if video_signature(pending) != evidence:
        raise ValueError('Source preparation changed decoded pixels or ordinal timestamps')
    pending.rename(clip)
    context = copy.deepcopy(baseline)
    context['source'] = dict(path=spec['source'], sha256=source_hash or sha256(Path(spec['source'])),
                             start_seconds=spec['start'], duration_seconds=float(tail - start))
    context['prepared_input'] = dict(path=str(clip), sha256=sha256(clip), frames=evidence,
                                     probe=dict(streams=[info]), source_times=[f['time'] for f in selected])
    context['runs'][0]['pre_encode_frames'] = []
    return context, selected, tail
