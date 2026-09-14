"""Lossless SDR delivery with restored VFR timestamps and sample-trimmed PCM audio."""
import time
from fractions import Fraction

import av

from art_experiment_frames import blend, frames, resize, signature, video_signature
from art_experiment_run import FFMPEG
from art_probe_support import command
from art_processing_session import checkpoint, progress
from art_video_audio import verify_audio


def milliseconds(value):
    return int(Fraction(value) * 1000 + Fraction(1, 2))


def export_video(result, spec, timeline, tail, directory, settings=None, original=None):
    start = Fraction(str(spec['start']))
    pts = [milliseconds(Fraction(f['time']) - start) for f in timeline]
    stop = milliseconds(tail - start)
    if any(b <= a for a, b in zip(pts, pts[1:] + [stop])):
        raise ValueError('Matroska 1ms time base cannot represent this interval without frame collisions')
    video = directory / 'video.partial.mkv'
    scale, info = spec['output_scale'], spec['info']
    size = info['width'] * scale, info['height'] * scale
    evidence, durations, started = [], {}, time.perf_counter()
    iterator = frames(result)
    reference = frames(original) if original is not None else None
    try:
        with av.open(str(video), 'w') as writer:
            stream = writer.add_stream('ffv1', rate=Fraction(info['average_rate']))
            stream.width, stream.height = size
            stream.pix_fmt, stream.time_base = 'bgr0', Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            stream.codec_context.color_range, stream.codec_context.colorspace = 2, 0
            for index, (stamp, next_stamp) in enumerate(zip(pts, pts[1:] + [stop])):
                checkpoint()
                frame = next(iterator, None)
                if frame is None or frame.pts * frame.time_base != index:
                    raise ValueError('Inference frame missing or reordered before delivery')
                pixels = resize(frame.to_ndarray(format='bgr24'), *size)
                if reference is not None:
                    raw = next(reference, None)
                    if raw is None or raw.pts * raw.time_base != index:
                        raise ValueError('Original reference frame missing or reordered')
                    pixels = blend(pixels, resize(raw.to_ndarray(format='bgr24'), *size), settings['retain'])
                out = av.VideoFrame.from_ndarray(pixels, format='bgr24')
                out.pts, out.time_base = stamp, Fraction(1, 1000)
                durations[stamp] = next_stamp - stamp
                evidence.append(signature(out))
                for packet in stream.encode(out):
                    packet.duration = durations[packet.pts]
                    writer.mux(packet)
                progress('encode-output', index + 1, len(pts))
            if next(iterator, None) is not None:
                raise ValueError('Inference produced extra frames')
            if reference is not None and next(reference, None) is not None:
                raise ValueError('Original reference has extra frames')
            for packet in stream.encode():
                packet.duration = durations[packet.pts]
                writer.mux(packet)
    finally:
        iterator.close()
        if reference is not None:
            reference.close()
    encode_seconds = time.perf_counter() - started
    pending = directory / 'result.partial.mkv'
    invocation = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-copyts', '-i', video]
    audio = spec['audio'] == 'pcm' and bool(info['audio_streams'])
    if audio:
        origin = Fraction(info['origin'])
        begin, finish = float(start + origin), float(tail + origin)
        invocation += ['-i', spec['source'], '-map', '0:v:0', '-map', '1:a',
                       '-filter:a', f'atrim=start={begin:.12f}:end={finish:.12f},asetpts=PTS-({begin:.12f})/TB',
                       '-c:a', 'pcm_f32le']
    else:
        invocation += ['-map', '0:v:0', '-an']
    invocation += ['-map_metadata', '-1', '-c:v', 'copy', '-avoid_negative_ts', 'disabled', pending]
    progress('mux-audio')
    started = time.perf_counter()
    command(invocation, directory / 'export.log')
    mux_seconds = time.perf_counter() - started
    if video_signature(pending) != evidence:
        raise ValueError('Delivery changed pixels, timestamps, frame count or dimensions')
    with av.open(str(pending)) as reader:
        audio_streams = list(reader.streams.audio)
        if len(audio_streams) != (len(info['audio_streams']) if audio else 0):
            raise ValueError('Output audio stream count differs')
        audio_info = [dict(index=s.index, codec=s.codec_context.name,
                           sample_rate=s.codec_context.sample_rate) for s in audio_streams]
        packets = [p for p in reader.demux(video=0) if p.pts is not None]
        observed_end = (packets[-1].pts + packets[-1].duration) * packets[-1].time_base if packets else None
        # Matroska DefaultDuration is nanoseconds; FFmpeg truncates it to the packet's ms time base.
        if (len(packets) != len(pts) or packets[-1].duration <= 0
                or abs(observed_end - Fraction(stop, 1000)) > Fraction(1, 1000)):
            raise ValueError('Delivery dropped frames or changed the final frame duration')
    audio_evidence = verify_audio(spec['source'], pending, start, tail, Fraction(info['origin']), len(audio_info))
    return dict(pending=str(pending), pre_encode_frames=evidence, output_dimensions=list(size),
                source_timeline=timeline, timestamp_origin='source container start_time',
                output_time_origin=str(start), video_end_seconds=str(Fraction(stop, 1000)),
                decoded_video_end_seconds=str(observed_end),
                timestamp_quantization='nearest 1ms; collisions rejected; source rational times retained',
                color='8-bit full-range BGR; explicit source SDR matrix/range; FFV1 lossless',
                audio_policy='trim at decoded sample boundaries; PCM float32' if audio else 'omitted',
                audio_streams=audio_info, audio_verification=audio_evidence, command=list(map(str, invocation)),
                encode_seconds=encode_seconds, audio_mux_seconds=mux_seconds)
