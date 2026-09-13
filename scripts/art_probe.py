"""M0 baseline CLI: save effective settings, timings and pre-encode pixel comparisons."""
import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from art_model_inspect import MODEL, ROOT, inspect_model, sha256
from art_probe_support import (binary_inventory, command, comparison_assets, decoded_hashes,
                               gpu_inventory, media_info, parse_diagnostics, sampled_run, save_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    args = parser.parse_args()
    if (not math.isfinite(args.start) or not math.isfinite(args.duration)
            or args.start < 0 or args.duration <= 0 or not 0 <= args.seed < 2**64 or args.device < 0):
        parser.error('Require finite start >= 0, duration > 0, uint64 seed, and device >= 0')
    source, directory, cli = args.input.resolve(), args.output_dir.resolve(), args.cli.resolve()
    ffmpeg = ROOT / 'third_party/ffmpeg-shared/bin/ffmpeg.exe'
    ffprobe = ffmpeg.with_name('ffprobe.exe')
    for path in (source, cli, ffmpeg, ffprobe):
        if not path.is_file():
            parser.error(f'Missing file: {path}')
    info = media_info(ffprobe, source)
    videos = [s for s in info['streams'] if s['codec_type'] == 'video']
    if not videos or args.start >= float(info['format']['duration']):
        parser.error('No video or start is beyond the source duration')
    stream = videos[0]
    pixel_formats = {'yuv420p', 'yuv422p', 'yuv444p', 'yuvj420p', 'yuvj422p', 'yuvj444p',
                     'rgb24', 'bgr24', 'bgr0', 'rgb0', 'bgra', 'rgba', 'nv12'}
    if (stream.get('color_transfer') in ('smpte2084', 'arib-std-b67')
            or stream.get('pix_fmt') not in pixel_formats):
        parser.error('M0 baseline supports common 8-bit SDR formats only')
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(schema_version=1, status='running', started_at=datetime.now(timezone.utc).isoformat(),
                  source=dict(path=str(source), sha256=sha256(source), stream_index=stream['index'],
                              start_seconds=args.start, duration_seconds=args.duration, probe=info),
                  recipe=dict(method='baseline-v1', model='realesr-animevideov3-x2',
                              seed=args.seed, rng='none (seed reserved for later experiments)',
                              target_layers=[], input_noise=0, weight_mutation=0,
                              feature_interference=0, time_mode='fixed'), runs=[])
    manifest = directory / 'run.json'
    save_json(manifest, record)
    try:
        model = inspect_model(MODEL.with_suffix('.param'), MODEL.with_suffix('.bin'))
        save_json(directory / 'model-inventory.json', model)
        record['model'] = {k: model[k] for k in ('param_sha256', 'binary_sha256')}
        record['recipe']['model_sha256'] = record['model']
        tracked = ['src/filter_realesrgan.cpp', 'src/conversions.cpp', 'src/fsutils.cpp',
                   'tools/video2x/src/video2x.cpp',
                   'third_party/librealesrgan_ncnn_vulkan/src/realesrgan.cpp',
                   'scripts/art_probe.py', 'scripts/art_probe_support.py', 'scripts/art_model_inspect.py']
        record['environment'] = dict(os=platform.platform(), python=sys.version,
                                      repository=command(['git', 'rev-parse', 'HEAD']).strip(),
                                      submodules=command(['git', 'submodule', 'status']).splitlines(),
                                      source_sha256={name: sha256(ROOT / name) for name in tracked},
                                      binaries=binary_inventory(cli),
                                      gpu=gpu_inventory(),
                                      ffmpeg=command([ffmpeg, '-version']).splitlines()[0])
        environment = ROOT / 'build/art/environment.json'
        if environment.exists():
            record['environment']['build'] = json.loads(environment.read_text(encoding='utf-8-sig'))
        clip = directory / 'input.mkv'
        extract = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', source,
                   '-ss', str(args.start), '-t', str(args.duration), '-map', f"0:{stream['index']}",
                   '-an', '-sn', '-fps_mode', 'passthrough', '-c:v', 'ffv1', '-level', '3',
                   '-pix_fmt', 'bgr0', clip]
        print('Preparing lossless comparison segment...', flush=True)
        command(extract, directory / 'extract.log')
        prepared = media_info(ffprobe, clip)
        frame_count = int(prepared['streams'][0]['nb_read_frames'])
        if frame_count < 2:
            raise ValueError('At least two frames are needed to measure model reuse')
        record['prepared_input'] = dict(path=str(clip), sha256=sha256(clip), probe=prepared,
                                        command=list(map(str, extract)), audio='omitted for M0 image comparison')
        for index in (1, 2):
            pending = directory / f'baseline-{index}.partial.mkv'
            result = directory / f'baseline-{index}.mkv'
            log = directory / f'baseline-{index}.log'
            invocation = [cli, '-i', clip, '-o', pending, '-p', 'realesrgan',
                          '--realesrgan-model', 'realesr-animevideov3', '-s', '2',
                          '-d', str(args.device), '-c', 'ffv1', '--pix-fmt', 'bgr0',
                          '--no-copy-audio-streams', '--no-copy-subtitle-streams',
                          '--log-level', 'debug', '--no-progress']
            print(f'Running unmodified inference {index}/2 ({frame_count} frames)...', flush=True)
            measured = sampled_run(invocation, log, directory / f'gpu-{index}.csv')
            diagnostics = parse_diagnostics(log)
            if len(diagnostics['pre_encode_frames']) != frame_count:
                raise ValueError('Input/output frame count differs')
            hashes = decoded_hashes(ffmpeg, pending, directory / f'decoded-{index}.sha256')
            if hashes != [x['sha256'] for x in diagnostics['pre_encode_frames']]:
                raise ValueError('Lossless output differs from pre-encode pixels')
            pending.rename(result)
            output_info = media_info(ffprobe, result)
            record['runs'].append(dict(command=list(map(str, invocation)), output=str(result),
                                       output_probe=output_info,
                                       measurements=measured, **diagnostics))
            save_json(manifest, record)
        first, second = record['runs']
        settings = ('gpu', 'tile', 'scale', 'prepadding', 'tta', 'precision')
        consistent = all(first['effective'][key] == second['effective'][key] for key in settings)
        equal = first['pre_encode_frames'] == second['pre_encode_frames']
        preserved = sha256(source) == record['source']['sha256'] and all(
            sha256(MODEL.with_suffix(suffix)) == record['model'][key]
            for suffix, key in (('.param', 'param_sha256'), ('.bin', 'binary_sha256')))
        record['verification'] = dict(frame_count=frame_count, effective_settings_match=consistent,
                                      pre_encode_frames_equal=equal, original_hashes_preserved=preserved,
                                      decoded_pixels_match_pre_encode=True)
        if not all((consistent, equal, preserved)):
            raise ValueError('Baseline reproducibility or source integrity check failed')
        print('Building input / baseline / blur comparison...', flush=True)
        record['comparison'] = comparison_assets(ffmpeg, clip, Path(first['output']), directory)
        record['status'] = 'completed'
    except BaseException as error:
        record['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_json(manifest, record)
    print(f'Baseline verified: {frame_count} frames; record: {manifest}')


if __name__ == '__main__':
    main()
