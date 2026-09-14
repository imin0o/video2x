"""Execute and verify one M1 candidate against a completed M0 baseline."""
import json
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import av
import numpy as np
import PIL

from art_experiment_frames import transform, video_signature
from art_experiment_model import MODEL_HASHES, model_copy, recipe
from art_model_inspect import MODEL, ROOT, sha256
from art_probe_support import (binary_inventory, command, decoded_hashes, parse_diagnostics,
                               sampled_run, save_json)
from art_processing_session import is_cancellation

FFMPEG = ROOT / 'third_party/ffmpeg-shared/bin/ffmpeg.exe'


def software_environment(scripts=None):
    scripts = (ROOT / 'scripts').glob('art_*.py') if scripts is None else scripts
    return dict(os=platform.platform(), python=sys.version, numpy=np.__version__,
                pyav=av.__version__, pillow=PIL.__version__,
                scripts={path.name: sha256(path) for path in scripts})


def baseline_context(directory, cli):
    path = directory / 'run.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    return validate_baseline(record, cli)


def validate_baseline(record, cli):
    if record.get('status') != 'completed' or record['recipe']['method'] != 'baseline-v1':
        raise ValueError('A completed M0 baseline is required')
    clip = Path(record['prepared_input']['path'])
    if sha256(clip) != record['prepared_input']['sha256']:
        raise ValueError('Prepared input hash differs from M0')
    if sha256(Path(record['source']['path'])) != record['source']['sha256']:
        raise ValueError('Original input hash differs from M0')
    if tuple(sha256(MODEL.with_suffix(x)) for x in ('.param', '.bin')) != MODEL_HASHES:
        raise ValueError('Original model hash differs from M0')
    if binary_inventory(cli) != record['environment']['binaries']:
        raise ValueError('CLI/DLL snapshot differs; generate a new M0 baseline with this CLI')
    return record


def resolved_model(working_directory, base, expected):
    """video2x tries models/ relative to its cwd before its executable directory; require that hit."""
    files = [working_directory / 'models/realesrgan' / f'{base.name}-x2{suffix}' for suffix in ('.param', '.bin')]
    if tuple(sha256(path) if path.is_file() else None for path in files) != tuple(expected):
        raise ValueError(f'Model under {working_directory} is missing or differs; the CLI would load another model')
    return [str(path) for path in files]


def comparison(baseline, candidate, directory, input_blur_control=None):
    clip = Path(baseline['prepared_input']['path'])
    normal = Path(baseline['runs'][0]['output'])
    blur = input_blur_control or clip.parent / 'blur.mkv'
    paths = [clip, normal, blur, candidate]
    labels = ['INPUT', 'NORMAL', 'GAUSSIAN BLUR', re.sub(r'[^A-Za-z0-9_-]', '_', directory.name)]
    if input_blur_control:
        labels[2] = 'INPUT BLUR - SAME STRENGTH'
    # Explicit font avoids this Windows FFmpeg build's missing fontconfig configuration.
    font = Path('C:/Windows/Fonts/arial.ttf')
    filters = ';'.join(f'[{i}:v]scale=640:-2:flags=lanczos,setsar=1,'
                       f'drawtext=fontfile=label.ttf:text={label}:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[v{i}]'
                       for i, label in enumerate(labels))
    filters += ';[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]'
    target = directory / 'comparison.mp4'
    invocation = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n']
    for path in paths:
        invocation += ['-i', path]
    shutil.copyfile(font, directory / 'label.ttf')
    try:
        command(invocation + ['-filter_complex', filters, '-map', '[v]', '-an', '-c:v', 'libx264',
                              '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', target],
                directory / 'comparison.log', cwd=directory)
    finally:
        (directory / 'label.ttf').unlink()
    command([FFMPEG, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', target,
             '-frames:v', '1', '-update', '1', directory / 'comparison.png'])
    return dict(video=str(target), still=str(directory / 'comparison.png'),
                layout=labels, display_width_per_panel=640, blur_control=str(blur))


def execute(baseline, directory, settings, cli, make_comparison=True, pin_tile=False):
    settings = recipe(settings)
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(schema_version=1, status='running', recipe=settings,
                  baseline=baseline, runs=[], transformations=[],
                  environment=software_environment(),
                  semantics=dict(rng='SHA256 domains + NumPy PCG64, m1-pcg64-v1',
                                 input_noise='normal, independent BGR, sigma in 8-bit levels; rint then clip',
                                 features='64 channel-constant normal biases, attenuation or seeded channel mask',
                                 time='source start + decoded clip PTS; raised-cosine blend of two fixed fields',
                                 reinput='second pass is unmodified; Lanczos to source dimensions first',
                                 source_color='source chroma + mean/std-matched damaged luma; gamut-limited; input blur applied to color reference',
                                 audio='omitted for M1 visual experiments'))
    manifest = directory / 'run.json'
    save_json(directory / 'recipe.json', settings)
    save_json(manifest, record)
    started = time.perf_counter()
    try:
        model_start = time.perf_counter()
        base, record['model'] = model_copy(directory / 'models/realesrgan', settings)
        record['model_prepare_ms'] = (time.perf_counter() - model_start) * 1000
        clip = Path(baseline['prepared_input']['path'])
        source_frames = video_signature(clip)
        baseline_run = baseline['runs'][0]
        input_path = clip
        if settings['input_noise'] or settings['input_blur']:
            input_path = directory / 'injected.mkv'
            record['transformations'].append(transform(clip, input_path, settings,
                                                       start=baseline['source']['start_seconds']))
        stream = baseline['prepared_input']['probe']['streams'][0]
        size = stream['width'], stream['height']
        for index in range(settings['passes']):
            if index:
                input_path = directory / 'reinput.mkv'
                record['transformations'].append(transform(directory / 'pass-1.mkv', input_path, recipe({}),
                                                           output_size=size, stage='resize'))
            output = directory / f'pass-{index + 1}.mkv'
            pending = output.with_suffix('.partial.mkv')
            invocation = [cli, '-i', input_path, '-o', pending, '-p', 'realesrgan',
                          '--realesrgan-model', base.name,
                          '-s', '2', '-d', str(baseline_run['effective']['gpu']),
                          '-c', 'ffv1', '--pix-fmt', 'bgr0', '--no-copy-audio-streams',
                          '--no-copy-subtitle-streams', '--log-level', 'debug', '--no-progress']
            log = directory / f'pass-{index + 1}.log'
            working_directory = directory if index == 0 else ROOT
            expected = (record['model']['param_sha256'], record['model']['binary_sha256']) if index == 0 else MODEL_HASHES
            model_files = resolved_model(working_directory, base, expected)
            tile = baseline_run['effective']['tile'] if pin_tile else None
            overrides = {} if tile is None else {'VIDEO2X_ART_TILE': str(tile)}
            measurements = sampled_run(invocation, log, directory / f'gpu-{index + 1}.csv',
                                       cwd=working_directory, tile=tile)
            diagnostics = parse_diagnostics(log)
            for key in ('gpu', 'tile', 'scale', 'prepadding', 'tta', 'precision'):
                if diagnostics['effective'][key] != baseline_run['effective'][key]:
                    raise ValueError(f'Effective {key} differs from baseline')
            hashes = decoded_hashes(FFMPEG, pending, directory / f'decoded-{index + 1}.sha256')
            if hashes != [f['sha256'] for f in diagnostics['pre_encode_frames']]:
                raise ValueError('Decoded output differs from pre-encode pixels')
            actual = video_signature(pending)
            if (len(actual) != len(source_frames)
                    or [f['time'] for f in actual] != [f['time'] for f in source_frames]
                    or any((f['width'], f['height']) != (size[0] * 2, size[1] * 2) for f in actual)):
                raise ValueError('Inference changed frame times/count or expected dimensions')
            pending.rename(output)
            record['runs'].append(dict(command=list(map(str, invocation)), cwd=str(working_directory), output=str(output),
                                       model_files=model_files, environment_overrides=overrides,
                                       measurements=measurements, **diagnostics))
            save_json(manifest, record)
        result = output
        if settings['output_blur'] or settings['retain'] or settings.get('source_color', 0):
            result = directory / 'result.mkv'
            record['transformations'].append(transform(output, result, settings, original=clip, stage='output'))
        record['verification'] = dict(frame_count=len(source_frames), tail_time=source_frames[-1]['time'],
                                      output_dimensions=[size[0] * 2, size[1] * 2],
                                      frames_times_dimensions_and_lossless_pixels=True,
                                      first_pass_matches_baseline=(record['runs'][0]['pre_encode_frames']
                                                                  == baseline_run['pre_encode_frames']))
        if make_comparison:
            control = directory / 'injected.mkv' if settings['input_blur'] and not settings['input_noise'] else None
            record['comparison'] = comparison(baseline, result, directory, control)
        record['result'] = str(result)
        record['status'] = 'completed'
    except BaseException as error:
        record['status'] = 'cancelled' if is_cancellation(error) else 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        preserved = (sha256(Path(baseline['source']['path'])) == baseline['source']['sha256']
                     and tuple(sha256(MODEL.with_suffix(x)) for x in ('.param', '.bin')) == MODEL_HASHES)
        record['original_hashes_preserved'] = preserved
        if not preserved:
            record['status'], record['error'] = 'failed', 'Original input/model integrity check failed'
        record['wall_seconds'] = time.perf_counter() - started
        save_json(manifest, record)
    if not preserved:
        raise ValueError(record['error'])
    return record
