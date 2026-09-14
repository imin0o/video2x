"""Process execution, runtime measurements and comparison assets for the M0 probe."""
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from art_model_inspect import ROOT, sha256
from art_processing_session import progress, run_process

TILE_VARIABLE = 'VIDEO2X_ART_TILE'


def command(args, log=None, cwd=ROOT, env=None, on_tick=None):
    args = list(map(str, args))
    if log:
        with Path(log).open('w', encoding='utf-8') as output:
            run_process(args, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, env=env, on_tick=on_tick)
        return ''
    return run_process(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       encoding='utf-8', errors='replace', env=env)


def save_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    pending.replace(path)


def media_info(ffprobe, path):
    return json.loads(command([ffprobe, '-v', 'error', '-count_frames',
                               '-show_streams', '-show_format', '-of', 'json', path]))


def sampled_run(args, log, samples, cwd=ROOT, tile=None):
    # A leftover shell value must not pin tiles silently; only an explicit request sets it.
    env = {key: value for key, value in os.environ.items() if key.upper() != TILE_VARIABLE}
    if tile is not None:
        env[TILE_VARIABLE] = str(tile)
    monitor = None
    started = time.perf_counter()
    smi = shutil.which('nvidia-smi')
    last_tick = 0

    def tick():
        nonlocal last_tick
        if time.perf_counter() - last_tick >= 0.5:
            last_tick = time.perf_counter()
            text = Path(log).read_text(encoding='utf-8', errors='replace')
            progress('inference', len(re.findall(r'\[art-baseline\] pts=-?\d+ pixel_sha256=', text)))
    try:
        with samples.open('w', encoding='utf-8') as output:
            if smi:
                monitor = subprocess.Popen([smi, '--query-gpu=uuid,name,memory.used',
                                            '--format=csv,noheader,nounits', '-lms', '200'],
                                           stdout=output, stderr=subprocess.DEVNULL)
            command(args, log, cwd=cwd, env=env, on_tick=tick if log else None)
    finally:
        if monitor:
            monitor.terminate()
            monitor.wait(timeout=10)
    elapsed = time.perf_counter() - started
    peaks = {}
    for row in csv.reader(samples.read_text(encoding='utf-8').splitlines()):
        if len(row) == 3 and row[2].strip().isdigit():
            key = row[0].strip()
            peaks[key] = dict(name=row[1].strip(), sampled_peak_mib=max(
                peaks.get(key, {}).get('sampled_peak_mib', 0), int(row[2].strip())))
    return dict(wall_seconds=elapsed, gpu_device_memory=peaks,
                memory_measurement='200 ms samples of total device use, including other applications; not an exact per-process peak')


def parse_diagnostics(log):
    text = log.read_text(encoding='utf-8', errors='replace')
    settings = re.search(r'\[art-baseline\] gpu=(\d+) heap_budget_mb=(\d+) tile=(\d+) scale=(\d+) prepadding=(\d+) tta=(true|false)', text)
    load = re.search(r'\[art-baseline\] model_load_ms=([\d.]+)', text)
    timing = re.findall(r'\[art-baseline\] pts=(-?\d+) inference_ms=([\d.]+) result=(-?\d+)', text)
    pixels = re.findall(r'\[art-baseline\] pts=(-?\d+) pixel_sha256=([0-9a-f]{64})', text)
    if not settings or not load or not pixels or len(pixels) != len(timing):
        raise ValueError('Missing baseline instrumentation; rebuild the instrumented CLI')
    if any(int(item[2]) != 0 for item in timing):
        raise ValueError('Inference failed')
    values = [float(item[1]) for item in timing]
    effective = dict(zip(('gpu', 'heap_budget_mb', 'tile', 'scale', 'prepadding'),
                         map(int, settings.groups()[:5])))
    effective['tta'] = settings[6] == 'true'
    effective['precision'] = dict(fp16_packed=True, fp16_storage=True, fp16_arithmetic=False,
                                  int8_storage=True, int8_arithmetic=False)
    effective['precision_source'] = 'RealESRGAN constructor in the recorded source snapshot'
    return dict(effective=effective, model_load_ms=float(load[1]),
                first_frame_inference_ms=values[0],
                reused_frame_mean_ms=statistics.mean(values[1:]) if len(values) > 1 else None,
                inference_ms=values,
                pre_encode_frames=[dict(pts=int(pts), sha256=digest) for pts, digest in pixels],
                hash_format='packed BGR24, before AVFrame conversion and encoding')


def decoded_hashes(ffmpeg, path, output):
    command([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', path,
             '-map', '0:v:0', '-pix_fmt', 'bgr24', '-fps_mode', 'passthrough',
             '-f', 'framehash', '-hash', 'sha256', output])
    return [row.split(',')[-1].strip() for row in output.read_text().splitlines()
            if row and not row.startswith('#')]


def comparison_assets(ffmpeg, clip, baseline, directory):
    blur = directory / 'blur.mkv'
    command([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', clip,
             '-vf', 'gblur=sigma=2', '-an', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', blur])
    output = directory / 'comparison.mp4'
    filters = ';'.join(f'[{i}:v]scale=640:-2:flags=lanczos,setsar=1[v{i}]' for i in range(3))
    filters += ';[v0][v1][v2]hstack=inputs=3[v]'
    command([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', clip,
             '-i', baseline, '-i', blur, '-filter_complex', filters, '-map', '[v]',
             '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', output])
    command([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', output,
             '-frames:v', '1', '-update', '1', directory / 'comparison.png'])
    return dict(video=str(output), still=str(directory / 'comparison.png'),
                layout=['input', 'unmodified model, scaled to input display size', 'Gaussian blur sigma=2'],
                display_width_per_panel=640)


def binary_inventory(cli):
    paths = [cli, *cli.parent.glob('*.dll')]
    return {str(path): sha256(path) for path in paths}


def gpu_inventory():
    smi = shutil.which('nvidia-smi')
    if not smi:
        return {'available': False}
    return {'available': True, 'csv': command([smi,
            '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv']).strip()}
