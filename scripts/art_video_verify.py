"""M3 GPU acceptance and performance evidence; visual/wait-time acceptance stays with the creator."""
import argparse
import json
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from art_experiment_run import FFMPEG, baseline_context, sampled_run
from art_model_inspect import ROOT
from art_probe_support import command, save_json
from art_processing import load_run, run
from art_processing_config import configuration
from art_processing_session import Cancelled, Session
from art_video_source import request


def indexed(record):
    return {f['time']: p['sha256'] for f, p in zip(record['source_timeline'], record['final_frames'], strict=True)}


def verify(baseline_dir, output, cli):
    baseline = baseline_context(baseline_dir, cli)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', checks={}, creator_acceptance=dict(flicker='pending', wait_time='pending'))
    config = configuration(json.loads((ROOT / 'build/art/m1-source-color/melt-16/recipe.json').read_text(encoding='utf-8')))
    settings = config['settings'] | dict(input_noise=6, time_mode='smooth', period=4, phase=.15, modulation_depth=.7)
    source = ROOT / 'test/test.mp4'
    try:
        print('Full 5-second source-resolution GPU export', flush=True)
        full = run(baseline, output / 'full', settings, cli, video=request(source, 0, 5))
        print('Mid-interval preview with exactly the same inference settings', flush=True)
        part = run(baseline, output / 'preview', settings, cli, video=request(source, 2.017, 2.084))
        if indexed(part) != {k: v for k, v in indexed(full).items() if k in indexed(part)}:
            raise AssertionError('A05 preview pixels differ from production')
        for index in range(settings['passes']):
            expected = {f['time']: p['sha256'] for f, p in zip(full['source_timeline'],
                        full['engine']['runs'][index]['pre_encode_frames'], strict=True)}
            actual = {f['time']: p['sha256'] for f, p in zip(part['source_timeline'],
                      part['engine']['runs'][index]['pre_encode_frames'], strict=True)}
            if actual != {k: v for k, v in expected.items() if k in actual}:
                raise AssertionError('A05 inference hashes differ')
        report['checks']['A05_full_preview_exact'] = True
        if (any((output / 'full' / name).exists() for name in ('video.partial.mkv', 'work/pass-1.mkv', 'work/injected.mkv'))
                or full['intermediates']['failures'] or not (output / 'full/input.mkv').is_file()):
            raise AssertionError('Successful run kept intermediate videos or removed evidence')
        report['checks']['B_intermediates_removed_after_success'] = True
        print('Single-frame export and original-size delivery', flush=True)
        single = run(baseline, output / 'single', settings, cli, video=request(source, 0, .01, 1))
        if len(single['final_frames']) != 1:
            raise AssertionError('Single frame dropped')
        retained = run(baseline, output / 'retain-one', settings | dict(retain=1), cli,
                       video=request(source, 0, .01, 1))
        from art_experiment_frames import video_signature
        if retained['final_frames'][0]['sha256'] != video_signature(output / 'retain-one/input.mkv')[0]['sha256']:
            raise AssertionError('Equal-size retain=1 differs from original pixels')
        report['checks']['equal_size_retain_one_exact'] = True
        print('Adopted M1 fixed recipe compatibility', flush=True)
        fixed = run(baseline, output / 'fixed', config, cli, video=request(source, 0, .05))
        legacy = json.loads((ROOT / 'build/art/m2-validation-2/a/run.json').read_text(encoding='utf-8'))
        if [f['sha256'] for f in fixed['final_frames']] != [f['sha256'] for f in legacy['final_frames']]:
            raise AssertionError('M3 source conversion changed adopted pixels')
        report['checks']['adopted_fixed_pixels'] = True
        print('VFR + PCM source with delayed audio', flush=True)
        media = output / 'vfr-audio.mkv'
        command([FFMPEG, '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i',
                 'testsrc2=size=128x64:rate=30:duration=1', '-itsoffset', '0.08', '-f', 'lavfi', '-i',
                 'sine=frequency=750:sample_rate=48000:duration=1', '-vf', r"select=not(eq(mod(n\,3)\,1))",
                 '-fps_mode', 'passthrough', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', '-c:a', 'pcm_f32le', media])
        small = settings | dict(passes=2, feature_strength=.2, feature_mode='mask')
        vfr = run(baseline, output / 'vfr', small, cli, video=request(media, .019, .985, 1))
        preview = run(baseline, output / 'vfr-preview', small, cli, video=request(media, .151, .39, 1))
        if indexed(preview) != {k: v for k, v in indexed(vfr).items() if k in indexed(preview)}:
            raise AssertionError('VFR two-pass preview differs')
        tail = run(baseline, output / 'tail', small, cli, video=request(media, .965, .97, 1))
        # The .9 frame straddles .965 and is clipped; the .967 frame is clipped at .97.
        if (len(tail['final_frames']) != 2 or Fraction(tail['source_timeline'][0]['display_start']) != Fraction(193, 200)
                or Fraction(tail['delivery']['video_end_seconds']) != Fraction(5, 1000)):
            raise AssertionError('Straddling head or short final frame duration differs')
        report['checks']['VFR_audio_tail_two_pass_equal_size'] = True
        for name, cancelled in (('cancelled', True), ('failed', False)):
            session = Session()

            def interrupted(*args, **kwargs):
                if not cancelled:
                    sampled_run(*args, **kwargs)
                    raise MemoryError('Injected allocation failure after GPU inference')
                timer = threading.Timer(.1, session.cancel)
                timer.start()
                try:
                    return sampled_run(*args, **kwargs)
                finally:
                    timer.cancel()
                    timer.join()

            with patch('art_experiment_run.sampled_run', side_effect=interrupted):
                try:
                    run(baseline, output / name, small, cli, video=request(media), session=session)
                except (Cancelled, MemoryError):
                    pass
                else:
                    raise AssertionError('Failure/cancellation was not exercised')
            failed = json.loads((output / name / 'run.json').read_text(encoding='utf-8'))
            if (failed['status'] != name or failed['result'] is not None or not failed['source_preserved']
                    or not failed['engine']['original_hashes_preserved'] or not (output / name / 'work/injected.mkv').is_file()):
                raise AssertionError('Failure lifecycle, kept intermediates or source preservation differs')
            report[name] = dict(error=failed['error'], response_seconds=failed.get('cancel_response_seconds'))
            run(baseline, output / f'after-{name}', small, cli, replay=vfr)
        report['checks']['A08_A10_cancel_failure_recovery'] = True
        print('Replay with audio in a new Python process', flush=True)
        subprocess.run([sys.executable, '-B', ROOT / 'scripts/art_process.py', '--replay',
                        output / 'vfr/run.json', '--output-dir', output / 'restart', '--cli', cli], check=True)
        report['checks']['restart_video_audio_exact'] = load_run(output / 'restart/run.json')['replay']['exact_frames_match']
        report['performance'] = {name: dict(wall_seconds=r['wall_seconds'], prepare_seconds=r['prepare_seconds'],
                                    timings=r['timings'],
                                    delivery_seconds=r['delivery']['encode_seconds'] + r['delivery']['audio_mux_seconds'],
                                    runs=[{k: p[k] for k in ('model_load_ms', 'first_frame_inference_ms',
                                          'reused_frame_mean_ms', 'measurements')} for p in r['engine']['runs']])
                                 for name, r in (('five_seconds', full), ('single_frame', single))}
        report['status'] = 'completed'
    except BaseException as error:
        report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
        raise
    finally:
        save_json(output / 'summary.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    args = parser.parse_args()
    verify(args.baseline_dir.resolve(), args.output_dir.resolve(), args.cli.resolve())
