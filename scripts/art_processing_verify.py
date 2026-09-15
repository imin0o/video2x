"""Explicit GPU acceptance: A/B/A, restart, zero, cancellation/failure and M1 compatibility."""
import argparse
import itertools
import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

from art_experiment_frames import frames, signature
from art_experiment_run import baseline_context, sampled_run
from art_model_inspect import ROOT
from art_probe_support import save_json
from art_processing import load_run, run
from art_processing_config import configuration
from art_processing_session import Cancelled, Session

ADOPTED_RUN_DIRECTORIES = {'melt-16': 'a', 'melt-24': 'melt-24'}  # Read back by art_release_report.


def verify(baseline_dir, output, cli, legacy):
    baseline = baseline_context(baseline_dir, cli)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', checks={})
    try:
        configs = {}
        for name in ('melt-16', 'melt-24'):
            configs[name] = configuration(json.loads((legacy / name / 'recipe.json').read_text(encoding='utf-8')))
        print('A: adopted melt-16', flush=True)
        first = run(baseline, output / ADOPTED_RUN_DIRECTORIES['melt-16'], configs['melt-16'], cli)
        print('B: independent input/weight/feature effects and two passes', flush=True)
        other = dict(seed=19, weight_strength=0.4, input_noise=3, feature_strength=0.2,
                     feature_mode='mask', passes=2, time_mode='smooth-constant')
        if run(baseline, output / 'b', other, cli)['final_frames'] == first['final_frames']:
            raise AssertionError('B did not change the output; A/B/A would be vacuous')
        print('A again in the same host process', flush=True)
        run(baseline, output / 'a-again', configs['melt-16'], cli, replay=first)
        report['checks']['A_B_A'] = True
        print('Cancellation during child inference', flush=True)
        session = Session()

        def cancelling(*args, **kwargs):
            timer = threading.Timer(0.7, session.cancel)
            timer.start()
            try:
                return sampled_run(*args, **kwargs)
            finally:
                timer.cancel()
                timer.join()

        with patch('art_experiment_run.sampled_run', side_effect=cancelling):
            try:
                run(baseline, output / 'cancelled', other, cli, session=session)
            except Cancelled:
                pass
            else:
                raise AssertionError('Cancellation was not exercised')
        run(baseline, output / 'after-cancel', configs['melt-16'], cli, replay=first)
        report['checks']['cancel_then_A'] = True
        print('Exception after GPU inference', flush=True)

        def failing(*args, **kwargs):
            sampled_run(*args, **kwargs)
            raise RuntimeError('Injected failure after GPU inference')

        with patch('art_experiment_run.sampled_run', side_effect=failing):
            try:
                run(baseline, output / 'failed', other, cli)
            except RuntimeError as error:
                if str(error) != 'Injected failure after GPU inference':
                    raise
            else:
                raise AssertionError('Failure was not exercised')
        run(baseline, output / 'after-failure', configs['melt-16'], cli, replay=first)
        report['checks']['failure_then_A'] = True
        for name, expected in (('cancelled', 'cancelled'), ('failed', 'failed')):
            record = json.loads((output / name / 'run.json').read_text(encoding='utf-8'))
            if record['status'] != expected or not record['engine']['original_hashes_preserved']:
                raise AssertionError('Failure lifecycle or source preservation differs')
        print('Replay in a new Python process', flush=True)
        subprocess.run([sys.executable, '-B', ROOT / 'scripts/art_process.py', '--replay',
                        output / 'a/run.json', '--output-dir', output / 'restarted', '--cli', cli], check=True)
        report['checks']['restart_A04'] = load_run(output / 'restarted/run.json')['replay']['exact_frames_match']
        print('Zero effects', flush=True)
        zero = run(baseline, output / 'zero', {}, cli)
        if not zero['engine']['verification']['first_pass_matches_baseline']:
            raise AssertionError('Zero effects differ from the unmodified baseline')
        report['checks']['zero_A06'] = True
        print('Compatibility with both adopted M1 videos', flush=True)
        strong = run(baseline, output / ADOPTED_RUN_DIRECTORIES['melt-24'], configs['melt-24'], cli)
        for name, actual in (('melt-16', first), ('melt-24', strong)):
            parent = json.loads((legacy / name / 'run.json').read_text(encoding='utf-8'))
            reference = frames(parent['result'])
            try:
                expected = [signature(f) for f in itertools.islice(reference, len(actual['final_frames']))]
            finally:
                reference.close()
            if expected != actual['final_frames']:
                raise AssertionError(f'{name} differs from adopted M1 pixels')
            report['checks'][name + '_M1_pixels'] = True
        report['frame_count'] = len(first['final_frames'])
        report['status'] = 'completed'
    except BaseException as error:
        report['status'], report['error'] = 'failed', str(error)
        raise
    finally:
        save_json(output / 'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--legacy-dir', type=Path, default=ROOT / 'build/art/m1-source-color')
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    args = parser.parse_args()
    verify(args.baseline_dir.resolve(), args.output_dir.resolve(), args.cli.resolve(), args.legacy_dir.resolve())


if __name__ == '__main__':
    main()
