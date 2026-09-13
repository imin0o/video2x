"""M1 experiment CLI: one editable recipe or the E01-E05 comparison suite."""
import argparse
import json
from pathlib import Path

from art_experiment_model import recipe
from art_experiment_run import baseline_context, execute
from art_model_inspect import ROOT
from art_probe_support import save_json


def candidates(seed):
    cases = [('zero', {})]
    for label, value in [('low', 3), ('medium', 12), ('high', 36)]:
        cases.append((f'input-noise-{label}', dict(input_noise=value)))
    for mode, strengths in [('noise', (0.15, 0.5, 1.0)), ('decay', (0.1, 0.4, 0.8))]:
        for label, value in zip(('low', 'medium', 'high'), strengths):
            cases.append((f'weight-{mode}-{label}', dict(weight_mode=mode, weight_strength=value)))
    for mode in ('noise', 'decay', 'mask'):
        for label, value in zip(('low', 'medium', 'high'), (0.1, 0.4, 0.8)):
            cases.append((f'feature-{mode}-{label}', dict(feature_mode=mode, feature_strength=value)))
    cases += [('input-blur', dict(input_blur=2)), ('output-blur', dict(output_blur=4)),
              ('input-smooth', dict(input_noise=36, time_mode='smooth')),
              ('reinput', dict(weight_strength=1.0, passes=2)),
              ('retain-half', dict(weight_strength=1.0, retain=0.5)),
              ('retain-original', dict(weight_strength=1.0, retain=1.0)),
              ('weight-noise-high-repeat', dict(weight_strength=1.0)),
              ('zero-repeat', {})]
    return [(name, recipe(values | dict(seed=seed))) for name, values in cases]


def time_suite(baseline, output, seed, cli):
    if float(baseline['prepared_input']['probe']['format']['duration']) < 4:
        raise ValueError('Time comparison needs at least one full 4-second cycle')
    settings = recipe(dict(seed=seed, input_noise=36))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', runs=[])
    manifest = output / 'summary.json'
    save_json(manifest, report)
    try:
        for mode in ('fixed', 'smooth'):
            print(f'Time comparison: {mode}', flush=True)
            run = execute(baseline, output / mode, settings | dict(time_mode=mode), cli)
            report['runs'].append(dict(mode=mode, record=str(output / mode / 'run.json'),
                                        frames=run['verification']['frame_count'],
                                        tail=run['verification']['tail_time'],
                                        model_load_ms=run['runs'][0]['model_load_ms'],
                                        input_update_ms=run['transformations'][0]['reused_update_mean_ms']))
            save_json(manifest, report)
        report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save_json(manifest, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--recipe', type=Path)
    group.add_argument('--suite', action='store_true')
    group.add_argument('--time-suite', action='store_true', help='Fixed/smooth comparison on a >=4 second M0 clip')
    parser.add_argument('--seed', type=int, default=0, help='Suite seed; single runs use the recipe seed')
    args = parser.parse_args()
    baseline = baseline_context(args.baseline_dir.resolve(), args.cli.resolve())
    output = args.output_dir.resolve()
    if args.recipe:
        values = recipe(json.loads(args.recipe.read_text(encoding='utf-8')))
        execute(baseline, output, values, args.cli.resolve())
        print(f'Verified candidate: {output / "comparison.mp4"}', flush=True)
        return
    if args.time_suite:
        time_suite(baseline, output, args.seed, args.cli.resolve())
        return
    cases = candidates(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    review = dict(status='running', baseline=str(args.baseline_dir.resolve()), candidates=[],
                  acceptance=dict(A01=None, A02=None, adopted=[], deferred=[],
                                  status='creator_review_required'))
    path = output / 'review.json'
    save_json(path, review)
    try:
        for index, (name, settings) in enumerate(cases, 1):
            print(f'[{index}/{len(cases)}] {name}', flush=True)
            record = execute(baseline, output / name, settings, args.cli.resolve())
            review['candidates'].append(dict(name=name, recipe=str(output / name / 'recipe.json'),
                                              run=str(output / name / 'run.json'),
                                              comparison=record['comparison']['video'],
                                              wall_seconds=record['wall_seconds']))
            save_json(path, review)
        for first, second in [('zero', 'zero-repeat'), ('weight-noise-high', 'weight-noise-high-repeat')]:
            a, b = [json.loads((output / name / 'run.json').read_text(encoding='utf-8'))
                    for name in (first, second)]
            if a['runs'][0]['pre_encode_frames'] != b['runs'][0]['pre_encode_frames'] or a['model'] != b['model']:
                raise ValueError(f'Repeated candidate differs: {first}')
        zero = json.loads((output / 'zero/run.json').read_text(encoding='utf-8'))
        if not zero['verification']['first_pass_matches_baseline']:
            raise ValueError('Zero strengths do not match the M0 baseline')
        review['verification'] = dict(zero_matches_baseline=True, repeated_weight_and_zero_match=True)
        review['status'] = 'completed_awaiting_creator'
    except BaseException as error:
        review['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        review['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save_json(path, review)
    print(f'Comparisons ready: {path}', flush=True)


if __name__ == '__main__':
    main()
