"""Escalated M1 weight experiments after the first suite failed creator review."""
import argparse
import html
import json
from pathlib import Path

import numpy as np

from art_experiment_frames import frames
from art_experiment_model import recipe
from art_experiment_run import baseline_context, execute
from art_model_inspect import ROOT
from art_probe_support import save_json


def candidates():
    all_layers = [f'Conv_{i}' for i in range(0, 35, 2)]
    melts = [(8, 0.4, 0.5), (16, 0.6, 0.5), (24, 0.8, 0.7)]
    return [(f'all-noise-{value:g}', dict(weight_layers=all_layers, weight_strength=value))
            for value in (0.25, 0.4, 0.6, 0.8, 1.2)] + [
                ('early-noise-2', dict(weight_layers=['Conv_0', 'Conv_2', 'Conv_4'], weight_strength=2)),
                ('late-noise-2', dict(weight_layers=['Conv_30', 'Conv_32', 'Conv_34'], weight_strength=2))] + [
                    (f'melt-{radius}', dict(weight_layers=all_layers, weight_strength=strength,
                                            input_blur=radius, luma_change=luma))
                    for radius, strength, luma in melts] + [
                    # Ablation: identical blur and color finishing with the distributed weights.
                    (f'melt-{radius}-weights-0', dict(input_blur=radius, luma_change=luma))
                    for radius, _, luma in melts]


def first_pixels(path):
    stream = frames(path)
    try:
        return next(stream).to_ndarray(format='bgr24')
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--select', nargs='+', choices=[name for name, _ in candidates()])
    parser.add_argument('--raw-colors', action='store_true', help='Uncorrected model colors for diagnostic intermediates')
    args = parser.parse_args()
    cli = args.cli.resolve()
    baseline = baseline_context(args.baseline_dir.resolve(), cli)
    settings = [(name, recipe(values | dict(seed=args.seed, source_color=0 if args.raw_colors else 1)))
                for name, values in candidates() if not args.select or name in args.select]
    normal = first_pixels(baseline['runs'][0]['output'])
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    duration = float(baseline['prepared_input']['probe']['format']['duration'])
    report = dict(status='running', acceptance='not_evaluated', duration_seconds=duration, candidates=[])
    save_json(directory / 'review.json', report)
    cards = []
    try:
        for name, values in settings:
            print(name, flush=True)
            run = execute(baseline, directory / name, values, cli)
            pixels = first_pixels(run['result'])
            metrics = dict(mae=float(np.mean(np.abs(pixels.astype(np.float32) - normal))),
                           saturated_fraction=float(np.mean((pixels == 0) | (pixels == 255))),
                           image_std=float(pixels.std()))
            report['candidates'].append(dict(name=name, run=str(directory / name / 'run.json'),
                                              comparison=run['comparison']['video'], metrics=metrics))
            save_json(directory / 'review.json', report)
            label = html.escape(name)
            cards.append(f'<article><h2>{label}</h2><video controls loop muted preload="none" '
                         f'poster="{label}/comparison.png" src="{label}/comparison.mp4" width="960"></video>'
                         f'<p><a href="{label}/recipe.json">Recipe</a></p></article>')
            print(json.dumps(metrics), flush=True)
        report['status'] = 'completed_awaiting_creator'
    except BaseException as error:
        report['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save_json(directory / 'review.json', report)
        (directory / 'index.html').write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>M1 destruction probe</title>'
            '<style>body{background:#151515;color:#eee;font:16px system-ui;margin:24px}'
            'video{max-width:100%;height:auto}a{color:#9bcaff}</style>'
            f'<h1>M1 destruction probe</h1><p>{duration:g} seconds per candidate. '
            'Top: input / normal. Bottom: blur control / candidate. '
            'Input-blur candidates use the same blur strength in their control. '
            'Visual acceptance is pending.</p>' + ''.join(cards) + '</html>',
            encoding='utf-8')


if __name__ == '__main__':
    main()
