"""Verify saved M1 experiments and build a local comparison index for creator review."""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from art_experiment_frames import frames, resize, video_signature
from art_probe_support import save_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def samples(path, indices):
    return {i: f.to_ndarray(format='bgr24') for i, f in enumerate(frames(path)) if i in indices}


def verify(directory, time_directory=None):
    review = read(directory / 'review.json')
    if review['status'] not in ('completed_awaiting_creator', 'expression_rejected'):
        raise ValueError('Complete the experiment suite before reviewing')
    runs = {c['name']: read(Path(c['run'])) for c in review['candidates']}
    baseline = runs['zero']['baseline']
    count = baseline['verification']['frame_count']
    indices = {0, count // 2, count - 1}
    normal = samples(baseline['runs'][0]['output'], indices)
    source = samples(baseline['prepared_input']['path'], indices)
    metrics, cards = {}, []
    for name, run in runs.items():
        if run['status'] != 'completed' or not run['original_hashes_preserved']:
            raise ValueError(f'Unverified run: {name}')
        selected = samples(run['result'], indices)
        if selected.keys() != normal.keys():
            raise ValueError(f'Missing sampled frames: {name}')
        mae, saturated = [], []
        for i, pixels in selected.items():
            if pixels.shape != normal[i].shape:
                raise ValueError(f'Output size differs: {name}')
            mae.append(float(np.mean(np.abs(pixels.astype(np.float32) - normal[i]))))
            saturated.append(float(np.mean((pixels == 0) | (pixels == 255))))
            if name == 'retain-original':
                original = resize(source[i], pixels.shape[1], pixels.shape[0])
                if not np.array_equal(pixels, original):
                    raise ValueError('Retain=1 does not reproduce the resized original')
        metrics[name] = dict(sample_indices=sorted(indices), mean_absolute_difference=float(np.mean(mae)),
                             saturated_channel_fraction=float(np.mean(saturated)),
                             wall_seconds=run['wall_seconds'], model_prepare_ms=run['model_prepare_ms'],
                             model_load_ms=run['runs'][0]['model_load_ms'],
                             reused_inference_ms=run['runs'][0]['reused_frame_mean_ms'],
                             input_update_ms=[x['reused_update_mean_ms'] for x in run['transformations']],
                             frames=run['verification']['frame_count'], tail=run['verification']['tail_time'])
        label = html.escape(name)
        cards.append(f'<article><h2>{label}</h2><video controls loop muted preload="none" '
                     f'poster="{label}/comparison.png" src="{label}/comparison.mp4"></video>'
                     f'<p><a href="{label}/recipe.json">Recipe</a> / <a href="{label}/run.json">Run</a></p>'
                     f'<pre>{html.escape(json.dumps(run["recipe"], indent=2))}</pre></article>')
    for name in ('zero', 'zero-repeat'):
        if metrics[name]['mean_absolute_difference'] != 0:
            raise ValueError('Zero candidate differs from the baseline')
    for name in ('weight-noise-high', 'feature-noise-high', 'feature-decay-high', 'feature-mask-high'):
        if metrics[name]['mean_absolute_difference'] == 0:
            raise ValueError(f'Nonzero effect did not reach inference: {name}')
    # Full-frame endpoint verification, not just the three visual samples.
    retained = video_signature(runs['retain-original']['result'])
    reference = frames(baseline['prepared_input']['path'])
    import hashlib
    try:
        for item, frame in zip(retained, reference, strict=True):
            pixels = resize(frame.to_ndarray(format='bgr24'), item['width'], item['height'])
            if (item['time'] != str(frame.pts * frame.time_base)
                    or item['sha256'] != hashlib.sha256(pixels.tobytes()).hexdigest()):
                raise ValueError('Retain=1 full-sequence endpoint verification failed')
    finally:
        reference.close()
    report = dict(status='verified', sampled_metrics=metrics, retain_one_all_frames_match=True,
                  note='Pixel differences and saturation describe effects; A01/A02 need creator judgment.')
    if time_directory:
        temporal = read(time_directory / 'summary.json')
        if temporal['status'] != 'completed':
            raise ValueError('Time comparison has not completed')
        for mode, short in [('fixed', 'input-noise-high'), ('smooth', 'input-smooth')]:
            long = read(time_directory / mode / 'run.json')
            if (long['status'] != 'completed' or long['runs'][0]['pre_encode_frames'][:count]
                    != runs[short]['runs'][0]['pre_encode_frames']):
                raise ValueError(f'Short/long sequence prefix differs: {mode}')
            link = quote(Path(os.path.relpath(time_directory / mode, directory)).as_posix())
            duration = float(long['baseline']['prepared_input']['probe']['format']['duration'])
            cards.append(f'<article><h2>{duration:g} seconds: {mode}</h2><video controls loop muted preload="none" '
                         f'poster="{link}/comparison.png" src="{link}/comparison.mp4"></video>'
                         f'<p><a href="{link}/recipe.json">Recipe</a> / '
                         f'<a href="{link}/run.json">Run</a></p></article>')
        report['short_long_prefixes_match'] = True
        report['time_comparison'] = temporal
    save_json(directory / 'metrics.json', report)
    feedback = review.get('acceptance', {}).get('creator_feedback', '')
    feedback_note = f'<p>Creator feedback: {html.escape(feedback)}</p>' if feedback else ''
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>M1 comparisons</title>'
            '<style>body{font:16px system-ui;background:#151515;color:#eee;margin:24px}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px}'
            'article{background:#252525;padding:16px}video{width:100%}a{color:#9bcaff}'
            'pre{white-space:pre-wrap}h2{font-size:19px}@media(max-width:600px){main{display:block}}</style>'
            '<h1>M1 comparisons</h1>' + feedback_note + '<p>Top: input / normal. Bottom: Gaussian blur / candidate. '
            'Each video contains the same timestamps and display sizes. Audio is omitted.</p>'
            '<p>A01: select a shape/texture recipe. A02: select an uncanny blur recipe. '
            'Record adopted/deferred candidates and reasons in <a href="review.json">review.json</a>.</p>'
            '<p><a href="metrics.json">Measured differences, timings and verification</a></p><main>'
            + ''.join(cards) + '</main></html>')
    (directory / 'index.html').write_text(page, encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--time-dir', type=Path)
    args = parser.parse_args()
    verify(args.suite_dir.resolve(), args.time_dir.resolve() if args.time_dir else None)
    print(f'Verified comparison index: {args.suite_dir / "index.html"}')
