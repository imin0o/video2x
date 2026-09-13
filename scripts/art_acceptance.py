"""Rebuild the M1 acceptance record from the tracked creator decision and re-verified run files."""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from art_experiment_frames import frames, video_signature
from art_experiment_model import MODEL_HASHES
from art_model_inspect import MODEL, ROOT, sha256
from art_probe_support import save_json

DECISION = ROOT / 'docs/art-tool-m1-acceptance.json'
WEIGHT_KEYS = ('weight_strength', 'weight_layers')


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def deferred_runs(decision):
    """Expand deferred groups to run records; every group keeps a reason and who gave it."""
    items = []
    for group in decision['deferred']:
        if not group['reason'].strip() or group['reason_source'] not in ('creator', 'record'):
            raise ValueError(f'Deferred group needs a reason and reason_source: {group["directory"]}')
        names = group.get('candidates') or [c['name'] for c in load(Path(group['directory']) / 'review.json')['candidates']]
        for name in names:
            run = Path(group['directory']) / name / 'run.json'
            if not (ROOT / run).is_file():
                raise ValueError(f'Deferred candidate has no run record: {run}')
            items.append(dict(run=run.as_posix(), reason=group['reason'], reason_source=group['reason_source']))
    adopted = [item['run'] for item in decision['adopted']]
    runs = adopted + [item['run'] for item in items]
    if not adopted or len(set(runs)) != len(runs):
        raise ValueError('Adopted runs must exist and must not also be deferred')
    return items


def verified_result(path):
    """Decode the final file now and require the pixels/times recorded before encoding."""
    run = load(path)
    final = run['transformations'][-1] if run['transformations'] else None
    if (run['status'] != 'completed' or run.get('original_hashes_preserved') is not True
            or final is None or run['result'] == run['runs'][-1]['output']):
        raise ValueError(f'A completed run with a verified output transform is required: {path}')
    signature, expected = video_signature(run['result']), run['verification']
    if (signature != final['pre_encode_frames'] or len(signature) != expected['frame_count']
            or signature[-1]['time'] != expected['tail_time']
            or any([f['width'], f['height']] != expected['output_dimensions'] for f in signature)):
        raise ValueError(f'Result file differs from its record: {path}')
    parent = run.get('parent_run')
    return run, dict(frames=len(signature), tail=signature[-1]['time'], dimensions=expected['output_dimensions'],
                     file_matches_recorded_pixels_and_times=True, source_color=run['recipe']['source_color'],
                     parent_record_unchanged=sha256(Path(parent['path'])) == parent['sha256'] if parent else None)


def sampled(path, indices):
    result = {}
    stream = frames(path)
    try:
        for index, frame in enumerate(stream):
            if index in indices:
                result[index] = frame.to_ndarray(format='bgr24').astype(np.int16)
            if len(result) == len(indices):
                break
    finally:
        stream.close()
    return result


def ablation(run, control, count):
    """Same blur and color finishing, distributed weights: isolates the weight-mutation share."""
    same = {k: v for k, v in run['recipe'].items() if k not in WEIGHT_KEYS}
    if control['recipe']['weight_strength'] != 0 or same != {
            k: v for k, v in control['recipe'].items() if k not in WEIGHT_KEYS}:
        raise ValueError('Control must differ from the adopted recipe only by zero weight mutation')
    indices = {0, count // 2, count - 1}
    a, b = sampled(run['result'], indices), sampled(control['result'], indices)
    differences = [np.abs(a[i] - b[i]) for i in sorted(indices)]
    return dict(sample_indices=sorted(indices),
                mean_absolute_difference=float(np.mean([d.mean() for d in differences])),
                pixels_with_channel_difference_16_or_more=float(np.mean([(d.max(axis=-1) >= 16).mean()
                                                                        for d in differences])))


def page(directory, record, decision):
    def link(path):
        return html.escape(quote(Path(os.path.relpath(ROOT / path, directory)).as_posix()))
    cards = []
    for item in decision['adopted']:
        for label, run in ((f'{item["name"]} ({item["role"]})', item['run']),
                           (f'{item["name"]}: weight mutation 0 control', item['control'])):
            folder = str(Path(run).parent)
            cards.append(f'<article><h2>{html.escape(label)}</h2><video controls loop muted preload="none" '
                         f'poster="{link(folder + "/comparison.png")}" src="{link(folder + "/comparison.mp4")}"></video>'
                         f'<p><a href="{link(folder + "/recipe.json")}">Recipe</a></p></article>')
    rows = ''.join(f'<tr><td>{html.escape(x["run"])}</td><td>{html.escape(x["reason"])}</td>'
                   f'<td>{x["reason_source"]}</td></tr>' for x in record['deferred'])
    (directory / 'index.html').write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>M1 acceptance</title>'
        '<style>body{background:#151515;color:#eee;font:16px system-ui;margin:24px}video{width:100%}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}'
        'a{color:#9bcaff}td{border-top:1px solid #444;padding:4px 8px;vertical-align:top}</style>'
        f'<h1>M1 acceptance</h1><p>Creator feedback ({decision["date"]}): {html.escape(decision["creator_feedback"])}</p>'
        '<p>Top: input / normal. Bottom: same input blur / candidate. '
        '<a href="review.json">Generated record</a></p><main>' + ''.join(cards) + '</main>'
        f'<h2>Deferred</h2><table>{rows}</table></html>', encoding='utf-8')


def build(directory, decision_path=DECISION):
    decision = load(decision_path)
    deferred = deferred_runs(decision)
    verification, effects = {}, {}
    for item in decision['adopted']:
        run, verification[item['name']] = verified_result(item['run'])
        if run['recipe']['source_color'] != 1:
            raise ValueError(f'Adopted output must keep source colors: {item["name"]}')
        pilot = load(item['pilot'])['transformations'][-1]['pre_encode_frames']
        verification[item['name']]['pilot_prefix_matches'] = run['transformations'][-1]['pre_encode_frames'][:len(pilot)] == pilot
        if not verification[item['name']]['pilot_prefix_matches']:
            raise ValueError(f'Pilot prefix differs: {item["name"]}')
        control, _ = verified_result(item['control'])
        effects[item['name']] = ablation(run, control, verification[item['name']]['frames'])
    if tuple(sha256(MODEL.with_suffix(s)) for s in ('.param', '.bin')) != MODEL_HASHES:
        raise ValueError('Distributed model hash differs')
    record = dict(status='completed_accepted', generated_by='scripts/art_acceptance.py',
                  decision=dict(path=Path(os.path.relpath(decision_path, ROOT)).as_posix(), sha256=sha256(decision_path)),
                  date=decision['date'], creator_feedback=decision['creator_feedback'],
                  constraints=decision['constraints'], acceptance=decision['acceptance'],
                  adopted=decision['adopted'], deferred=deferred, verification=verification,
                  weight_mutation_ablation=effects, distributed_model_hashes_preserved=True)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / 'review.json', record)
    page(directory, record, decision)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'build/art/m1-source-color')
    parser.add_argument('--decision', type=Path, default=DECISION)
    args = parser.parse_args()
    result = build(args.output_dir.resolve(), args.decision.resolve())
    print(json.dumps(result['weight_mutation_ablation'], indent=2))
