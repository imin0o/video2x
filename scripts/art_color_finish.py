"""Finish a verified raw M1 inference using source colors without rerunning the GPU."""
import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import PIL

from art_experiment_frames import transform
from art_experiment_model import MODEL_HASHES, recipe
from art_experiment_run import comparison
from art_model_inspect import MODEL, ROOT, sha256
from art_probe_support import save_json


def finish(run_path, directory, source_color=1, luma_change=0.5):
    parent = json.loads(run_path.read_text(encoding='utf-8'))
    if parent['status'] != 'completed' or parent['result'] != parent['runs'][-1]['output']:
        raise ValueError('Color finishing requires a completed raw inference without output transforms')
    settings = recipe(parent['recipe'] | dict(source_color=source_color, luma_change=luma_change))
    baseline = parent['baseline']
    clip = Path(baseline['prepared_input']['path'])
    original = Path(baseline['source']['path'])
    if sha256(clip) != baseline['prepared_input']['sha256'] or sha256(original) != baseline['source']['sha256']:
        raise ValueError('Source hash differs from the recorded inference')
    directory.mkdir(parents=True, exist_ok=False)
    record = parent | dict(status='running', recipe=settings, transformations=list(parent['transformations']),
                           parent_run=dict(path=str(run_path), sha256=sha256(run_path)))
    record.pop('comparison', None)
    record['finish_environment'] = dict(numpy=np.__version__, pyav=av.__version__, pillow=PIL.__version__,
                                       scripts={p.name: sha256(p) for p in (ROOT / 'scripts').glob('art_*.py')})
    manifest = directory / 'run.json'
    save_json(directory / 'recipe.json', settings)
    save_json(manifest, record)
    started = time.perf_counter()
    try:
        result = directory / 'result.mkv'
        evidence = transform(parent['result'], result, settings, original=clip, stage='output')
        record['transformations'].append(evidence)
        record['result'] = str(result)
        control = run_path.parent / 'injected.mkv' if settings['input_blur'] and not settings['input_noise'] else None
        record['comparison'] = comparison(baseline, result, directory, control)
        record['status'] = 'completed'
    except BaseException as error:
        record['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        preserved = (sha256(original) == baseline['source']['sha256']
                     and tuple(sha256(MODEL.with_suffix(s)) for s in ('.param', '.bin')) == MODEL_HASHES)
        record['original_hashes_preserved'] = preserved
        record['finish_wall_seconds'] = time.perf_counter() - started
        if not preserved:
            record['status'], record['error'] = 'failed', 'Original input/model integrity check failed'
        save_json(manifest, record)
    if not preserved:
        raise ValueError(record['error'])
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-color', type=float, default=1)
    parser.add_argument('--luma-change', type=float, default=0.5)
    args = parser.parse_args()
    result = finish(args.run.resolve(), args.output_dir.resolve(), args.source_color, args.luma_change)
    print(result['comparison']['video'])
