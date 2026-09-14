"""Common M2 execution API for CLI and future GUI; no shared mutable model/GPU cache."""
import copy
import json
import platform
import sys
import uuid
from pathlib import Path

import av
import numpy as np
import PIL

from art_experiment_frames import video_signature
from art_experiment_run import execute, validate_baseline
from art_model_inspect import ROOT, sha256
from art_probe_support import binary_inventory, gpu_inventory, save_json
from art_processing_config import configuration
from art_processing_rng import manifest
from art_processing_session import Session, checkpoint


def environment(cli):
    return dict(os=platform.platform(), python=sys.version, numpy=np.__version__,
                pyav=av.__version__, pillow=PIL.__version__, gpu=gpu_inventory(),
                binaries=binary_inventory(cli),
                scripts={p.name: sha256(p) for p in (ROOT / 'scripts').glob('art_*.py')})


def load_run(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if (type(record.get('schema_version')) is not int or record['schema_version'] != 2
            or record.get('status') != 'completed'):
        raise ValueError('Replay needs a completed M2 run.json')
    configuration(record['configuration'])
    return record


def run(baseline, directory, values, cli, session=None, replay=None):
    """All input validation precedes output creation. Each pass uses a fresh CLI process."""
    config = configuration(values)
    baseline = copy.deepcopy(baseline)
    cli, directory = Path(cli).resolve(), Path(directory).resolve()
    session = session or Session()
    with session:
        checkpoint()
        validate_baseline(baseline, cli)
        effective = baseline['runs'][0]['effective']
        if (type(effective['tile']) is not int or effective['tile'] not in (32, 64, 100, 200)
                or effective['scale'] != 2 or effective['prepadding'] != 10 or effective['tta']):
            raise ValueError('Unsupported inference settings')
        snapshot = environment(cli)
        if replay is not None:
            if (replay.get('schema_version') != 2 or replay.get('status') != 'completed'
                    or config != replay['configuration'] or baseline != replay['baseline']
                    or snapshot != replay['environment']):
                raise ValueError('Replay recipe, input or environment differs from the saved run')
        checkpoint()
        directory.mkdir(parents=True, exist_ok=False)
        record = dict(schema_version=2, run_id=str(uuid.uuid4()), status='running',
                      configuration=config, baseline=baseline, environment=snapshot,
                      rng=manifest(config['settings']), effective=effective,
                      isolation='fresh model bytes and child GPU process per pass; no model cache',
                      result=None)
        path = directory / 'run.json'
        save_json(directory / 'recipe.json', config)
        save_json(path, record)
        try:
            checkpoint()
            engine = execute(baseline, directory / 'work', config['settings'], cli,
                             make_comparison=False, pin_tile=True)
            checkpoint()
            record['engine'] = engine
            record['result'] = engine['result']
            record['result_sha256'] = sha256(Path(engine['result']))
            record['final_frames'] = video_signature(engine['result'])
            record['realized'] = dict(model=engine['model'],
                                     model_files=engine['runs'][0]['model_files'],
                                     settings=copy.deepcopy(config['settings']))
            if replay is not None:
                matches = (record['final_frames'] == replay['final_frames']
                           and [r['pre_encode_frames'] for r in engine['runs']]
                           == [r['pre_encode_frames'] for r in replay['engine']['runs']])
                record['replay'] = dict(parent_run_id=replay['run_id'], exact_frames_match=matches)
                if not matches:
                    raise ValueError('Replay pre-encode frame hashes differ; A04 is not satisfied')
            checkpoint()
            record['status'] = 'completed'
        except BaseException as error:
            record['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
            record['error'] = f'{type(error).__name__}: {error}'
            engine_path = directory / 'work/run.json'
            if engine_path.is_file():
                record['engine'] = json.loads(engine_path.read_text(encoding='utf-8'))
            raise
        finally:
            save_json(path, record)
        return record
