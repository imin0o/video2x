"""Common M2 execution API for CLI and future GUI; no shared mutable model/GPU cache."""
import ast
import copy
import json
import uuid
from pathlib import Path

from art_experiment_frames import video_signature
from art_experiment_run import execute, software_environment, validate_baseline
from art_model_inspect import ROOT, sha256
from art_probe_support import binary_inventory, gpu_inventory, save_json
from art_processing_config import configuration
from art_processing_rng import manifest
from art_processing_session import Session, checkpoint, is_cancellation


def processing_scripts():
    """Import closure of this API; CLI, probe and verification edits keep saved runs replayable."""
    found, pending = {}, [Path(__file__).stem]
    while pending:
        name = pending.pop()
        path = ROOT / 'scripts' / f'{name}.py'
        if name in found or not name.startswith('art_') or not path.is_file():
            continue
        found[name] = path
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module)
            elif isinstance(node, ast.Import):
                pending += [alias.name for alias in node.names]
    return list(found.values())


def environment(cli):
    return software_environment(processing_scripts()) | dict(gpu=gpu_inventory(),
                                                             binaries=binary_inventory(cli))


def completed_run(record):
    """Reject partial or malformed records before any output or GPU work."""
    engine = record.get('engine') if isinstance(record, dict) else None
    runs = engine.get('runs') if isinstance(engine, dict) else None
    if (not isinstance(runs, list) or not runs
            or any(not isinstance(r, dict) or 'pre_encode_frames' not in r for r in runs)
            or type(record.get('schema_version')) is not int or record['schema_version'] not in (2, 3)
            or record.get('status') != 'completed' or not isinstance(record.get('final_frames'), list)
            or not {'run_id', 'configuration', 'baseline', 'environment'} <= record.keys()):
        raise ValueError('Replay needs a completed M2 run.json')
    configuration(record['configuration'])
    if record['schema_version'] == 3 and not {'video_request', 'source_sha256', 'delivery'} <= record.keys():
        raise ValueError('M3 replay needs source and interval settings')
    if record['schema_version'] == 3:
        from art_video_source import validate_request
        validate_request(record['video_request'])
        if not isinstance(record['delivery'], dict) or not isinstance(record['delivery'].get('audio_verification'), list):
            raise ValueError('M3 replay needs decoded audio verification')
    return record


def load_run(path):
    return completed_run(json.loads(Path(path).read_text(encoding='utf-8')))


def without_baseline(engine):
    # The run record embeds the M0 baseline once at top level.
    return {key: value for key, value in engine.items() if key != 'baseline'}


def run(baseline, directory, values, cli, session=None, replay=None, video=None, keep_intermediates=False):
    """All input validation precedes output creation. Each pass uses a fresh CLI process."""
    config = configuration(values)
    if replay is not None:
        completed_run(replay)
    if video is not None or (replay and replay['schema_version'] == 3):
        from art_video_run import run_video
        return run_video(baseline, directory, values, cli, video or replay['video_request'], session, replay,
                         keep_intermediates)
    if keep_intermediates:
        raise ValueError('M2 runs always keep intermediates; the option applies to M3 video runs')
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
        if replay is not None and (config != replay['configuration'] or baseline != replay['baseline']
                                   or snapshot != replay['environment']):
            raise ValueError('Replay recipe, input or environment differs from the saved run')
        checkpoint()
        directory.mkdir(parents=True, exist_ok=False)
        record = dict(schema_version=2, run_id=str(uuid.uuid4()), status='running',
                      configuration=config, baseline=baseline, environment=snapshot,
                      rng=manifest(config['settings']), effective=effective,
                      isolation='fresh model bytes and child GPU process per pass; no model cache',
                      result=None)
        session.run_id, session.pass_count = record['run_id'], config['settings']['passes']
        path = directory / 'run.json'
        save_json(directory / 'recipe.json', config)
        save_json(path, record)
        try:
            checkpoint()
            engine = execute(baseline, directory / 'work', config['settings'], cli,
                             make_comparison=False, pin_tile=True)
            checkpoint()
            record['engine'] = without_baseline(engine)
            record['result'] = engine['result']
            record['result_sha256'] = sha256(Path(engine['result']))
            record['final_frames'] = video_signature(engine['result'])
            # Feature draws are in model.feature; weight draws live in the hashed model files.
            record['realized'] = dict(model=engine['model'], model_files=engine['runs'][0]['model_files'])
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
            record['status'] = 'cancelled' if is_cancellation(error) else 'failed'
            record['error'] = f'{type(error).__name__}: {error}'
            engine_path = directory / 'work/run.json'
            if 'engine' not in record and engine_path.is_file():
                record['engine'] = without_baseline(json.loads(engine_path.read_text(encoding='utf-8')))
            raise
        finally:
            record['timings'] = dict(sorted(session.timings.items()))
            save_json(path, record)
            session.finish(record['status'])
        return record
