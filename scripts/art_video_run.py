"""M3 lifecycle: source intervals, delivery, durable progress and replay."""
import json
import time
import uuid
from pathlib import Path

from art_experiment_run import FFMPEG, execute, validate_baseline
from art_model_inspect import sha256
from art_probe_support import save_json
from art_processing_config import configuration
from art_processing_rng import manifest
from art_processing_session import Session, checkpoint, is_cancellation, progress
from art_video_output import export_video
from art_video_source import prepare, validate_request


def run_video(baseline, directory, values, cli, video, session=None, replay=None):
    from art_processing import environment, without_baseline

    config = configuration(values)
    spec = validate_request(video)
    cli, directory = Path(cli).resolve(), Path(directory).resolve()
    session = session or Session()
    with session:
        checkpoint()
        validate_baseline(baseline, cli)
        effective = baseline['runs'][0]['effective']
        if (type(effective['tile']) is not int or effective['tile'] not in (32, 64, 100, 200)
                or effective['scale'] != 2 or effective['prepadding'] != 10 or effective['tta']):
            raise ValueError('Unsupported inference settings')
        snapshot = environment(cli) | dict(ffmpeg_sha256=sha256(FFMPEG))
        source_hash = sha256(Path(spec['source']))
        if replay and (config != replay['configuration'] or baseline != replay['baseline']
                       or snapshot != replay['environment'] or spec != replay['video_request']
                       or source_hash != replay['source_sha256']):
            raise ValueError('Replay recipe, source or environment differs from the saved run')
        directory.mkdir(parents=True, exist_ok=False)
        record = dict(schema_version=3, run_id=str(uuid.uuid4()), status='running', result=None,
                      configuration=config, baseline=baseline, environment=snapshot, effective=effective,
                      video_request=spec, source_sha256=source_hash, rng=manifest(config['settings']))
        path = directory / 'run.json'
        save_json(directory / 'recipe.json', config)
        save_json(path, record)
        started, last_save = time.perf_counter(), 0
        callback = session.on_progress

        def report(event):
            nonlocal last_save
            record['progress'] = event
            if time.perf_counter() - last_save >= 0.5:
                save_json(path, record)
                last_save = time.perf_counter()
            if callback:
                callback(event)

        session.on_progress = report
        try:
            progress('prepare-source')
            prepared, timeline, tail = prepare(spec, directory, baseline)
            record['prepare_seconds'] = time.perf_counter() - started
            record['prepared_source'] = prepared['source']
            record['source_timeline'] = timeline
            final_retain = spec['output_scale'] == 1 and config['settings']['retain'] != 0
            engine_settings = config['settings'] | ({'retain': 0} if final_retain else {})
            engine = execute(prepared, directory / 'work', engine_settings, cli,
                             make_comparison=False, pin_tile=True)
            record['engine'] = without_baseline(engine)
            record['realized'] = dict(model=engine['model'], model_files=engine['runs'][0]['model_files'])
            original = prepared['prepared_input']['path'] if final_retain else None
            delivery = export_video(engine['result'], spec, timeline, tail, directory,
                                    settings=config['settings'], original=original)
            delivery['retain_stage'] = 'after output resize' if final_retain else 'inference resolution'
            record['delivery'] = delivery
            record['final_frames'] = delivery['pre_encode_frames']
            if replay:
                equal = (record['final_frames'] == replay['final_frames']
                         and delivery['audio_verification'] == replay['delivery']['audio_verification']
                         and [r['pre_encode_frames'] for r in engine['runs']]
                         == [r['pre_encode_frames'] for r in replay['engine']['runs']])
                record['replay'] = dict(parent_run_id=replay['run_id'], exact_frames_match=equal)
                if not equal:
                    raise ValueError('Replay pre-encode frame hashes differ; A04 is not satisfied')
            if sha256(Path(spec['source'])) != source_hash:
                raise ValueError('Original source changed during processing')
            progress('verified', len(timeline), len(timeline))
            checkpoint()
            pending = Path(delivery['pending'])
            record['result_sha256'] = sha256(pending)
            checkpoint()
            pending.rename(directory / 'result.mkv')
            record['result'], record['status'] = str(directory / 'result.mkv'), 'completed'
        except BaseException as error:
            record['status'] = 'cancelled' if is_cancellation(error) else 'failed'
            record['error'] = f'{type(error).__name__}: {error}'
            engine_path = directory / 'work/run.json'
            if 'engine' not in record and engine_path.is_file():
                record['engine'] = without_baseline(json.loads(engine_path.read_text(encoding='utf-8')))
            raise
        finally:
            session.on_progress = callback
            record['wall_seconds'] = time.perf_counter() - started
            if session.cancel_requested_at is not None:
                record['cancel_response_seconds'] = time.perf_counter() - session.cancel_requested_at
            save_json(path, record)
            # Same-name sidecar is authoritative even for failed/partial exports.
            save_json(directory / ('result.json' if record['status'] == 'completed' else 'result.partial.json'), record)
        return record
