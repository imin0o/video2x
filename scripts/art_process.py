"""M2: execute portable recipes, migrate M1 settings, or replay a saved video run."""
import argparse
import json
from pathlib import Path

from art_experiment_run import baseline_context
from art_model_inspect import ROOT
from art_probe_support import save_json
from art_processing import load_run, run
from art_processing_config import DEFAULTS, configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--recipe', type=Path)
    group.add_argument('--replay', type=Path)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--save-recipe', type=Path, help='Validate/migrate and save without processing')
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    for key, default in DEFAULTS.items():
        if key == 'version':
            continue
        parser.add_argument('--' + key.replace('_', '-'), dest=key,
                            type=str if isinstance(default, list) else type(default), default=None,
                            help='Override recipe value' + (' (comma-separated layers)' if isinstance(default, list) else ''))
    args = parser.parse_args()
    overrides = {key: getattr(args, key) for key in DEFAULTS if key != 'version' and getattr(args, key) is not None}
    if 'weight_layers' in overrides:
        overrides['weight_layers'] = [layer.strip() for layer in overrides['weight_layers'].split(',')]
    if args.replay and (overrides or args.baseline_dir or args.save_recipe):
        parser.error('--replay uses saved settings and input; overrides are not allowed')
    if args.save_recipe and (args.output_dir or args.baseline_dir):
        parser.error('--save-recipe does not process; --output-dir and --baseline-dir are not used')
    if not args.save_recipe and not args.output_dir:
        parser.error('--output-dir is required for processing')
    if args.recipe and not args.save_recipe and not args.baseline_dir:
        parser.error('--baseline-dir is required with --recipe')
    if args.output_dir and args.output_dir.exists():
        parser.error('--output-dir must name a new directory')
    try:
        replay = load_run(args.replay) if args.replay else None
        config = replay['configuration'] if replay else configuration(json.loads(args.recipe.read_text(encoding='utf-8')))
        config = configuration(config | dict(settings=config['settings'] | overrides))
        if args.save_recipe:
            # Never overwrite a source recipe or an existing saved recipe.
            with args.save_recipe.open('x', encoding='utf-8') as output:
                json.dump(config, output, indent=2, ensure_ascii=False)
            return
        baseline = replay['baseline'] if replay else baseline_context(args.baseline_dir.resolve(), args.cli.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.error(f'{type(error).__name__}: {error}')
    try:
        result = run(baseline, args.output_dir, config, args.cli, replay=replay)
    except (OSError, ValueError) as error:
        if args.output_dir.exists():
            raise  # Processing started; run.json records the failure.
        parser.error(f'{type(error).__name__}: {error}')
    save_json(args.output_dir / 'result.json', dict(video=result['result'], run=str((args.output_dir / 'run.json').resolve())))
    print(f'Verified video: {result["result"]}', flush=True)


if __name__ == '__main__':
    main()
