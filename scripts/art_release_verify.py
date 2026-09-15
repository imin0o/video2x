"""Run M5 verification against the installed personal Windows configuration."""
import argparse
import subprocess
import sys
from pathlib import Path

from art_model_inspect import ROOT
from art_probe_support import save_json
from art_processing_session import NO_WINDOW
from art_release_report import inventory, write_report


def verify(output, baseline, cli):
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', completed=[])
    before = None
    try:
        before = inventory(cli, baseline)
        save_json(output / 'environment.json', before)
        suites = [('processing', 'art_processing_verify.py', []), ('video', 'art_video_verify.py', []),
                  ('gui', 'art_gui_verify.py', []),
                  ('workflow', 'art_release_gui.py', ['--phase', 'create']),
                  ('workflow', 'art_release_gui.py', ['--phase', 'restart'])]
        for name, script, extra in suites:
            label = name + ('-' + extra[-1] if extra else '')
            print(f'M5: {label}', flush=True)
            invocation = [sys.executable, '-B', str(ROOT / 'scripts' / script),
                          '--output-dir', str(output / name), '--baseline-dir', str(baseline)]
            invocation += ['--cli', str(cli)] + extra
            with (output / f'{label}.log').open('w', encoding='utf-8') as log:
                subprocess.run(invocation, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, creationflags=NO_WINDOW)
            state['completed'].append(label)
            save_json(output / 'verification.json', state)
        after = inventory(cli, baseline)
        if before != after:
            save_json(output / 'environment-after.json', after)
            raise ValueError('Source, model, scripts, binaries or environment changed during verification')
        state['originals_and_environment_preserved'] = True
        report = write_report(output)
        state['status'] = report['status']
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_json(output / 'summary.json', state)
        raise
    finally:
        save_json(output / 'verification.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'build/art/m2-baseline-2')
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    args = parser.parse_args()
    verify(args.output_dir.resolve(), args.baseline_dir.resolve(), args.cli.resolve())
