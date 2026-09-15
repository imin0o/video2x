"""M5 GUI workflow in two independent Python processes, with real GPU output."""
import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase

from art_experiment_model import model_copy
from art_gui import ArtWindow
from art_model_inspect import MODEL, ROOT
from art_probe_support import save_json
from art_video_verify import indexed


def workflow(output, baseline, cli, phase):
    app = QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/meiryo.ttc')
    app.setFont(QFont('Meiryo', 9))
    settings = output / 'settings.json'
    options = argparse.Namespace(input=ROOT / 'test/test.mp4', baseline_dir=baseline, cli=cli,
                                 recipe=settings if phase == 'restart' else
                                 ROOT / 'build/art/m1-source-color/melt-16/recipe.json')
    window = ArtWindow(options)
    window.show()
    outcomes = []
    window.controller.outcome.connect(outcomes.append)

    def wait():
        deadline = time.monotonic() + 300
        while window.controller.worker is not None:
            app.processEvents()
            if time.monotonic() > deadline:
                window.cancel()
                raise TimeoutError('GUI render exceeded 300 seconds')
            time.sleep(.01)
        app.processEvents()

    def render(production=False, expected='completed'):
        count = len(outcomes)
        window.start(production)
        wait()
        if len(outcomes) != count + 1 or outcomes[-1]['state'] != expected:
            raise AssertionError(f'Expected {expected}: {outcomes[count:]}')
        return outcomes[-1]

    try:
        if phase == 'create':
            paths = window.paths.widgets
            paths['output_root'].setText(str(output))
            paths['start'].setText('0.017')
            paths['end'].setText('0.05')
            trials = []
            for strength in (0, 6, 12):
                window.form.widgets['input_noise'].setValue(strength)
                trials.append(render()['record'])
            if len({tuple(f['sha256'] for f in r['final_frames']) for r in trials}) != 3:
                raise AssertionError('A03 strengths did not produce different pixels')
            if len({r['configuration']['settings']['seed'] for r in trials}) != 1:
                raise AssertionError('A03 seed changed')
            window.form.widgets['input_noise'].setValue(6)
            # The real save action; only the native file dialog result is supplied.
            with patch('art_gui.QFileDialog.getSaveFileName', return_value=(str(settings), 'JSON (*.json)')), \
                    patch('art_gui.QMessageBox.warning') as dialog:
                window.save_settings()
            if dialog.called or not settings.exists():
                raise AssertionError(f'GUI settings were not saved: {window.status.text()}')
            returned = render()['record']
            if returned['final_frames'] != trials[1]['final_frames']:
                raise AssertionError('A03 selected strength could not be restored')
            save_json(output / 'create.json', dict(status='passed', pid=os.getpid(),
                      snapshot=window.snapshot(), record=str(Path(returned['result']).with_name('run.json')),
                      strength_trials=[r['result'] for r in trials], A03=True))
            return

        original = json.loads((output / 'create.json').read_text(encoding='utf-8'))
        reference = json.loads(Path(original['record']).read_text(encoding='utf-8'))
        if window.snapshot() != original['snapshot']:
            raise AssertionError('Restart did not restore all GUI controls')
        replay = render()['record']
        if (replay['final_frames'] != reference['final_frames']
                or replay['delivery']['audio_verification'] != reference['delivery']['audio_verification']):
            raise AssertionError('Restart output differs')
        # Exercise the real error dialog call without blocking unattended verification.
        invalid = output / 'invalid-recipe.json'
        save_json(invalid, dict(seed=-1))
        before = window.snapshot()
        with patch('art_gui.QMessageBox.warning') as dialog:
            window.load_settings(invalid)
            if dialog.call_count != 1 or window.snapshot() != before:
                raise AssertionError('Invalid recipe was not rejected before changing controls')
            invalid_error = str(dialog.call_args.args[-1])

        def missing_copy(*args, **kwargs):
            base, record = model_copy(*args, **kwargs)
            base.with_name(MODEL.with_suffix('.bin').name).unlink()  # Only this run's generated model copy.
            return base, record

        with patch('art_experiment_run.model_copy', side_effect=missing_copy):
            missing = render(expected='failed')
        if 'missing or differs' not in missing['error'] or missing['error'] not in window.status.text():
            raise AssertionError('Model failure cause was not displayed')
        if render()['record']['final_frames'] != reference['final_frames']:
            raise AssertionError('Model failure contaminated the next run')
        window.paths.widgets['start'].setText('0')
        window.paths.widgets['end'].setText('0.25')
        production = render(True)['record']
        if indexed(reference) != {k: v for k, v in indexed(production).items() if k in indexed(reference)}:
            raise AssertionError('Restarted GUI preview differs from production')
        window.grab().save(str(output / 'gui.png'))
        save_json(output / 'summary.json', dict(status='passed', pid=os.getpid(),
                  independent_process_restart=True, controls_restored=True, exact_replay=True,
                  invalid_recipe_error=invalid_error, missing_model=missing,
                  failure_recovery=True, preview_production_exact=True, preview=replay['result'],
                  production=production['result'], production_frames=len(production['final_frames'])))
    finally:
        if window.controller.worker is not None:
            window.cancel()
            # Keep Qt alive until the owned thread has stopped, including timeout failures.
            while window.controller.worker is not None:
                app.processEvents()
                time.sleep(.01)
        window.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--phase', choices=('create', 'restart'), required=True)
    args = parser.parse_args()
    if args.phase == 'create':
        args.output_dir.mkdir(parents=True, exist_ok=False)
    workflow(args.output_dir.resolve(), args.baseline_dir.resolve(), args.cli.resolve(), args.phase)
