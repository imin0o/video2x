"""Exercise the real GUI controller and GPU without requiring desktop input automation."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase

from art_gui import ArtWindow
from art_gui_settings import save_document
from art_model_inspect import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'build/art/m2-baseline-2')
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    app = QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/meiryo.ttc')
    app.setFont(QFont('Meiryo', 9))
    options = argparse.Namespace(input=ROOT / 'test/test.mp4',
                                 recipe=ROOT / 'build/art/m1-source-color/melt-16/recipe.json',
                                 baseline_dir=args.baseline_dir, cli=args.cli)
    window = ArtWindow(options)
    window.show()
    outcomes, heartbeats = [], 0
    window.controller.outcome.connect(outcomes.append)

    def wait(predicate, timeout=180):
        nonlocal heartbeats
        deadline = time.monotonic() + timeout
        while not predicate():
            app.processEvents()
            heartbeats += 1
            if time.monotonic() > deadline:
                window.controller.cancel()
                raise TimeoutError('GUI verification timed out')
            time.sleep(.01)
        app.processEvents()

    def render():
        window.start(False)
        wait(lambda: window.controller.worker is None)
        if outcomes[-1]['state'] != 'completed':
            raise RuntimeError(outcomes[-1])
        return outcomes[-1]['record']

    try:
        paths = window.paths.widgets
        paths['output_root'].setText(str(args.output_dir))
        paths['start'].setText('0.017')
        paths['end'].setText('0.05')
        window.form.widgets['input_noise'].setValue(6)
        window.form.widgets['time_mode'].setCurrentText('smooth')
        saved = window.snapshot()
        settings_path = args.output_dir / 'settings.json'
        save_document(settings_path, window.form.config(), window.paths.values())
        preview = render()
        wait(lambda: window.player.position is not None)
        window.grab().save(str(args.output_dir / 'gui.png'))
        window.close()
        options.recipe = settings_path
        window = ArtWindow(options)
        window.controller.outcome.connect(outcomes.append)
        window.show()
        assert window.snapshot() == saved
        replay = render()
        assert replay['final_frames'] == preview['final_frames']
        paths = window.paths.widgets
        paths['start'].setText('0')
        paths['end'].setText('0.08')
        window.start(True)
        wait(lambda: window.controller.worker is None)
        assert outcomes[-1]['state'] == 'completed', outcomes[-1]
        production = outcomes[-1]['record']
        times = [item['time'] for item in production['source_timeline']]
        indexes = [times.index(item['time']) for item in preview['source_timeline']]
        # Signatures include ordinal PTS: compare the pixels at corresponding source times.
        assert [production['final_frames'][i]['sha256'] for i in indexes] == [f['sha256'] for f in preview['final_frames']]
        paths['end'].setText('1')
        cancelled_at = []

        def cancel_inference(event):
            if event['stage'] == 'inference' and not cancelled_at:
                cancelled_at.append(time.monotonic())
                window.cancel()

        window.controller.progress.connect(cancel_inference)
        window.start(False)
        wait(lambda: window.controller.worker is None)
        assert outcomes[-1]['state'] == 'cancelled', outcomes[-1]
        cancelled = json.loads((Path(outcomes[-1]['directory'])/'run.json').read_text(encoding='utf-8'))
        assert cancelled['status'] == 'cancelled' and cancelled['source_preserved']
        window.controller.progress.disconnect(cancel_inference)
        paths['source'].setText(str(args.output_dir / 'missing.mp4'))
        window.start(False)
        wait(lambda: window.controller.worker is None)
        assert outcomes[-1]['state'] == 'failed'
        window.load_settings(settings_path)
        recovery = render()
        assert recovery['final_frames'] == preview['final_frames']
        summary = dict(status='passed', preview=preview['result'], production=production['result'],
                       preview_frames=len(preview['final_frames']), production_frames=len(production['final_frames']),
                       exact_interval_pixels=True, reload_exact_pixels=True, cancellation=True,
                       cancellation_record_preserved=cancelled['source_preserved'], failure_recovery=True,
                       qt_event_loop_iterations=heartbeats)
        (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
    finally:
        if window.controller.worker:
            window.controller.cancel()
            wait(lambda: window.controller.worker is None)
        window.close()


if __name__ == '__main__':
    main()
