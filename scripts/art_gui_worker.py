"""Serialize requests and deliver immutable snapshots across the Qt thread boundary."""
import copy
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from art_experiment_run import baseline_context
from art_processing import run
from art_processing_session import Session, is_cancellation
from art_video_source import request


class RenderThread(QThread):
    progress = Signal(str, object)
    outcome = Signal(str, object)

    def __init__(self, token, snapshot, directory, parent=None):
        super().__init__(parent)
        self.token, self.snapshot, self.directory = token, copy.deepcopy(snapshot), Path(directory)
        self.session = Session(on_progress=self.report)
        self.last_report, self.last_stage = 0, None

    def report(self, event):
        now = time.monotonic()
        if event['state'] != 'running' or event['stage'] != self.last_stage or now - self.last_report > 0.1:
            self.progress.emit(self.token, dict(event))
            self.last_report, self.last_stage = now, event['stage']

    def run(self):
        try:
            values = self.snapshot['controls']
            self.session.check()
            video = request(values['source'], float(values['start']),
                            float(values['end']) if values['end'].strip() else None,
                            int(values['output_scale']), values['audio'])
            baseline = baseline_context(Path(values['baseline_dir']), Path(values['cli']))
            self.session.check()
            result = run(baseline, self.directory, self.snapshot['configuration'], values['cli'],
                         video=video, session=self.session)
            outcome = dict(state='completed', record=result, snapshot=self.snapshot)
        except Exception as error:
            outcome = dict(state='cancelled' if is_cancellation(error) else 'failed',
                           error=f'{type(error).__name__}: {error}')
        self.outcome.emit(self.token, outcome | dict(directory=str(self.directory)))


class RenderController(QObject):
    progress = Signal(object)
    outcome = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.token = None
        self.cancelled = False

    def start(self, snapshot, directory):
        if self.worker is not None:
            raise RuntimeError('実行中の処理が終了するまでお待ちください')
        self.token, self.cancelled = uuid.uuid4().hex, False
        self.worker = RenderThread(self.token, snapshot, directory, self)
        self.worker.progress.connect(self.accept_progress)
        self.worker.outcome.connect(self.accept_outcome)
        self.worker.finished.connect(self.release)
        self.busy_changed.emit(True)
        self.worker.start()

    def accept_progress(self, token, event):
        if token == self.token and not self.cancelled:
            self.progress.emit(event)

    def accept_outcome(self, token, outcome):
        if token != self.token:
            return
        if self.cancelled and outcome['state'] == 'completed':
            # Work past the last checkpoint still finishes; keep it off the display but say it was saved.
            outcome = dict(state='cancelled', directory=outcome['directory'],
                           error=f"取消が間に合わず処理は完了しました。結果は保存済みですが、表示には採用していません: "
                                 f"{outcome['record']['result']}")
        self.outcome.emit(outcome)

    def cancel(self):
        if self.worker is not None:
            self.cancelled = True
            self.worker.session.cancel()

    def release(self):
        self.worker.deleteLater()
        self.worker = None
        self.busy_changed.emit(False)
