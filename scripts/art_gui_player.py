"""Frame-paired, bounded-memory comparison using the authoritative source timeline."""
import queue
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


def comparison_frames(record, stop):
    original = Path(record['result']).parent / 'input.mkv'
    with av.open(str(original)) as source, av.open(record['result']) as result:
        left, right = source.decode(video=0), result.decode(video=0)
        for item in record['source_timeline']:
            if stop.is_set():
                return
            pictures = []
            for frame in (next(left), next(right)):
                width = min(frame.width, 960)
                height = max(1, round(frame.height * width / frame.width))
                array = frame.reformat(width=width, height=height, format='rgb24').to_ndarray()
                pictures.append(QImage(array.data, width, height, array.strides[0], QImage.Format_RGB888).copy())
            yield pictures, float(Fraction(item['display_start'])), float(Fraction(item['display_end']))


class ComparisonPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.record, self.thread, self.pending, self.pictures = None, None, None, None
        self.stop = threading.Event()
        self.frames = queue.Queue(maxsize=2)
        self.playing = False
        layout, panes = QVBoxLayout(self), QHBoxLayout()
        self.labels = [QLabel('元映像'), QLabel('生成結果')]
        for label in self.labels:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(220, 180)
            label.setStyleSheet('background:#15191e; color:#cbd5e1; border:1px solid #39424e;')
            panes.addWidget(label, 1)
        layout.addLayout(panes, 1)
        self.stamp = QLabel('左：元映像 ／ 右：結果 · 同じ元時刻 · 比較再生は無音')
        layout.addWidget(self.stamp)
        row = QHBoxLayout()
        self.play_button, self.restart_button = QPushButton('再生 / 一時停止'), QPushButton('先頭へ')
        self.play_button.clicked.connect(self.toggle)
        self.restart_button.clicked.connect(self.restart)
        row.addWidget(self.play_button)
        row.addWidget(self.restart_button)
        layout.addLayout(row)
        self.play_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(10)

    def shutdown(self):
        self.timer.stop()
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.thread = None

    def load(self, record):
        self.shutdown()
        self.record = record
        self.restart()

    def restart(self):
        if self.record is None:
            return
        self.shutdown()
        self.frames = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.playing, self.pending, self.position = False, None, None
        self.play_button.setEnabled(True)
        self.restart_button.setEnabled(True)
        self.thread = threading.Thread(target=self.decode, daemon=True)
        self.thread.start()
        self.timer.start(10)

    def decode(self):
        def put(value):
            while not self.stop.is_set():
                try:
                    self.frames.put(value, timeout=.05)
                    return
                except queue.Full:
                    pass
        try:
            for item in comparison_frames(self.record, self.stop):
                if self.stop.is_set():
                    return
                put(item)
            put('end')
        except Exception as error:
            put(f'比較表示エラー: {error}')

    def toggle(self):
        if self.position is None:
            return
        if self.playing:
            self.position = min(self.end, self.position + time.monotonic() - self.anchor)
        self.anchor = time.monotonic()
        self.playing = not self.playing

    def tick(self):
        if self.record is None:
            return
        if self.pending is None:
            try:
                self.pending = self.frames.get_nowait()
            except queue.Empty:
                return
        if self.position is not None and not self.playing:
            return
        current = self.position + time.monotonic() - self.anchor if self.position is not None else None
        if isinstance(self.pending, str):
            if current is not None and current < self.end:
                return
            self.playing = False
            self.play_button.setEnabled(False)
            self.stamp.setText('再生終了 · 先頭へ戻して再生できます' if self.pending == 'end' else self.pending)
            self.pending = None
            return
        pictures, start, end = self.pending
        if current is not None and current < start:
            return
        # If decoding is slow, slow both panes together rather than allowing frame drift.
        self.position, self.anchor, self.end = start, time.monotonic(), end
        self.pictures = pictures
        self.show_pictures()
        self.stamp.setText(f'元時刻 {start:.6f} 秒 · 左：元映像 / 右：結果 · 無音')
        self.pending = None

    def show_pictures(self):
        for label, picture in zip(self.labels, self.pictures or ()):
            label.setPixmap(QPixmap.fromImage(picture).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        # A paused frame is only redrawn here, so rescale it to the new pane size.
        super().resizeEvent(event)
        self.show_pictures()
