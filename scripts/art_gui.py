"""M4 desktop entry point; all inference uses the shared M2/M3 API."""
import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSplitter,
                               QTextEdit, QVBoxLayout, QWidget)

from art_gui_form import SettingsForm
from art_gui_paths import PathsForm
from art_gui_player import ComparisonPlayer
from art_gui_settings import document, load_document, save_document
from art_gui_worker import RenderController
from art_model_inspect import ROOT


class ArtWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle('Video2X Art · M4')
        self.resize(1280, 900)
        self.result_snapshot, self.last_directory, self.closing = None, None, False
        self.form, self.paths = SettingsForm(), PathsForm(args)
        self.player, self.controller = ComparisonPlayer(), RenderController(self)
        self.form.changed.connect(self.changed)
        self.paths.changed.connect(self.changed)
        central, layout = QWidget(), QHBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)
        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split)
        split.addWidget(self.form)
        right, body = QWidget(), QVBoxLayout()
        right.setLayout(body)
        split.addWidget(right)
        body.addWidget(self.paths)
        row = QHBoxLayout()
        self.buttons = {}
        for name, label, action in [('load', '設定読込', self.load_settings), ('save', '設定保存', self.save_settings),
                                    ('preview', '区間プレビュー', lambda: self.start(False)),
                                    ('export', '本番書出し', lambda: self.start(True)),
                                    ('cancel', '取消', self.cancel), ('folder', '出力先を開く', self.open_output)]:
            button = QPushButton(label)
            button.clicked.connect(action)
            row.addWidget(button)
            self.buttons[name] = button
        body.addLayout(row)
        body.addWidget(QLabel('プレビュー・本番とも指定区間を処理します。全編は開始0・終了空欄。音声 pcm はPCM変換、omit は除外。'))
        self.dirty = QLabel('生成結果はまだありません')
        body.addWidget(self.dirty)
        body.addWidget(self.player, 1)
        self.result_info = QTextEdit()
        self.result_info.setReadOnly(True)
        self.result_info.setMaximumHeight(110)
        self.result_info.setPlaceholderText('最後に生成した結果の設定と出力先')
        body.addWidget(self.result_info)
        self.progress = QProgressBar()
        body.addWidget(self.progress)
        self.status = QLabel('準備完了 · 元動画と基準記録を選択してください')
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.addWidget(self.status)
        self.controller.progress.connect(self.on_progress)
        self.controller.outcome.connect(self.on_outcome)
        self.controller.busy_changed.connect(self.on_busy)
        self.on_busy(False)
        self.buttons['folder'].setEnabled(False)
        if args.recipe:
            self.load_settings(args.recipe)

    def snapshot(self):
        return document(self.form.config(), self.paths.values())

    def changed(self):
        if self.result_snapshot is None:
            return
        try:
            same = self.snapshot() == self.result_snapshot
        except ValueError:
            same = False
        self.dirty.setText('表示中の結果と現在の設定は一致' if same else '設定変更あり · 表示中の結果には未反映')

    def load_settings(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'レシピ / GUI設定を読込', '', 'JSON (*.json)')
        if not path:
            return
        try:
            config, controls = load_document(path)
            self.form.set_config(config)
            if controls:
                self.paths.set_values(controls)
            self.status.setText(f'設定を読み込みました: {path}')
        except (OSError, ValueError, KeyError) as error:
            self.show_error(error)

    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, '新しいファイルへ設定保存', '', 'JSON (*.json)')
        if path:
            try:
                save_document(path, self.form.config(), self.paths.values())
                self.status.setText(f'設定を保存しました: {path}')
            except (OSError, ValueError) as error:
                self.show_error(error)

    def start(self, production):
        try:
            snapshot = self.snapshot()
            values = snapshot['controls']
            if not values['source'].strip() or not values['output_root'].strip():
                raise ValueError('元動画と保存先を指定してください')
            name = ('export-' if production else 'preview-') + datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
            directory = Path(values['output_root']).resolve() / name
            self.controller.start(snapshot, directory)
            self.status.setText(f'開始準備中 · 出力先: {directory}')
        except (OSError, ValueError, RuntimeError) as error:
            self.show_error(error)

    def cancel(self):
        self.controller.cancel()
        self.buttons['cancel'].setEnabled(False)
        self.status.setText('取消要求済み · 処理の終了を待っています')

    def on_busy(self, busy):
        for key in ('preview', 'export'):
            self.buttons[key].setEnabled(not busy)
        self.buttons['cancel'].setEnabled(busy)
        if self.closing and not busy:
            self.close()

    def on_progress(self, event):
        total, done = event['total'], event['completed']
        self.progress.setRange(0, total if total and event['total_exact'] else 0)
        if done is not None:
            self.progress.setValue(done)
        self.status.setText(f"{event['stage']} · {done if done is not None else '?'} / {total or '?'} · {event['elapsed_seconds']:.1f}秒")

    def on_outcome(self, outcome):
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if outcome['state'] == 'completed' else 0)
        self.last_directory = outcome['directory']
        self.buttons['folder'].setEnabled(Path(self.last_directory).is_dir())
        state = {'completed': '完了', 'cancelled': '取消済み', 'failed': '失敗'}[outcome['state']]
        self.status.setText(f"{state} · {outcome.get('error', '')}\n出力先: {self.last_directory}")
        if outcome['state'] == 'completed':
            self.result_snapshot = outcome['snapshot']
            self.result_info.setPlainText(outcome['record']['result'] + '\n'
                                         + json.dumps(self.result_snapshot, ensure_ascii=False, indent=2))
            self.player.load(outcome['record'])
            self.changed()

    def open_output(self):
        if self.last_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_directory))

    def show_error(self, error):
        self.status.setText(f'エラー: {error}')
        QMessageBox.warning(self, '設定を確認してください', str(error))

    def closeEvent(self, event):
        if self.controller.worker is not None:
            self.closing = True
            self.cancel()
            event.ignore()
        else:
            self.player.shutdown()
            event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'build/art/m2-baseline-2')
    parser.add_argument('--cli', type=Path, default=ROOT / 'build/art/install/bin/video2x.exe')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = ArtWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
