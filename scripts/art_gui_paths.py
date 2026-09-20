"""Source, interval and Windows execution paths."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLineEdit, QPushButton, QWidget)

from art_gui_form import ComboBox
from art_gui_settings import FIELDS
from art_model_inspect import ROOT


class PathsForm(QWidget):
    changed = Signal()

    def __init__(self, args):
        super().__init__()
        layout = QFormLayout(self)
        self.widgets = {}
        defaults = dict(source=str(args.input or ''), start='0', end='5', output_scale='1', audio='pcm',
                        baseline_dir=str(args.baseline_dir), cli=str(args.cli),
                        output_root=str(ROOT / 'build/art/gui'))
        labels = ('元動画', 'プレビュー開始（秒）', 'プレビュー終了（秒・空欄は末尾）', '出力倍率', '音声',
                  '基準記録フォルダー', 'Video2X実行ファイル', '保存先フォルダー')
        for key, label in zip(FIELDS, labels):
            row = QHBoxLayout()
            if key in ('output_scale', 'audio'):
                widget = ComboBox()
                widget.addItems(['1', '2'] if key == 'output_scale' else ['pcm', 'omit'])
                widget.setCurrentText(defaults[key])
                widget.currentTextChanged.connect(self.changed)
            else:
                widget = QLineEdit(defaults[key])
                widget.textChanged.connect(self.changed)
            widget.setAccessibleName(label)
            widget.setObjectName(key)
            self.widgets[key] = widget
            row.addWidget(widget)
            if key in ('source', 'baseline_dir', 'cli', 'output_root'):
                button = QPushButton('選択')
                button.clicked.connect(lambda checked=False, k=key: self.browse(k))
                row.addWidget(button)
            layout.addRow(label, row)

    def browse(self, key):
        if key in ('baseline_dir', 'output_root'):
            value = QFileDialog.getExistingDirectory(self, 'フォルダーを選択', self.widgets[key].text())
        else:
            value, _ = QFileDialog.getOpenFileName(self, 'ファイルを選択', self.widgets[key].text())
        if value:
            self.widgets[key].setText(value)

    def values(self):
        return {k: w.currentText() if isinstance(w, QComboBox) else w.text() for k, w in self.widgets.items()}

    def set_values(self, values):
        for key, value in values.items():
            widget = self.widgets[key]
            if isinstance(widget, QComboBox):
                widget.setCurrentText(value)
            else:
                widget.setText(value)
