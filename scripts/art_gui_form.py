"""Only supported M2 controls are exposed, grouped by processing stage."""
import secrets

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from art_processing_config import DEFAULTS, configuration

GROUPS = [
    ('入力', [('input_noise', 'ノイズ強度', 0, 128), ('input_blur', 'ぼかし', 0, 30)]),
    ('モデル内部 · 重み', [('weight_mode', '方式', ['noise', 'decay']),
                         ('weight_strength', '強度', 0, 2), ('weight_layers', '対象層（カンマ区切り）', None)]),
    ('モデル内部 · 中間特徴', [('feature_mode', '方式', ['noise', 'decay', 'mask']),
                            ('feature_strength', '強度', 0, 1),
                            ('feature_layer', '対象層', [f'PRelu_{i}' for i in range(1, 34, 2)])]),
    ('時間変化 · 入力ノイズに作用', [('time_mode', '方式', ['fixed', 'smooth', 'smooth-constant']),
                                  ('modulation_depth', '変調幅', 0, 1), ('period', '周期（秒）', .01, 3600),
                                  ('phase', '位相（周期比）', -3600, 3600)]),
    ('再入力', [('passes', 'pass数', ['1', '2'])]),
    ('出力', [('output_blur', 'ぼかし', 0, 30), ('retain', '原像保持量', 0, 1),
              ('source_color', '元の色の保持', 0, 1), ('luma_change', '明度変化', 0, 1)]),
]


class SettingsForm(QScrollArea):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.widgets = {}
        self.original_numbers = {}
        body = QWidget()
        layout = QVBoxLayout(body)
        self.seed = QLineEdit('0')
        self.seed.setAccessibleName('固定seed')
        self.seed.textChanged.connect(self.changed)
        layout.addWidget(QLabel('seed · 変更しない限り固定'))
        layout.addWidget(self.seed)
        randomize = QPushButton('seedを再抽選（生成はしない）')
        randomize.clicked.connect(lambda: self.seed.setText(str(secrets.randbits(64))))
        layout.addWidget(randomize)
        for title, fields in GROUPS:
            box, form = QGroupBox(title), QFormLayout()
            box.setLayout(form)
            for key, label, *spec in fields:
                if len(spec) == 2:
                    widget = QDoubleSpinBox()
                    widget.setDecimals(8)
                    widget.setRange(*spec)
                    widget.setSingleStep(.05 if spec[1] <= 2 else 1)
                    widget.valueChanged.connect(self.changed)
                elif isinstance(spec[0], list):
                    widget = QComboBox()
                    widget.addItems(spec[0])
                    widget.currentTextChanged.connect(self.changed)
                else:
                    widget = QLineEdit()
                    widget.textChanged.connect(self.changed)
                widget.setAccessibleName(title + ' ' + label)
                widget.setObjectName(key)
                self.widgets[key] = widget
                form.addRow(label, widget)
            layout.addWidget(box)
        reset = QPushButton('全創作効果を0に戻す')
        reset.clicked.connect(self.reset_effects)
        layout.addWidget(reset)
        layout.addStretch()
        self.setWidget(body)
        self.setWidgetResizable(True)
        self.setMinimumWidth(360)
        self.set_config(configuration({}))
        self.widgets['weight_mode'].currentTextChanged.connect(self.update_enabled)
        self.widgets['time_mode'].currentTextChanged.connect(self.update_enabled)
        self.update_enabled()

    def update_enabled(self):
        self.widgets['weight_strength'].setMaximum(1 if self.widgets['weight_mode'].currentText() == 'decay' else 2)
        for key in ('period', 'phase', 'modulation_depth'):
            self.widgets[key].setEnabled(self.widgets['time_mode'].currentText() != 'fixed')

    def config(self):
        values = dict(DEFAULTS, seed=int(self.seed.text()))
        for key, widget in self.widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                original = self.original_numbers.get(key, widget.value())
                values[key] = original if widget.value() == round(original, 8) else widget.value()
            elif isinstance(widget, QComboBox):
                values[key] = int(widget.currentText()) if key == 'passes' else widget.currentText()
            else:
                values[key] = [x.strip() for x in widget.text().split(',')]
        return configuration(values)

    def set_config(self, config):
        values = configuration(config)['settings']
        self.seed.setText(str(values['seed']))
        # Mode first: decay's range must not clip a newly loaded noise strength.
        self.widgets['weight_mode'].setCurrentText(values['weight_mode'])
        self.update_enabled()
        for key, widget in self.widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(values[key])
                self.original_numbers[key] = values[key]
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(values[key]))
            else:
                widget.setText(','.join(values[key]))
        self.changed.emit()

    def reset_effects(self):
        for key in ('weight_strength', 'feature_strength', 'input_noise', 'input_blur',
                    'output_blur', 'retain', 'source_color', 'luma_change'):
            self.original_numbers[key] = 0
            self.widgets[key].setValue(0)
        self.widgets['passes'].setCurrentText('1')
