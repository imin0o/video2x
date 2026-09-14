"""GUI documents wrap the unchanged portable processing recipe."""
import json
import math
from pathlib import Path

from art_processing_config import configuration

FIELDS = ('source', 'start', 'end', 'output_scale', 'audio', 'baseline_dir', 'cli', 'output_root')


def document(config, controls):
    validate_controls(controls)
    controls = dict(controls)
    for key in ('source', 'baseline_dir', 'cli', 'output_root'):
        if controls[key].strip():
            controls[key] = str(Path(controls[key]).resolve())
    return dict(gui_version=1, configuration=configuration(config), controls=controls)


def load_document(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('設定はJSONオブジェクトで指定してください')
    if 'gui_version' not in value:
        return configuration(value), None
    if (type(value['gui_version']) is not int or value['gui_version'] != 1
            or set(value) != {'gui_version', 'configuration', 'controls'}):
        raise ValueError('未対応のGUI設定バージョンです')
    controls = value['controls']
    validate_controls(controls)
    return configuration(value['configuration']), controls


def validate_controls(controls):
    if (not isinstance(controls, dict) or set(controls) != set(FIELDS)
            or any(not isinstance(controls[k], str) for k in FIELDS)):
        raise ValueError('GUI操作値が不正です')
    # Numeric strings are validated before changing any widgets, without requiring the source online.
    start, end = float(controls['start']), controls['end'].strip()
    if not math.isfinite(start) or start < 0 or (end and (not math.isfinite(float(end)) or float(end) <= start)):
        raise ValueError('区間は 0 ≤ 開始 < 終了で指定してください')
    if controls['output_scale'] not in ('1', '2') or controls['audio'] not in ('pcm', 'omit'):
        raise ValueError('未対応の出力条件です')


def save_document(path, config, controls):
    # Existing recipes, videos and run records cannot be replaced through this action.
    value = document(config, controls)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
