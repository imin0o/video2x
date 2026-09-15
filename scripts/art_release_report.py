"""Source-backed M5 inventory and acceptance index; never infer creator approval."""
import html
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from urllib.parse import quote

from art_experiment_run import FFMPEG, software_environment
from art_model_inspect import MODEL, ROOT, sha256
from art_probe_support import binary_inventory, command, gpu_inventory, save_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inventory(cli, baseline):
    paths = [ROOT / 'test/test.mp4', MODEL.with_suffix('.param'), MODEL.with_suffix('.bin')]
    baseline_record = baseline / 'run.json'
    paths += [baseline_record, Path(load(baseline_record)['prepared_input']['path'])]
    paths += sorted((ROOT / 'scripts').glob('art_*.py'))
    paths += [ROOT / 'scripts/start_art_gui.pyw', ROOT / 'scripts/art-gui-requirements.txt',
              ROOT / 'scripts/art-experiment-requirements.txt', ROOT / 'docs/art-tool-m1-acceptance.json']
    paths += sorted((ROOT / 'build/art/m1-source-color').glob('melt-*/recipe.json'))
    return dict(software=software_environment(), windows=platform.win32_ver(), machine=platform.machine(),
                python_executable=sys.executable,
                packages={p: importlib.metadata.version(p) for p in ('numpy', 'av', 'Pillow', 'PySide6')},
                gpu=gpu_inventory(), binaries=binary_inventory(cli),
                ffmpeg=dict(path=str(FFMPEG), sha256=sha256(FFMPEG),
                            version=command([FFMPEG, '-version']).splitlines()[0]),
                files={str(p): sha256(p) for p in paths})


def summarize(output):
    processing, video, gui = [load(output / name / 'summary.json') for name in ('processing', 'video', 'gui')]
    workflow = load(output / 'workflow/summary.json')
    create = load(output / 'workflow/create.json')
    if (processing['status'] != 'completed' or video['status'] != 'completed'
            or gui['status'] != 'passed' or workflow['status'] != 'passed'
            or create['status'] != 'passed' or not create['A03']
            or not all(processing['checks'].values()) or not all(video['checks'].values())):
        raise ValueError('A verification suite did not pass')
    decision_path = ROOT / 'docs/art-tool-m1-acceptance.json'
    decision = load(decision_path)
    adopted = []
    for item in decision['adopted']:
        run_path = output / 'processing' / ('a' if item['name'] == 'melt-16' else item['name']) / 'run.json'
        record = load(run_path)
        if sha256(Path(record['result'])) != record['result_sha256']:
            raise ValueError(f'Adopted verification output changed: {run_path}')
        adopted.append(item | dict(final_run=str(run_path), final_video=record['result'],
                                   final_recipe=str(run_path.with_name('recipe.json')),
                                   final_video_sha256=record['result_sha256'],
                                   compared_frames=len(record['final_frames'])))
    checks = {
        'A01': dict(technical='passed', creator='accepted_M1', evidence='processing/summary.json'),
        'A02': dict(technical='passed', creator='accepted_M1', evidence='processing/summary.json'),
        'A03': dict(technical='passed', evidence='workflow/create.json'),
        'A04': dict(technical='passed', evidence='workflow/summary.json'),
        'A05': dict(technical='passed', evidence='video/summary.json'),
        'A06': dict(technical='passed', evidence='processing/summary.json'),
        'A07': dict(technical='passed', creator='pending_flicker_review', evidence='video/summary.json'),
        'A08': dict(technical='passed', evidence=['workflow/summary.json', 'video/summary.json', 'gui/summary.json'],
                    limitation='MemoryError injected after inference; no physical GPU exhaustion'),
        'A09': dict(technical='measured', creator='pending_wait_time_limit', evidence='video/summary.json'),
        'A10': dict(technical='passed', evidence=['processing/summary.json', 'gui/summary.json'])}
    return dict(status='technical_passed_creator_pending', acceptance=checks, adopted=adopted,
                creator_decision=dict(path=str(decision_path), sha256=sha256(decision_path)),
                adopted_comparison_scope='M1 acceptance retained; fixed-recipe sampled prefix pixel equality',
                performance=video['performance'], cancellation=video['cancelled'],
                production=video['performance']['five_seconds']['wall_seconds'],
                workflow='workflow/summary.json',
                pending=['Review flicker in video/full/result.mkv', 'Decide acceptable wait times'])


def page(output, record):
    def link(path, label):
        relative = Path(os.path.relpath(Path(path), output)).as_posix()
        return f'<a href="{html.escape(quote(relative))}">{html.escape(label)}</a>'

    rows = ''.join(f'<tr><td>{key}</td><td>{value["technical"]}</td>'
                   f'<td>{html.escape(value.get("creator", ""))}</td></tr>'
                   for key, value in record['acceptance'].items())
    adopted = ''.join('<li>' + html.escape(item['name']) + ': '
                      + link(item['final_video'], '再生成動画') + ' / '
                      + link(item['final_recipe'], 'レシピ') + ' / '
                      + link(item['final_run'], '実行記録') + ' / '
                      + link(ROOT / Path(item['run']).parent / 'comparison.mp4', 'M1採用時の比較') + '</li>'
                      for item in record['adopted'])
    links = ''.join('<li>' + link(output / path, label) + '</li>' for path, label in (
        ('video/full/result.mkv', '5秒の本番動画（ちらつき確認用）'),
        ('video/vfr/result.mkv', '可変フレームレート・音声付き'),
        ('workflow/settings.json', 'GUI保存設定'), ('workflow/gui.png', '再起動後のGUI'),
        ('summary.json', '測定値・受入記録'), ('environment.json', 'Windows・GPU・DLL・モデル構成')))
    (output / 'index.html').write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>M5 制作確認</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 16px}'
        'td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}li{margin:12px 0}</style>'
        '<h1>M5 制作確認</h1><p>自動検証は成功。ちらつきと許容待ち時間は制作者の判定待ち。</p>'
        '<table><tr><th>項目</th><th>技術確認</th><th>制作者の判断</th></tr>' + rows + '</table>'
        '<h2>採用レシピとの対応</h2><p>再生成動画は採用映像の冒頭区間と画素を比較しています。</p><ul>'
        + adopted + '</ul><h2>確認用ファイル</h2><ul>' + links + '</ul></html>', encoding='utf-8')


def write_report(output):
    record = summarize(output)
    save_json(output / 'summary.json', record)
    page(output, record)
    return record
