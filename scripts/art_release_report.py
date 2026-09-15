"""Source-backed M5 inventory and acceptance index; never infer creator approval."""
import argparse
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
from art_processing_verify import ADOPTED_RUN_DIRECTORIES


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
    verification = load(output / 'verification.json')
    if not verification.get('originals_and_environment_preserved') or verification.get('status') == 'failed':
        raise ValueError('Verification integrity did not pass')
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
        if item['name'] not in ADOPTED_RUN_DIRECTORIES:
            raise ValueError(f'No verification run for adopted recipe: {item["name"]}')
        run_path = output / 'processing' / ADOPTED_RUN_DIRECTORIES[item['name']] / 'run.json'
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
    script = Path(__file__).resolve()
    return dict(status='technical_passed_creator_pending', acceptance=checks, adopted=adopted,
                creator_decision=dict(path=str(decision_path), sha256=sha256(decision_path)),
                report_script=dict(path=str(script), sha256=sha256(script)),
                adopted_comparison_scope='M1 acceptance retained; fixed-recipe sampled prefix pixel equality',
                performance=video['performance'], cancellation=video['cancelled'],
                production=video['performance']['five_seconds']['wall_seconds'],
                workflow='workflow/summary.json',
                pending=['Review flicker in video/full/result.mkv', 'Decide acceptable wait times'])


def accept(output, record, decision_path):
    decision = load(decision_path)
    if (decision.get('schema_version') != 1 or decision.get('status') != 'accepted_by_creator'
            or decision.get('acceptance') != {'A07': True, 'A09': True}
            or not decision.get('creator_instruction')
            or (ROOT / decision['verification_directory']).resolve() != output.resolve()):
        raise ValueError('Creator decision does not accept this verification')
    required = {'environment.json', 'verification.json', 'processing/summary.json', 'video/summary.json',
                'gui/summary.json', 'workflow/create.json', 'workflow/summary.json',
                'video/full/result.mkv', 'video/full/recipe.json', 'video/full/run.json'}
    if set(decision['evidence']) != required:
        raise ValueError('Creator decision must bind all verification evidence')
    for name, expected in decision['evidence'].items():
        if sha256(output / name) != expected:
            raise ValueError(f'Accepted evidence changed: {name}')
    for key in ('A07', 'A09'):
        record['acceptance'][key]['creator'] = 'accepted_by_creator'
    record.update(status='completed_accepted', pending=[],
                  final_creator_decision=dict(path=str(decision_path.resolve()), sha256=sha256(decision_path),
                                              date=decision['date'], instruction=decision['creator_instruction']))


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
    status = ('自動検証と制作者による受入が完了しました。' if record['status'] == 'completed_accepted'
              else '自動検証は成功。ちらつきと許容待ち時間は制作者の判定待ち。')
    if 'final_creator_decision' in record:
        links += '<li>' + link(record['final_creator_decision']['path'], '制作者の受入記録') + '</li>'
    (output / 'index.html').write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>M5 制作確認</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 16px}'
        'td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}li{margin:12px 0}</style>'
        f'<h1>M5 制作確認</h1><p>{status}</p>'
        '<table><tr><th>項目</th><th>技術確認</th><th>制作者の判断</th></tr>' + rows + '</table>'
        '<h2>採用レシピとの対応</h2><p>再生成動画は採用映像の冒頭区間と画素を比較しています。</p><ul>'
        + adopted + '</ul><h2>確認用ファイル</h2><ul>' + links + '</ul></html>', encoding='utf-8')


def write_report(output, decision_path=None):
    previous = output / 'summary.json'
    if decision_path is None and previous.exists() and load(previous).get('status') == 'completed_accepted':
        raise ValueError('Accepted report requires --decision to regenerate')
    record = summarize(output)
    if decision_path is not None:
        accept(output, record, decision_path)
    save_json(output / 'summary.json', record)
    page(output, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--decision', type=Path)
    args = parser.parse_args()
    result = write_report(args.output_dir.resolve(), args.decision)
    print(result['status'])
