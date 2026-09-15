# M5 個人制作版

[art_release_verify.py](../scripts/art_release_verify.py)でM5を検証する。
同じCLI、DLL、モデルで処理基盤とGUIを再検証し、映像、レシピ、実行記録、測定値を一つの確認ページにまとめる。
GUIのタイトルを「Video2X Art · 個人制作版」に更新した。
2026-09-14にR22とR23を完了し、2026-09-15の制作者の指示でR21を完了した。
2026-09-16に修正後のコードで全検証を取り直し、制作者の目視による受入へ記録を改めた。

## 検証対象のWindows構成

対象はこの作業ツリーを配置した個人利用のWindows環境とする。
別PCへ移すためのインストーラーは含めない。

| 項目 | 検証構成 |
|---|---|
| OS | Windows 11 x64、10.0.26200 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER、16376 MiB |
| NVIDIAドライバ | 610.62 |
| Python | 3.14.3、`C:\Python314\python.exe` |
| Python依存 | PySide6 6.11.1、NumPy 2.4.2、PyAV 18.0.0、Pillow 11.3.0 |
| FFmpeg | 7.1 full_build shared |
| 推論モデル | `realesr-animevideov3-x2`、M1採用時と同じparamとbin |

Windows 10、別GPU、別ドライバの動作は未検証。
VulkanドライバとMSVCランタイムはこのPCに導入済みの構成を利用する。
Python依存の指定範囲は開発用requirementsに残し、再現時には上表のバージョンを使う。

CLIと同じ`build/art/install/bin`に配置するDLLは以下。
各ファイルのSHA256は検証出力の`environment.json`に記録する。

```text
avcodec-61.dll, avdevice-61.dll, avfilter-10.dll, avformat-61.dll
avutil-59.dll, postproc-58.dll, swresample-5.dll, swscale-8.dll
boost_program_options-vc143-mt-x64-1_86.dll, spdlog.dll, ncnn.dll
libvideo2x.dll, librealesrgan-ncnn-vulkan.dll
librealcugan-ncnn-vulkan.dll, librife-ncnn-vulkan.dll
```

## 配置と起動

次のファイルを作業ツリー内に保持する。
基準記録は元動画と準備済み入力を絶対パスとハッシュで参照するため、フォルダーだけを移動するとそのまま再利用できない。

```text
scripts/art_*.py, scripts/start_art_gui.pyw
build/art/install/bin/video2x.exe と上記DLL
third_party/ffmpeg-shared/bin/ffmpeg.exe
models/realesrgan/realesr-animevideov3-x2.param
models/realesrgan/realesr-animevideov3-x2.bin
build/art/m2-baseline-2/run.json とその参照入力
build/art/m1-source-color/melt-16/recipe.json
build/art/m1-source-color/melt-24/recipe.json
test/test.mp4
```

既存環境での起動はリポジトリ直下から行う。

```powershell
python -B scripts/art_gui.py --recipe build/art/m1-source-color/melt-16/recipe.json --input test/test.mp4
```

依存パッケージを同じバージョンで準備する場合は、使用するPython環境で次を実行する。

```powershell
python -m pip install numpy==2.4.2 av==18.0.0 Pillow==11.3.0 PySide6==6.11.1
```

`.pyw`がこのPythonに関連付いていれば`start_art_gui.pyw`のダブルクリックでも起動できる。
CLIを変更した場合は[M0の手順](art-tool-m0.md)で基準記録を取り直し、GUIの「基準記録」をそのフォルダーに変更する。
MSVCでのビルドは[build-art-windows.ps1](../scripts/build-art-windows.ps1)を使う。

## 保存から本番出力まで

1. `melt-16`を読み込み、元動画と短い開始秒、終了秒を指定する。
2. seedを固定して効果を調整し、「区間プレビュー」で比較する。
3. 「設定保存」で新しいJSONへ保存して終了する。
4. GUIを起動し直し、「設定読込」で保存したJSONを開く。
5. 「区間プレビュー」で再生成する。元の条件が同じなら対応フレームの画素が一致する。
6. 本番の区間と保存先を指定して「本番書出し」を押す。全編は開始0、終了空欄。

完成品は実行ごとのフォルダー内の`result.mkv`で、`result.json`に条件と検証結果が残る。
比較画面は無音のため、音声付き動画は外部プレイヤーで視聴する。
既存のGUI設定ファイルを保存先に指定すると上書きを拒否するので、改訂版には別名を使う。

## 自動検証と証拠

```powershell
python -B -m unittest discover -s scripts/tests
ruff check scripts
python -B scripts/art_release_verify.py --output-dir build/art/m5-verified-new
```

出力先は未使用のフォルダーを指定する。
検証はGPUを順番に使用し、子プロセスのログと完了段階を記録する。
途中で失敗した場合は`summary.json`と`verification.json`を失敗状態で残す。
失敗状態の出力に`art_release_report.py`を実行しても、確認ページは作らない。

| 記録 | 確認内容 |
|---|---|
| `processing/summary.json` | M1採用レシピの冒頭画素、全効果0、A→B→A、取消と例外後の再生成 |
| `video/summary.json` | 5秒本番、途中区間、単フレーム、短い末尾、VFR、音声、工程別計測 |
| `gui/summary.json` | 実GUIの進捗、GPU取消、失敗後の復帰、同期表示 |
| `workflow/create.json` | 同一seedの3強度、採用値への復帰、GUI設定保存 |
| `workflow/summary.json` | 別Pythonプロセスで起動、全操作値復元、再生成、本番書出し |
| `environment.json` | OS、GPU、ドライバ、Python依存、CLIとDLL、元動画、元モデルのハッシュ |
| `index.html` | A01〜A10、採用映像と再生成映像、レシピと計測記録のリンク |

不正レシピはGUIに適用する前に拒否し、モデル欠落は生成したコピーのbinだけを除いて確認する。
元モデルには手を加えない。
メモリ不足はGPU推論後に`MemoryError`を注入し、失敗記録と復帰を確認する。
物理的なGPUメモリ枯渇は再現していない。
GUI検証はQtの`offscreen`モードを使い、ネイティブのファイル選択とダブルクリックは自動操作しない。
設定保存はファイル選択の戻り値だけを差し替え、「設定保存」と同じ保存処理を通す。

モデル読込、初回推論、同じ読込内の後続フレームの平均を分けて記録する。
前処理、出力変換、エンコードは`timings`に残す。
GPUメモリは200ms間隔のデバイス全体の使用量であり、他アプリを含む。
GUIの実行をまたぐモデル常駐は行わない。

## 2026-09-14の測定結果

[確認ページ](../build/art/m5-verified-final/index.html)に、採用レシピと全検証結果へのリンクを保存した。
77件の回帰テスト、Ruff、実GPUの全検証が成功した。
元動画、元モデル、CLI、DLL、検証に使用したPythonコードは実行前後でハッシュが一致した。
別プロセスのGUIで保存した全操作値を復元し、プレビュー2フレームと本番15フレームの対応画素が一致した。
この記録の設定保存は、「設定保存」の処理を通さず保存関数を直接呼ぶ版で取得した。

[初回の動画検証記録](../build/art/m5-verified/video/summary.json)では全チェックが成功した。
元動画は1920×888で、5秒本番は3840×1776の300フレーム。
音声の220500サンプルを保持し、同期検証も成功した。

| 計測 | 5秒本番 | 単フレーム |
|---|---:|---:|
| 処理全体 | 200.09秒 | 4.63秒 |
| 入力準備 | 6.48秒 | 0.11秒 |
| モデル読込 | 152.35ms | 162.58ms |
| 初回フレーム推論 | 105.90ms | 108.39ms |
| 後続フレーム推論の平均 | 97.12ms | 対象なし |
| 最終エンコードと音声結合 | 16.65秒 | 0.46秒 |
| GPUデバイス全体の使用量の最大値 | 1535 MiB | 1546 MiB |

取消応答は0.038秒だった。
上表は各1回の測定で、推論時間は処理全体の一部に限る。
単フレーム検証の出力倍率は1、5秒本番は2であり、全体時間をフレーム数だけで比例換算しない。

GUI検証画像の日本語フォントを修正した後に全検証を取り直した。
[再測定](../build/art/m5-verified-final/video/summary.json)の5秒本番は220.42秒で、初回から約10%変動した。
検証画像にはMeiryoを明示し、実行前後の構成照合には修正後のスクリプトを含めた。

## 2026-09-16の再検証

失敗した検証から確認ページを作れる問題と、GUI検証の設定保存が「設定保存」の処理を通らない問題を修正した。
その後、同じ構成で全検証を取り直し、[確認ページ](../build/art/m5-verified-2026-09-16/index.html)を受入の対象にした。
81件の回帰テスト、Ruff、実GPUの全検証が成功し、元動画、元モデル、CLI、DLL、Pythonコードのハッシュは実行前後で一致した。
5秒本番は198.90秒、単フレームは4.09秒、取消応答は0.038秒だった。
5秒本番のうち、推論のサブプロセスは101.35秒、最終エンコードと音声結合は15.74秒だった。

## 制作者による最終確認

M1で採用済みの`melt-16`と`melt-24`は、最終構成で再生成した冒頭区間との画素一致を確認した。
この一致は長い映像全体の再評価を代替しない。
2026-09-16に再検証の5秒本番映像と待ち時間の測定値を提示し、制作者は映像を目視して「目視した。これでOK」とA07とA09を受け入れた。
この判断を[受入記録](art-tool-m5-acceptance.json)に残し、2026-09-15に視聴なしの完了指示で記録した受入を置き換えた。
対象は記録済みの個人制作環境で、単フレーム約4.1〜4.6秒、5秒動画約199〜220秒の測定結果とする。

受入記録は検証環境、本番映像、レシピ、各検証結果のSHA256を保持する。
次のコマンドで一致を確認して、受入済みの確認ページを再生成できる。

```powershell
python -B scripts/art_release_report.py --output-dir build/art/m5-verified-2026-09-16 --decision docs/art-tool-m5-acceptance.json
```

`verification.json`は受入前の技術検証記録として保持し、最終状態は`summary.json`の`completed_accepted`に記録する。
`--decision`を省いて実行すると、受入済みの`summary.json`を上書きせずに停止する。
`summary.json`には、確認ページの生成に使った`art_release_report.py`のSHA256も残す。
新たなGPU検証には今回の受入を自動適用しない。
`m5-verified-final`は2026-09-14の技術検証記録として残し、現在の受入記録では確認ページを再生成できない。
