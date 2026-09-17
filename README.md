# Video2X Art（非公式改造版）

**このリポジトリは、[K4YT3X/Video2X](https://github.com/k4yt3x/video2x)をベースにした非公式の改造版です。**
個人の映像制作向けに、AIモデルの重みや中間特徴に干渉する加工機能と、日本語GUIを追加しています。
アプリの表示名は「Video2X Art · 個人制作版」です。
上流のインストーラーには、本改造版の追加機能は含まれません。

Video2X本来の超解像とフレーム補間のコードを土台に、映像を溶かすような変形や質感の変化を試し、設定を保存して再生成できる制作環境を目指しています。
現在のArt処理は、`realesr-animevideov3-x2`モデルを使うWindows向けの構成です。

## 改造版で追加した機能

| 機能 | 内容 |
| --- | --- |
| 入力の加工 | ノイズ、ぼかし、元動画の時刻に沿ったノイズの時間変化 |
| モデルへの干渉 | 選択した畳み込み層の重みへのノイズや減衰、中間特徴へのノイズ、減衰、マスク |
| 再入力と仕上げ | 最大2回の推論、出力のぼかし、原像保持、元の色を使った仕上げ |
| 動画出力 | 区間指定、等倍または2倍出力、可変フレームレートへの対応、PCM音声の保持または除外 |
| 日本語GUI | 元映像と結果の同期比較、設定保存と読込、進捗表示、取消 |
| 再生成と検証 | seed固定、JSONレシピ、実行条件とハッシュの記録、保存済み実行の再生成と照合 |

元モデルから作業用コピーを作って加工し、元動画と元モデルのハッシュを検証します。
GUIとCLIは共通のPython処理基盤を使用します。

## 動作環境と制約

動作確認済みの構成はWindows 11 x64、NVIDIA GeForce RTX 4070 Ti SUPER、Python 3.14.3です。
Python依存はNumPy 2.4.2、PyAV 18.0.0、Pillow 11.3.0、PySide6 6.11.1で検証しています。
詳細な構成と測定結果は[個人制作版の検証記録](docs/art-tool-m5.md)を参照してください。

- Vulkan対応GPUとドライバーが必要です。Windows 10、別GPU、LinuxでのArt機能は未検証です。
- 対応入力は正方画素の8ビットSDR動画です。HDRや未対応の画素形式、色行列は拒否します。
- Art処理のモデルは`realesr-animevideov3-x2`に固定し、モデルのハッシュを検査します。
- 出力は可逆圧縮FFV1のMatroska動画（`.mkv`）です。処理中は中間動画も保存するため、十分な空き容量が必要です。
- 比較画面は無音です。音声付きの完成品は外部プレイヤーで確認してください。
- プレビューも本番と同じ推論条件で処理します。リアルタイム処理やリアルタイム再生は保証しません。
- 本改造版専用のインストーラーは含みません。ソース、ビルド済みCLI、依存DLL、モデル、基準記録をそろえて使用します。

## Windowsでの準備

以下のコマンドはリポジトリ直下で実行します。
ビルドにはGit、Visual Studio 2022のC++ x64ツール、CMake、Vulkan SDK、Pythonが必要です。

### 1. Python依存とCLIを準備する

```powershell
python -m pip install -r scripts/art-gui-requirements.txt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build-art-windows.ps1
```

Vulkan SDKの既定パスは`C:\VulkanSDK\1.4.350.0`です。
異なる場合はビルドコマンドに`-VulkanSdk <インストール先>`を追加します。
ビルドスクリプトは必要なsubmoduleとFFmpeg、ncnnの依存を取得し、CLIを`build/art/install/bin/video2x.exe`へ配置します。
詳細は[基準環境とビルド手順](docs/art-tool-m0.md)に記載しています。

### 2. 基準記録を作成する

**基準記録**は、無改変モデルの出力、GPU、タイル条件、実行環境を保存したものです。
Art処理はこの記録を読み込んで推論条件を決めます。
手元の対応動画を`test/test.mp4`として用意するか、`--input`を自分の動画のパスに置き換えてください。

```powershell
python -B scripts/art_model_inspect.py --output build/art/model-check.json
python -B scripts/art_probe.py --input test/test.mp4 --output-dir build/art/m2-baseline-2 --start 0 --duration 5 --seed 0
```

出力先には未使用のフォルダーを指定します。
`build/`内の基準記録や採用レシピ、`test/`内の動画はGit管理外のため、cloneだけでは取得できません。
既存の基準記録は参照入力の絶対パスとハッシュを保持するので、別環境では作成し直してください。

## GUIの使い方

基準記録を作成したら、次のコマンドで起動します。

```powershell
python -B scripts/art_gui.py --input test/test.mp4 --baseline-dir build/art/m2-baseline-2
```

依存を導入したPythonに`.pyw`が関連付いている場合は、[start_art_gui.pyw](scripts/start_art_gui.pyw)のダブルクリックでも起動できます。
CLIや基準記録の場所は画面から変更できます。

1. 元動画、開始秒、終了秒、保存先を選びます。全編を処理する場合は開始を`0`、終了を空欄にします。
2. 入力、モデル内部、再入力、出力の効果を調整します。既存のレシピは「設定読込」で開けます。
3. seedを固定して「区間プレビュー」を実行し、左の元映像と右の結果を比較します。
4. 気に入った設定を「設定保存」で新しいJSONファイルへ保存します。
5. 本番の区間を指定して「本番書出し」を実行します。

完成品は実行ごとのフォルダーに`result.mkv`として保存されます。
音声の`pcm`はPCM変換して保持、`omit`は除外を意味します。
既存の設定ファイルは上書きできないため、変更版は別名で保存してください。
操作の詳細は[GUIの説明](docs/art-tool-m4.md)を参照してください。

## CLIでの処理と再生成

GUIで生成した実行フォルダーの`recipe.json`はCLIでも使用できます。
以下の`path/to/recipe.json`は、そのファイルのパスに置き換えてください。
「設定保存」で作るGUI設定JSONには画面の操作値も含まれるため、CLIには実行フォルダー内のレシピを渡します。

```powershell
python -B scripts/art_process.py --recipe path/to/recipe.json --baseline-dir build/art/m2-baseline-2 --input test/test.mp4 --start 0 --end 5 --output-scale 2 --audio pcm --output-dir build/art/my-production
```

`--end`を省略すると末尾まで処理します。
`--output-dir`には未使用のフォルダーを指定してください。
保存した実行を同じ条件で再生成するには、次のコマンドを使います。

```powershell
python -B scripts/art_process.py --replay build/art/my-production/run.json --output-dir build/art/my-replay
```

再生成時は映像の画素と音声のPCMハッシュを照合します。
コード、CLI、DLL、FFmpeg、Python依存、GPU環境などが記録時から変わると再生成を拒否します。
環境を更新した場合は、基準記録を取り直してレシピから新規に処理してください。

| 出力ファイル | 内容 |
| --- | --- |
| `result.mkv` | 完成動画 |
| `result.json` | 完成動画と実行記録への参照 |
| `run.json` | 設定、入力、環境、進捗、計測値、検証結果 |
| `recipe.json` | CLIでも再利用できる加工設定 |
| `input.mkv` | 比較用の入力映像 |

成功時はその実行で生成した中間動画を削除します。
調査用に残す場合は`--keep-intermediates`を指定します。
失敗時と取消時は中間ファイルとログを残します。
詳しくは[動画処理と出力仕様](docs/art-tool-m3.md)を参照してください。

## 検証と開発資料

Python処理とGUIの回帰テストは次のコマンドで実行できます。

```powershell
python -B -m unittest discover -s scripts/tests
```

実GPUを使った一連の検証には、ローカルの基準記録、採用レシピ、テスト動画が必要です。
準備するファイルと検証コマンドは[個人制作版の検証記録](docs/art-tool-m5.md)を参照してください。
2026年9月16日の記録では、81件の回帰テスト、Ruff、実GPU検証が成功しています。
これは記録された個人制作環境での結果です。

- [要件定義](docs/art-tool-requirements.md)
- [開発ロードマップ](docs/art-tool-roadmap.md)
- [M0：基準環境と試作用CLI](docs/art-tool-m0.md)
- [M1：表現の試作](docs/art-tool-m1.md)
- [M2：共通処理基盤](docs/art-tool-m2.md)
- [M3：時間変化と本番動画](docs/art-tool-m3.md)
- [M4：簡易GUI](docs/art-tool-m4.md)
- [M5：個人制作版と受入記録](docs/art-tool-m5.md)

## 上流のVideo2Xについて

本改造版の基盤であるVideo2Xは、C/C++による動画の超解像とフレーム補間のプロジェクトです。
Anime4K、Real-ESRGAN、Real-CUGAN、RIFEなどの処理を備えています。
上流版の説明や配布物は以下を参照してください。
本改造版の追加機能と導入手順は、このREADMEと`docs/art-tool-*.md`に記載しています。

- [上流リポジトリ](https://github.com/k4yt3x/video2x)
- [上流版のリリース](https://github.com/k4yt3x/video2x/releases)
- [上流のドキュメント](https://docs.video2x.org/)
- [このリポジトリ内の上流由来ドキュメント](docs/book/src/README.md)

## ライセンスと謝辞

本プロジェクトのライセンスは[GNU AGPL version 3](LICENSE)です。
上流の著作権表示は以下のとおりです。

Copyright (C) 2018-2025 K4YT3X and [contributors](https://github.com/k4yt3x/video2x/graphs/contributors).

Video2Xの開発者と貢献者、および以下の依存プロジェクトに感謝します。

| プロジェクト | ライセンス |
| --- | --- |
| [FFmpeg](https://www.ffmpeg.org/) | LGPLv2.1、GPLv2 |
| [ncnn](https://github.com/Tencent/ncnn) | BSD 3-Clause |
| [Anime4K](https://github.com/bloc97/Anime4K) | MIT |
| [Real-CUGAN ncnn Vulkan](https://github.com/nihui/realcugan-ncnn-vulkan) | MIT |
| [RIFE ncnn Vulkan](https://github.com/nihui/rife-ncnn-vulkan) | MIT |
| [Real-ESRGAN ncnn Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) | MIT |

追加のライセンス情報は[NOTICE](NOTICE)を参照してください。
