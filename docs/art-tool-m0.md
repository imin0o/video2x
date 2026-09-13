# M0 基準環境と試作用CLI

対象は[ロードマップ](art-tool-roadmap.md)のR01〜R04。
RealESRGANの無改変処理を基準に、モデル構造、実効設定、処理時間と再実行結果を記録する。

## ビルドと実行

リポジトリのルートで次を実行する。
Visual Studio 2022のC++ x64ツール、CMake、Vulkan SDK、Pythonが必要となる。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build-art-windows.ps1
python -B scripts/art_model_inspect.py --output build/art/model-check.json
python -B scripts/art_probe.py --input test/test.mp4 --output-dir build/art/m0-next-run --start 0 --duration 5 --seed 0
python -B -m unittest discover -s scripts/tests -v
```

Vulkan SDKの場所を変更する場合はビルドスクリプトへ `-VulkanSdk <path>` を渡す。
依存取得済みの再ビルドには `-SkipDependencies` を指定する。
試作用CLIの出力先は新しいディレクトリを指定する。既存の結果は上書きしない。

ビルドスクリプトは `PROCESSOR_ARCHITECTURE=AMD64` と `PreferredToolArchitecture=x64` を設定する。
生成したMSBuildプロジェクトで `x64` / `Native64Bit` を確認できなければビルドを止める。
ビルド引数、依存版、取得アーカイブのSHA-256は `build/art/environment.json` に保存する。
旧ビルド文書の `USE_SYSTEM_*` ではなく、現行CMakeの `VIDEO2X_USE_EXTERNAL_*` を使う。

| 依存 | M0で固定した版 |
|---|---|
| FFmpeg shared | 7.1（既存Windows CIと同じ） |
| ncnn shared | 20241226（既存Windows CIと同じ） |
| ソースsubmodule | 親リポジトリのgitlink指定コミット |
| Vulkan SDK | この環境の導入済み1.4.350.0 |

Windows CLIでは、ローカル変数と推論処理の破棄後、ドライバDLLの終了処理より前にncnnのGPUインスタンスを解放する。
この順序がない構成では、全フレーム出力後に `0xc0000005` で終了した。
デバッガで `ncnn::destroy_gpu_instance` → `VulkanDevice` のデストラクタ → Vulkan/NVIDIAドライバの呼出しを確認した。

## モデル構造と変更箇所

対象モデルは `models/realesrgan/realesr-animevideov3-x2` に固定する。
読み取り専用の調査スクリプトで、全層の入出力blob、重み形状、バイナリ内の位置、形式、最小値、最大値とRMSをJSONへ保存する。
層数、バイナリ末尾までの消費、有限値を検査し、未対応の演算や重み形式はエラーにする。

| 項目 | 確認結果 |
|---|---|
| 構造 | 41層、42 blob。Convolution 18、PReLU 17、その他6層 |
| テンソル | 53個。畳込み重み18個がfp16、bias 18個とPReLU slope 17個がfp32 |
| バイナリ長 | 1,247,368 bytes。調査スクリプトが末尾まで過不足なく消費 |
| パラメータSHA-256 | `1393f7c0e885f9d15a0668329a13f695ebd0ea45791f46d28145d7934824d224` |
| 重みSHA-256 | `548a36f9c3f4ab8da56cd3b13badf23968bee207b396dad14d04b830e5f2ab2d` |
| 主な対象層 | `Conv_0`〜`Conv_34`（偶数番号）。中間64チャネルの3×3畳込みを含む |
| 倍率 | 内部でPixelShuffleによる4倍化と入力の残差加算を行い、最後に0.5倍へ縮小するx2モデル |

| 変更案 | 可否と境界 |
|---|---|
| ロード前の重み変異 | `.param` のサイズと型に従い、作業用モデルの畳込み重みを数値として変更できる構造。fp16タグ `0x01306B47`、4 bytes境界、biasとslopeのfp32を区別する |
| ロード後の重み変更 | 現行ラッパーの `net` はprivate。ncnnは `load_model` 中にpipelineを作り、Winograd等の形式へ変換してGPUに転送する。CPU側の重みだけを書き換えても反映を保証できず、再ロードかpipeline再作成が必要 |
| 中間特徴への干渉 | 各タイルの `Extractor` で入力blobから `output` を抽出する。例えば `Conv_0` 出力の `54`、`PRelu_1` 出力の `56` が候補。現行APIには任意blobへのhookがなく、GPU演算の追加またはグラフ変更が必要 |
| 特徴ノイズとmask | タイルの元座標とprepaddingを考慮する必要がある。GPUでの実装、余白と本体の対応、追加転送コストはM1で検証する |

この可否判定は形式とコードの調査結果であり、変異後の推論や映像表現の成立はM1で確認する。
試作では、元モデルから毎回作業用コピーを生成してロードする方式を第一候補とする。
ロード後の内部状態や既存GPUバッファを書き換える方式は、変換済みの重みとキャッシュを再構築する検証が別途必要となる。

根拠は [モデル構造](../models/realesrgan/realesr-animevideov3-x2.param)、[RealESRGAN](../third_party/librealesrgan_ncnn_vulkan/src/realesrgan.cpp)、[ncnn modelbin](../third_party/ncnn/src/modelbin.cpp)、[ncnn Net](../third_party/ncnn/src/net.cpp)、[Vulkan convolution](../third_party/ncnn/src/layer/vulkan/convolution_vulkan.cpp)。
実行DLLはncnn 20241226、ソースsubmoduleは `305837f`（2025-05-03）であり、同一版とは扱わない。
実行版に対応するタグ `20241226`（`5285895`）の上記処理も確認した。実際のDLLのSHA-256は各実行記録に残す。

## 比較と実行記録

試作用CLIは8-bit SDR動画の指定区間をFFV1/BGR0へ切り出し、同じ無改変モデルで2回処理する。
入力の解像度とフレームレートを保ち、出力はモデルの2倍寸法とする。
音声はこの画素比較用素材から除く。元ファイルは変更しない。
seedは保存するが、M0では乱数処理を使わず、方式を `baseline-v1` として記録する。

| ファイル | 内容 |
|---|---|
| `run.json` | 状態、seed、方式、対象層、実効値、入力とモデルのハッシュ、環境、コマンド、計測結果、検証判定 |
| `model-inventory.json` | 読み取り専用のモデル調査結果 |
| `input.mkv` / `baseline-1.mkv` / `baseline-2.mkv` | 入力区間と、2回分の無改変結果 |
| `baseline-*.log` / `decoded-*.sha256` | エンコード前の画素ハッシュと、動画から復号した画素ハッシュ |
| `gpu-*.csv` | 200ms間隔のGPU全体の使用メモリ。別アプリを含むため、プロセス単独の厳密なピーク値ではない |
| `comparison.mp4` / `comparison.png` | 左から元映像、通常処理、Gaussian blur（sigma=2）。各パネル幅640pxで同時刻を比較 |

画素ハッシュは推論出力の連続BGR24バイト列から、色変換とエンコードの前に計算する。
2回の各フレームのハッシュと時刻が一致し、保存動画の復号後も同じ画素になり、元ファイルのハッシュが保たれたときだけ成功にする。
CLIが異常終了した場合は出力を `.partial.mkv` のまま残し、JSONを失敗状態にする。

推論時間はモデルの `process` 呼出し区間を計測し、画素ハッシュ計算とエンコード時間を含めない。
全体の経過時間はデコード、FFV1書出し、ログと画素ハッシュ計算を含むため、推論時間とは区別する。
初回フレームと2フレーム目以降の平均を分け、同一プロセス内でのモデル再利用と、別プロセスでの再実行を確認する。
モデル読込時間は `load` 呼出しの時間であり、GPUデバイス作成は全体の経過時間に含む。

## 指定素材での実測

2026-09-13、Windows 11（10.0.26200）、RTX 4070 Ti SUPER（16,376 MiB）、NVIDIAドライバ610.62で実行した。
入力は `test/test.mp4` の動画ストリーム1、先頭0〜5秒（1920×888、60fps、300フレーム）。
人物、紙幣の細かな模様、手と紙幣の動きが含まれる。
無改変出力は3840×1776、TTA無効、実効タイル200、prepadding 10。

| 1回目の計測 | 結果 |
|---|---|
| モデル読込 | 149.654 ms |
| 最初のフレームの推論 | 108.414 ms |
| 再利用時の推論平均（残り299フレーム） | 103.145 ms |
| CLI全体（FFV1書出しと検証用ログを含む） | 83.057秒 |
| GPU全体の使用メモリの観測最大値 | 3,910 MiB（200msサンプリング、他アプリを含む） |

2回目はモデル読込139.659ms、再利用時の推論平均99.924ms、CLI全体73.935秒、GPU全体の観測最大値4,137 MiBだった。
2回とも終了コード0で、全300フレームの時刻とエンコード前画素が一致し、FFV1から復号した画素も一致した。
元動画とモデルのSHA-256が処理前後で変わらないことを確認した。
モデル形式と実行記録の検査テスト6件、Ruff、終了時クラッシュの2フレーム回帰確認が成功した。

詳細な実効値と全フレームのハッシュは [実行記録](../build/art/m0-test/run.json)、視覚比較は [比較動画](../build/art/m0-test/comparison.mp4) に保存した。
