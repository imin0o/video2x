# M1 表現試作の実行と比較

[ロードマップ](art-tool-roadmap.md)のR05〜R09を、M0のWindows CLIを呼び出すPython試作として実装した。
配布モデルから実行ごとのモデルを生成し、入力ノイズ、重み変異、中間特徴への干渉、再入力と原像保持を比較する。
R10のA01（形状と質感）とA02（不気味なぼやけ）は制作者の選択で判定する。

## 実行方法

M0のビルドを用意し、Python環境へ[依存パッケージ](../scripts/art-experiment-requirements.txt)を入れる。
今回の環境はPython 3.14、NumPy 2.4.2、PyAV 18.0.0、Pillow 11.3.0。
追加のC++ビルドは不要で、既存のncnnに含まれるVulkan対応の`Scale`層を使う。

```powershell
python -m pip install -r scripts/art-experiment-requirements.txt
python -B scripts/art_probe.py --input test/test.mp4 --output-dir build/art/m1-baseline --duration 1 --seed 0
python -B scripts/art_experiment.py --baseline-dir build/art/m1-baseline --output-dir build/art/m1-results --suite --seed 0
python -B scripts/art_experiment_review.py --suite-dir build/art/m1-results
python -B -m unittest discover -s scripts/tests -v
ruff check scripts
```

出力先には未使用のディレクトリを指定する。
完了したM0記録と入力のハッシュ、CLIとDLLのハッシュを照合してから実行する。
作業用ディレクトリの`models/realesrgan`へモデルを置き、そこをCLIの作業ディレクトリに指定する。
各候補の`run.json`にはコマンドと作業ディレクトリも記録するため、配布モデルとの取り違えを確認できる。

個別の候補は、保存された`recipe.json`を編集して実行する。
省略したキーには試作版の既定値を適用し、実効設定を出力先へ保存する。
未知のキー、非有限値、範囲外の強度、未対応層は開始前に拒否する。

```powershell
python -B scripts/art_experiment.py --baseline-dir build/art/m1-baseline --output-dir build/art/m1-custom --recipe build/art/m1-results/weight-noise-high/recipe.json
```

時間変化を1周期以上比較する場合は、M0で用意した5秒区間を使う。
生成後に`--time-dir`を付けて一覧を更新すると、短区間と長区間の先頭フレームの一致も検査する。

```powershell
python -B scripts/art_experiment.py --baseline-dir build/art/m0-test --output-dir build/art/m1-time --time-suite --seed 0
python -B scripts/art_experiment_review.py --suite-dir build/art/m1-results --time-dir build/art/m1-time
```

## 操作値と処理の意味

| 設定 | 意味と範囲 |
|---|---|
| `seed` | 0〜2⁶⁴−1。SHA-256で用途と層ごとのseedを導出し、NumPy PCG64で生成 |
| `input_noise` | 0〜128。8ビットBGR画素へ加える正規乱数の標準偏差。チャネル間は独立 |
| `weight_mode` / `weight_strength` | `noise`は対象重みのRMS×強度の正規乱数加算（0〜2）。`decay`は重み×(1−強度)（0〜1） |
| `weight_layers` | `Conv_0`〜`Conv_34`の偶数番号から指定。既定は`Conv_16` |
| `feature_mode` / `feature_strength` | `noise`、`decay`、`mask`。強度0〜1 |
| `feature_layer` | 64チャネルの`PRelu_1`〜`PRelu_33`の奇数番号。既定は`PRelu_17` |
| `time_mode` | `fixed`は固定入力ノイズ場。`smooth`は二つの固定入力ノイズ場を補間 |
| `period` / `phase` | 周期は秒（0.01〜3600）、位相は周期単位（−3600〜3600） |
| `input_blur` / `output_blur` | Pillow GaussianBlurの半径（0〜30）。入力前／推論後の比較用 |
| `passes` | 1または2。2回目は元の入力寸法へLanczosで戻し、無改変モデルで処理 |
| `retain` | 0〜1。推論結果と、出力寸法へそろえた元フレームの混合。0は処理結果、1は元フレーム |

入力ノイズは元画像の座標で一度生成し、タイル分割前に加える。
加算後は偶数丸めで整数化し、0〜255へ制限する。
強度0は画素をそのまま返す。

重みは畳込みのfp16/fp32データだけを書き換え、タグ、アラインメント、bias、PReLU slopeを保つ。
すべての変異を元モデルから生成するため、別の強度を挟んでも同じseedと強度へ戻せる。
RMSが0のテンソルは乱数加算でも0を保ち、表現できない値は保存前に拒否する。

中間特徴への干渉はPReLU直後へ`Scale`を挿入し、チャネルごとの倍率とbiasをGPUで適用する。
`noise`は特徴値の単位で強度を標準偏差とする固定bias、`decay`は全チャネルを1−強度倍、`mask`はseedで選んだチャネルを0倍にする。
maskの強度は各チャネルを落とす確率であり、64チャネル中の比率と厳密に一致するとは限らない。
空間方向に一定なのでタイル境界で乱数場がずれず、prepaddingにも同じ操作がかかる。
空間ノイズや空間maskはこの試作の対象外とした。
保存するモデルの有限値は検査するが、GPU内部の全活性値を監視する機構は未実装。

時間変化の係数は、元動画の開始秒とデコードしたPTSから求める。
`a=(1−cos(2π(t/period+phase)))/2`、ノイズ場は`(1−a)N0+aN1`とする。
補間中央では分散が端点の半分になる。
重みと特徴は固定し、入力ノイズを変調するためにフレームごとのモデル再ロードは行わない。

## 比較と検証記録

27条件の比較は先頭1秒、1920×888、60fpsを対象とし、出力は3840×1776。
比較動画は左上が元映像、右上が通常処理、左下がGaussian blur、右下が候補で、表示幅は各640画素。
顔のぼかしと再生UIは元動画にも含まれるため、紙幣、衣服、背景の変化も合わせて比較する。
E02の入力ぼかし半径2と出力ぼかし半径4は、2倍出力に対して同じ表示寸法で比べる設定とした。
音声は映像表現の比較から除外している。

| 実験 | 候補名 |
|---|---|
| E01 | `weight-noise-*`、`weight-decay-*`、`feature-noise-*`、`feature-decay-*`、`feature-mask-*` |
| E02 | `input-blur`、`input-noise-*`、`output-blur` |
| E03 | `input-noise-high`（固定）、`input-smooth`（周期4秒） |
| E04 | `weight-noise-high`（1回）、`reinput`（2回目は無改変） |
| E05 | `weight-noise-high`（保持0）、`retain-half`（保持0.5）、`retain-original`（保持1） |

各推論でエンコード前の全フレームのSHA-256と復号画素を比較し、入力から末尾までの時刻、フレーム数、出力寸法を検査する。
変換処理でもPyAVのエンコード前画素と復号画素を照合する。
`zero`と`zero-repeat`、`weight-noise-high`とその再実行を比較し、異なる設定を挟んだ再生成を確認する。
実行失敗と取消は記録に残し、書出し途中のファイルには`.partial.mkv`を使う。

`metrics.json`には先頭、中間、末尾の画素差と飽和チャネル率を記録する。
飽和率には素材にもともとある黒帯なども含む。
これらの値は白飛びや黒つぶれの比較に使い、作品としての合否は決めない。
`retain=1`は全フレームについて出力寸法へ変換した元フレームとの一致も検査する。

数値、モデル保存形式、乱数の独立性、時間比較の区間検査とVFRの短い末尾について、[回帰テスト15件](../build/art/m1-tests.log)が成功した。
新規Pythonファイルはすべて200行以下で、Ruffも通過した。
5秒の時間変化検証と一部の比較生成は同時に実行しているため、経過時間は参考測定として扱う。
GPU使用メモリも他のアプリを含むデバイス全体の200msサンプリング値で、処理単体の厳密なピークではない。

2026-09-13、27条件のGPU処理と再生成検証が完了した。
全条件で60フレーム、3840×1776、末尾PTSは0.983秒だった。
重みの高強度と強度0を、それぞれ異なる設定の実行後に再生成し、全フレームの一致を確認した。
中間特徴ノイズも、[先行確認](../build/art/m1-feature-smoke/run.json)と`feature-noise-medium`でモデルと全60フレームが一致した。

[5秒の時間比較](../build/art/m1-time/summary.json)では固定と補間の両方が300フレーム、末尾PTSは4.983秒だった。
入力ノイズの更新は固定で平均26.9ms、補間で33.2ms、初回モデル読込はそれぞれ166.1msと156.2ms。
M1の変調対象は入力ノイズとした。
二つの固定乱数場を使う方式で比較動画を作れ、フレームごとのモデル再ロードも避けられるためである。

R10では、[比較一覧](../build/art/m1-results/index.html)からA01とA02の候補を選び、[評価記録](../build/art/m1-results/review.json)の`acceptance`へ採用、見送りと理由を残す。
採用が決まるまではロードマップのR10を未チェックにする。
画素と重みの再現性を確認できても、目標とする不気味なぼやけが成立したとは判定しない。
