# M2 再生成できる処理基盤

2026-09-14、R11〜R13を実装し、Windows上の実GPUで再生成を確認した。
採用済みの`melt-16`と`melt-24`は、M1と同じ画素結果を保っている。

## CLIで保存して再生成する

[M2 CLI](../scripts/art_process.py)はM1のレシピを読み込み、入力ノイズと重みの強度を個別に指定できる。
省略値を補った全設定、モデル名とハッシュ、処理方式、乱数方式をスキーマ版2で保存する。
未知のキー、未対応モデル、版の不一致、範囲外や非有限の値は出力の作成前に拒否する。

```powershell
python -B scripts/art_process.py --recipe build/art/m1-source-color/melt-16/recipe.json --save-recipe build/art/melt-16-v2.json
python -B scripts/art_process.py --recipe build/art/melt-16-v2.json --baseline-dir build/art/m2-baseline --output-dir build/art/m2-custom --input-noise 3 --weight-strength 0.6
python -B scripts/art_process.py --replay build/art/m2-custom/run.json --output-dir build/art/m2-replay
```

保存先のファイルと出力ディレクトリには未使用の名前を指定する。
`--replay`は保存時の設定と入力を使い、上書き指定を受け付けない。
入力、モデル、CLIとDLL、Python依存、GPUとドライバ、処理スクリプトが保存時の記録と異なる場合は開始前に停止する。
別環境で新しい結果を作る場合は`--recipe`と、その環境で作成した基準記録を使う。

タイルサイズを指定する機構をC++側へ追加したため、M0のビルドを更新する。
既存のM0記録は旧バイナリのハッシュを保持しているので、新しいビルドで基準を作り直す。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build-art-windows.ps1 -SkipDependencies
python -B scripts/art_probe.py --input test/test.mp4 --output-dir build/art/m2-baseline-new --duration 0.05 --seed 0
```

## 共通設定と実行状態

GUIからも[共通APIの`run()`](../scripts/art_processing.py)を呼び出せる。
設定は検証時に深くコピーし、呼出元のリストや既定値を共有しない。
モデル取得、変異、入力変換、推論、出力変換は一つの`Session`内で実行する。

各実行は配布モデルから専用のモデルを生成し、passごとに新しいCLIプロセスを起動する。
GPU上の変異モデルと中間特徴は子プロセスの終了時に破棄する。
共有モデルキャッシュは設けず、異なる実行への状態持越しを避けた。
2回目のpassは、M1と同じく元寸法へLanczosで戻してから無改変モデルへ渡す。

別スレッドから`Session.cancel()`を呼ぶと、フレーム間と子プロセス待機中に取消を検出する。
子プロセスを終了し、その終了を待ってからセッションを解放する。
取消済みのセッションは再開せず、次の要求には新しい`Session`を渡す。
取消と例外は`cancelled`／`failed`として記録し、途中の動画は`.partial.mkv`のまま残す。

## 乱数と実効条件の記録

[用途別RNG](../scripts/art_processing_rng.py)は、seedと用途名からSHA-256でPCG64のseedを導出する。
`input`、`weight`、`feature`、`modulation`、`pass`を分離し、対象ごとに新しいストリームを作る。
ある層や用途で乱数を消費しても、他の層、用途、次の実行には影響しない。
M1の`m1-pcg64-v1`と既存の用途名を保つため、過去の乱数列も変わらない。

現在の変調は二つの入力ノイズ場の決定的な補間であり、2回目のpassも追加乱数を使わない。
`modulation`と`pass`の専用ストリームは予約済みとして記録する。
全操作値は`configuration.settings`と`realized.settings`に保存し、抽選を適用した重みと特徴の倍率やbiasは`work/models/realesrgan`のモデル本体に残す。
実行記録にはそのパスとハッシュ、用途別の導出seedも保存する。

`run.json`はM0の元動画情報、区間、準備済み入力、基準推論を内包する。
実行ごとのモデル、コマンド、作業ディレクトリ、実効条件、出力動画とそのハッシュ、フレームごとの画素ハッシュと時刻を対応づける。
CLIの`result.json`から最終動画と実行記録へたどれる。

推論のタイルは、基準記録の値を子プロセス専用の`VIDEO2X_ART_TILE`へ設定する。
対応値は32、64、100、200で、PythonとC++の両方で検証する。
GPU、scale、prepadding、TTA、精度も実際の診断ログと照合し、一致しない出力は成功扱いにしない。
環境変数を指定しない通常のCLI実行では、従来のGPUメモリ予算によるタイル選択を使う。

## 検証結果

回帰テスト36件、Ruff、Windows x64ビルドが成功した。
[GPU検証スクリプト](../scripts/art_processing_verify.py)の結果は[summary.json](../build/art/m2-validation/summary.json)に保存した。
入力は`test/test.mp4`の先頭0.05秒、1920×888の3フレームで、出力は3840×1776。
差分の許容誤差は設けず、推論のエンコード前ハッシュと、可逆圧縮を検証した最終画素を比較した。

| 条件 | 結果 |
|---|---|
| A→B→A | 全フレーム一致。Bは入力ノイズ、重み変異、特徴mask、2 passを使用 |
| 推論中の取消→A | 全フレーム一致。取消記録と元入力、元モデルの保全を確認 |
| GPU推論後の例外→A | 全フレーム一致。失敗記録と元データ保全を確認 |
| Python終了、別プロセス起動、記録から再生成 | 全フレーム一致。A04を確認 |
| 全効果0 | M0の無改変推論と一致。A06を確認 |
| 採用済みM1の標準、強め | 両候補とも、承認済み動画の先頭3フレームと一致 |

```powershell
python -B -m unittest discover -s scripts/tests -v
ruff check scripts
python -B scripts/art_processing_verify.py --baseline-dir build/art/m2-baseline --output-dir build/art/m2-validation-new
```

今回のGPU受入は上記の短区間と現在の環境を対象とする。
本番の長区間、音声同期、区間指定と時間変化の統合はM3で検証する。
