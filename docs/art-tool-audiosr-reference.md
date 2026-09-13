# AudioSR改造から動画アートツールへ取り込む設計

調査日：2026-09-13。
参照先：`Z:\_projects\versatile_audio_super_resolution`、HEAD `d225968`と未コミット変更を含む作業ツリー。
この調査はソースとUI接続の確認であり、音声の生成実験や動画への移植実験は行っていない。
[動画側の要件定義書](art-tool-requirements.md)を補足する。

## 1. 参考にする改造の特徴

音声側では、入力、推論途中のテンソル、生成結果の再入力、最終出力のそれぞれに介入している。
操作値だけでなく実行時に決まった値を保存し、偶然得られた結果を再生成する構成も参考になる。
動画側では、この処理段階の分離と再現方法を取り込む。

音声側で確認できた内部改変は、主に条件情報と推論途中のテンソルへの干渉である。
UNetへのhookも中間出力を変化させる処理であり、学習済み重みを書き換える処理とは区別する。
動画側の重み変異は引き続き要件に含めるが、音声側で実証された方式としては扱わない。

## 2. 実装済み機能とUI接続の対応

「UI接続あり」は画面の値から処理への受渡しをソースで確認したことを意味し、効果の実測を意味しない。
READMEの機能説明より実際の呼出し経路を優先する。

| 音声側の改造 | 確認状態と根拠 | 動画側への対応案 |
|---|---|---|
| 入力のlowpass、gain、reverseなど | UI接続あり。[app.py](../../versatile_audio_super_resolution/app.py)の`process_audio_channel()`、`inference()` | 推論前の低域通過、色変化、ノイズ。推論後のぼかしと分けて比較する |
| DDIM Steps、GuidanceとそのJitter／Ramp | UI接続あり。[app.py](../../versatile_audio_super_resolution/app.py)の`process_chunk()`と`process_audio_channel()` | 変異強度の時間変調は参考になる。step数やCFGの直接移植はしない |
| 条件の混合、mask、dropout、glitch | UI接続あり。[app.py](../../versatile_audio_super_resolution/app.py)の`build_diffusion_glitch_config()`、[conditioning.py](../../versatile_audio_super_resolution/audiosr/conditioning.py) | 入力または中間特徴を部分的に欠損させ、モデルの応答を変える実験 |
| CFG schedule／inversion、初期latentのseed morph | UI接続あり。[app.py](../../versatile_audio_super_resolution/app.py) 85〜116行、[ddim.py](../../versatile_audio_super_resolution/audiosr/latent_diffusion/models/ddim.py) | 二つの固定乱数場の補間などを検証する。latentと画像モデルの特徴を同一視しない |
| UNet中間出力hook、`e_t`、`pred_x0`、`x_prev`の変異、step順序変更 | DDIM内に実装あり。現行appの設定組立てとUI入力一覧には対応キーなし。[ddim.py](../../versatile_audio_super_resolution/audiosr/latent_diffusion/models/ddim.py) | 中間特徴のノイズ、減衰、maskを調査する動機になる。GUIから使える既存機能として移植予定を立てない |
| Generation PassesとTwo-Stage Generate | UI接続あり。[app.py](../../versatile_audio_super_resolution/app.py)の`process_audio_channel()` 538〜550行、`process_chunk()` 341〜364行 | 処理結果を再入力し、強い変異の後に弱い処理を行う実験 |
| Lowband Mel／Waveform Restore Amount | UI接続あり。[ddpm.py](../../versatile_audio_super_resolution/audiosr/latent_diffusion/models/ddpm.py) 1554〜1579、1634〜1679行 | 原像保持量の操作を検討する。音声の低域復元と同じ効果は保証しない |
| Randomize All Glitchとカテゴリ別探索 | [app.py](../../versatile_audio_super_resolution/app.py)の`randomize_glitch_ui()`、[batch_plan.py](../../versatile_audio_super_resolution/audiosr/batch_plan.py) | 対象グループを選び、現在値を固定して残りを探索する操作案 |
| 出力と実効設定の対保存 | [app.py](../../versatile_audio_super_resolution/app.py)の`save_output_with_params()`、[execution.py](../../versatile_audio_super_resolution/audiosr/execution.py) | 動画と同名のJSONを保存し、後から同じ設定で再生成する |

RealESRGAN wrapperの現行経路には、DDIMの反復、CFG計算、初期拡散latentに対応する操作点を確認できない。
推論を複数回繰り返す回数と、拡散のstep数は別の設定である。
「低stepのようなぼやけ」は引き続き見た目の受入条件で判定する。

## 3. 動画側の比較実験

次の候補は、今回の参照依頼に基づいて追加する検証項目である。
全項目を初期GUIへ搭載する要件にはせず、同じ素材、seed、推論条件で比較し、A01とA02に役立つものを採用する。

| ID | 比較する処理 | 確認すること |
|---|---|---|
| E01 | 通常処理／重み変異／中間特徴のノイズ、減衰、mask | 形状が曖昧になるか、黒つぶれや乱雑な粒子だけに終わらないか |
| E02 | 入力ぼかし→通常推論／入力ノイズ→通常推論／通常推論→出力ぼかし | 推論を介した変化と単純な後加工の違いが採用理由になるか |
| E03 | 固定乱数場／二つの固定乱数場の滑らかな補間 | 時間方向の変化を作りつつ、区間と本番の一致を維持できるか |
| E04 | 強い変異の1回処理／その結果への弱い変異または通常処理 | 2回目で面白い構造が残るか、過剰に修復されるか |
| E05 | 原像保持量を0、中間、1に変更 | 元の構図や動きを残しながら、変異の見た目を制御できるか |

E01は対象層の取得と変更をncnn側で実現できることが前提である。
実験ごとに実際に変更したテンソル、層、方式を記録し、未対応モデルでは実行を開始しない。
E03の補間位置は元動画の時刻で決め、切出し区間の先頭から再計算しない。

E04では各passの入力寸法と出力寸法を固定し、倍率が反復ごとに累積しない設計を検証する。
処理順、各passのレシピ、リサイズ方式、passごとのseedを保存し、中間結果はメモリ内で受け渡す。
回数増加による待ち時間とGPUメモリも比較する。
音声側の追加passには末尾chunkを除外する分岐があり、二段生成では一部の干渉設定だけを無効化している。
動画側では末尾を含む全対象フレームに同じpass契約を適用し、二段目へ引き継ぐ設定と解除する設定を明示する。

E05の最初の比較は、出力と同じ寸法と色表現にそろえた元フレームとの混合とする。
保持量0は処理結果、1は寸法変換済み元フレームに一致させる。
音声の帯域別復元に相当する空間周波数別混合は追加の実験候補とし、通常の混合とは別方式で記録する。

## 4. 探索と再生成で取り込む要件

- 画面とレシピでは「入力」「モデル内部」「再入力」「出力」の作用先を区別する。初期版で未対応の群は操作可能に見せない。
- ランダム化を採用する場合は設定値の変更と生成実行を分ける。対象群と探索範囲を記録し、seed固定だけで設定まで固定したように見せない。
- マスターseedから用途別の乱数を導出し、入力ノイズ、中間特徴、重み変異、時間変調、passの乱数消費を分離する。
- 出力には解決済みの値を持つJSONを対保存する。乱数で選んだ層や方式、範囲、実際のseedも保存し、再生時に再抽選しない。
- プレビューをA、別設定をBとしたA→B→Aの順序で、最後のAが最初のAに一致することを確認する。
- キャッシュしたモデルの変異、hook、履歴が別の実行へ残らないよう、正常終了、取消、例外で状態を復元する。
- モデルを共有するGPU処理は実行単位で管理する。キャッシュの取得だけを排他して推論中の変更が競合する構成を避ける。

音声側では`execution.py`の`derived_seed()`、`random_scope()`、`isolated()`が、用途に基づくseed導出、乱数状態の復元、実効設定の記録と履歴の寿命を扱う。
動画側では各作用先まで乱数を分離する設計を明示し、音声側の全処理がすでに同じ粒度で分離されているとは解釈しない。
音声のバッチ探索にある大量生成と再開機構は参考に留め、合意済みの短区間プレビューを中心とした制作フローを維持する。

## 5. 参照した作業ツリーの識別

参照時に`app.py`、`ddim.py`、`reconstruction.py`、`execution.py`などに未コミット変更があった。
HEADだけでは今回読んだ内容を特定できないため、主要ファイルのSHA-256を以下に記録する。
この記録は変更前後の識別用であり、ソースのバックアップではない。

| ファイル | SHA-256 |
|---|---|
| `app.py` | `70fe58e9b5e7893c76b0d69af5aaa6d985745159dcf266f6d125bbd7581201a4` |
| `audiosr/latent_diffusion/models/ddim.py` | `f571c40781aeee605901f70de1c2e7c49f4ea199357005b2f72fb8a52d2f9ac0` |
| `audiosr/conditioning.py` | `2b9932e82de941b6b3e7872e9c164ef0d33b0c7fc4db15c4d14c37d444c5cab5` |
| `audiosr/reconstruction.py` | `3d819518aea47a7446b2264e5bde0e08c4dc02ea2a6ea9354fb6c4672809d743` |
| `audiosr/execution.py` | `bcb1c690ec7828531f0ec77aaabd8218f77ce8713374ca397585d7bd0bc21420` |
| `audiosr/batch_plan.py` | `6e5f8e6087e52779225ef8a97c56bcbb1aa5e37074860e8f035de836c689b078` |
