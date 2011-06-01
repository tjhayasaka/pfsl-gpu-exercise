2011 コンペ 
===========

2010年の青木研主催のコンペティションと同様のルールで行います。

お題は『3次元拡散方程式』です。

  - 空間は立方体、大きさは一辺が 1.0 で固定

  - 境界条件はノイマン (壁で勾配ゼロ)

  - 初期条件は cosine bell

  - t=0 から 0.1 まで進める

  - 格子は 64^3、128^3、256^3 の三種類で、実行時にプログラムのオプショ
    ンで指定できる。

CPU 計算の実装例と GPU 計算の雛形を github
(https://github.com/tjhayasaka/pfsl-gpu-exercise) にて公開しています。

        $ git clone https://tjhayasaka@github.com/tjhayasaka/pfsl-gpu-exercise.git

で、手元のマシンにコピーしてください (git がインストールされていない場
合はインストールしてください)。また、実装例と雛形は更新されることがある
ので、適宜 git pull するようにしてください。

CPU 計算の実装例は solver/cpu@reference@example.org にあります。

GPU 計算の雛形 solver/gpu-blank@reference@example.org を各自のディレクトリ
(solver/gpu@メールアドレス、例えば gpu@hayasaka@pfsl.mech.tohoku.ac.jp)
にコピーし、中身を完成させてください。実質的に書換える必要があるのは
MySolver.h 内の二つの関数だけです。性能や保守性の向上のために他の部分を
改良することも歓迎します。

CPU 計算の実装例では、配列のサイズとして各軸方向に +2 を使っていますが、
GPU 計算ではどのような配列のサイズを使ってもかまいません。

計算結果が正しいかどうかは、`double` で計算したときに CPU と GPU の
Error で表示される計算結果が (ほぼ) 完全に一致していることで確認するこ
とができます。

提出
----

2099年66月666日00時00分までに以下のいずれかの方法で
hayasaka@pfsl.mech.tohoku.ac.jp まで提出してください。

  - github 上の自分専用のフォークに push する。pull request を出すか、
    メールでブランチ名を教えてください。

  - ソースコードを .tar.bz2 などにアーカイブしてメールで送る。

提出されたファイルは、hayasaka@pfsl.mech.tohoku.ac.jp がまとめた上で
github 上で公開する予定です。著作権や特許に注意してください。

勝敗
----

勝敗判定は、hayasaka@pfsl.mech.tohoku.ac.jp が行います。勝敗は、山口研
の GTX580 で `float` での計算速度を計測した結果の SCORE で決定します。
提出方法、テストのしやすさ等は、勝敗判定の上では考慮しません。

SCORE は 2010年と同じで、

        SCORE = (A)/30 + (B)/40 + (C)/50
        (A) = 64×64×64 の Performance (単位 GFlops)
        (B) = 128×128×128 の Performance (単位 GFlops)
        (C) = 256×256×256 の Performance (単位 GFlops)

(A) の比率がかなり高いので注意してください。

賞品はありません。

実装例について
==============

Debian GNU/Linux (sid)、CUDA SDK 4.0、g++-4.4 でコンパイル、実行を確認
済み。

実装例のコンパイルと実行
------------------------

        $ cd gpu-exercise/diffusion3d
        $ vi Makefile.impl  # 環境に合わせて変更
        $ make

上のようにすると、用意されている 4 種類全ての実装がコンパイルされる。特
定の実装のみコンパイルする場合は

        $ make SOLVER_IMPL_NAMES=cpu@reference@example.org

また、`float` だけでなく `double` でもコンパイルしたいときは

        $ make FLOATS="float double"

実行:

        $ bin/float@cpu@reference@example.org.exe --nx 64

用意されている実装は、

                素直な CPU版 - cpu@reference@example.org
          ちょっと速い CPU版 - cpu@hayasaka@pfsl.mech.tohoku.ac.jp
         そこそこ速い GPU 版 - gpu@hayasaka@pfsl.mech.tohoku.ac.jp
           中身が空の GPU 版 - gpu-blank@reference@example.org

の 4種類。

実装例の gpu@ と cpu@
---------------------

以下の理由により、gpu@ と cpu@ で計算結果に微妙な差がある。

  - 中間表現形式の差 (NVIDIA の GPU より Intel の CPU の方が良い。単精
    度の場合に顕著)

  - 境界条件適用のタイミングが違う (初回のイテレーションの結果が後まで
    伝播する) (とは言え今回の条件ではほとんど差は出ない)

C++
---

C++ の HPC 向きの機能を使用している。関数テンプレート、クラステンプレー
ト、関数や演算子のオーバーロード等。記述が簡潔、安全、柔軟、実行が高速。
コンパイルは遅い。

`FLOAT`
-------

プリプロセッサマクロ `FLOAT` で、計算に使用する浮動小数点型を指定できる
ようにしてある (今のところ `float` か `double` のみ)。`FLOAT` を直接参
照しているのは `main()` 内のみ、としたかったのだが、コンパイル時間短縮
のため solver/gpu@hayasaka@pfsl.mech.tohoku.ac.jp/gpu_impl.h にも参照個
所がある。他の部分からは、テンプレートのパラメータ `Float` として参照さ
れる。

`Vec3<T>`
---------

cuda の `float3` 等のテンプレート化されたラッパとして `Vec3<T>` を定義
している。`T` としてはいまのところ `unsigned int`, `int`, `float`,
`double` のみが使用可能。

`Float`
-------

計算式中に `Float(1.0)` など、定数をキャストしているところがある。例えば、

        Float a = 0.5;  a = 1.0 - cos(a);

ではなく

        Float a = 0.5;  a = Float(1.0) - cos(a);

のように記述している。前者と後者は `Float` が `double` の時は同じ動作だ
が `float` の時はそれぞれ

        前者:
        float a = float(1.0 - cos((double)f)); // C の場合
        float a = float(1.0 - double(cosf(f))); // C++ の場合

        後者:
        float a = 1.0f - cosf(0.5f);

の意味になる。今回は付随する計算も常に `Float` の型で行われるように、後
者で統一してある。なお、C++ では `cos()` などの関数は引数の型によってオー
バーロードされていて、`float` が渡されれば `float` を返す。そのため、
C++ では引数の型に合わせて関数名を変更する必要がない (C では `cos` と
`cosf` を使い分ける必要がある)。

`printf` の引数については明示的にキャストしていないが、`Float` が
`double` の場合はそのまま、`float` の場合は自動的に `double` に変換され
て渡される。

おしまい
--------
