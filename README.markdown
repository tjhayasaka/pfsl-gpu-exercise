2011 コンペ 結果
================
                                                                                                               
    順位        実装 (提出者)             nx=64 で  nx=128で  nx=256で  SCORE (1)
                                          のGFLOPS  のGFLOPS  のGFLOPS
    1 gpu@nakaaki@pfsl.mech.tohoku.ac.jp   49.995   137.798   146.738   8.046
    2 gpu@miyagawa@pfsl.mech.tohoku.ac.jp  78.952   101.341    93.944   7.044 (2)
    3 gpu@kyoya@pfsl.mech.tohoku.ac.jp     70.613    85.781    79.989   6.098

    (1) GTX580 で `float` でランダムな順に 70回ずつ実行し、最も高い FLOPS の値を採用した。
        SCORE = (A)/30 + (B)/40 + (C)/50
        (A) = 64×64×64 の Performance (単位 GFlops)
        (B) = 128×128×128 の Performance (単位 GFlops)
        (C) = 256×256×256 の Performance (単位 GFlops)

    (2) 実際の FLOPS (と SCORE) は表の 10/13

GTX580 のキャッシュの効果が良くわかる結果となりました。

  - 1 は shared memory を使用して、global memory へのアクセスを減らし
    (要素あたり r/w 各1回のみ)、かつ coalesced になるように工夫されている。

  - 2 と 3 は要素あたり read 5回、write 1回。

  - 通常 iteration 一回あたり 13flops のところ、2 は式を簡略化し
    10flops しか使用していない (`mul.f32` * 1 + `mad.f32` (2flops) * 3 + `add.f32` * 3)。
    flops 計算の式を修正してないので値は 3割水増しだが、工夫の一環として認める。

  - nx=64時には shared memoryを使わない 2 や 3が速い。

  - MySolver 以外のコードをいじった人はいなかった。

各実装および結果の詳細を master branch にマージ済みです。
github (https://github.com/tjhayasaka/pfsl-gpu-exercise/commits/) にて公開しています。

2011 コンペ 
===========

  - HPC のための C/C++ プログラミング

  - CUDA を使用した GPU計算プログラミング

  - エディタ、make、git 等の周辺ツールの使用

の初歩的な技術習得を目的として、コンペティションを開催します。

お題は『3次元拡散方程式』です。

  - 空間は立方体、大きさは一辺が 1.0 で固定

  - 境界条件はノイマン (壁で勾配ゼロ)

  - 初期条件は cosine bell

  - t=0 から 0.1 まで進める

  - 格子は 64^3、128^3、256^3 の三種類で、実行時にプログラムのオプショ
    ンで指定できる。

CPU 計算の実装例と GPU 計算の雛形を github
(https://github.com/tjhayasaka/pfsl-gpu-exercise) にて公開しています。

        $ git clone https://github.com/tjhayasaka/pfsl-gpu-exercise.git

で、手元のマシンにコピーしてください
(git がインストールされていない場合はインストールしてください)。
古い root certificates がインストールされている等が原因で接続できないことがあります。
この問題は `GIT_SSL_NO_VERIFY=true` で回避できることもありますがおすすめしません。

また、実装例と雛形は更新されることがあるので、適宜 git pull するようにしてください。
案内等も含め更新を逃さないように、github にアカウントを作って watch するか、
RSS リーダで巡回することをお勧めします。

コンパイルには CUDA の SDK の他、g++ と boost が必要です。

CPU 計算の実装例は solver/cpu@reference@example.org にあります。

GPU 計算の雛形 solver/gpu-blank@reference@example.org を
各自のディレクトリ (solver/gpu@メールアドレス、
例えば solver/gpu@hayasaka@pfsl.mech.tohoku.ac.jp)
にコピーし、中身を完成させてください。実質的に書換える必要があるのは
MySolver.h 内の二つの関数だけです。性能や保守性の向上のために他の部分を
改良することも歓迎します。

初期値は main が与えます。境界条件は各自のコードで適用してください。

CPU 計算の実装例では、配列のサイズとして各軸方向に +2 を使っていますが、
GPU 計算ではどのような配列のサイズを使ってもかまいません。

計算結果が正しいかどうかは、`double` で計算したときに CPU と GPU の
Error で表示される計算結果が (ほぼ) 完全に一致していることで確認するこ
とができます。

参加
----

有効な email address を持つ個人の方であれば、どなたでも参加できます。
山口研関係者以外でも、プロでも可。
『拡散方程式』やその数値解法を知らなくても、実際の計算方法は簡単なので
CPU 実装例から容易に理解できるはずです。

参加登録の必要はありません。提出をもって参加とします。

提出、連絡先
------------

2011年 7月 4日(月) 00時00分までに以下のいずれかの方法で
hayasaka@pfsl.mech.tohoku.ac.jp までソースコードを提出してください。

  - github 上の自分専用のフォークに push する。pull request を出すか、
    メールでブランチ名を教えてください。

  - ソースコードを .tar.bz2 などにアーカイブしてメールで送る。

提出されたファイルは、hayasaka@pfsl.mech.tohoku.ac.jp がまとめた上で
github 上で公開する予定です。著作権や特許に注意してください。また
email address や氏名の公開を拒否する場合は、提出時にその旨を明示してく
ださい。

質問等も上記のアドレスまでお願いします。

勝敗
----

勝敗判定は、hayasaka@pfsl.mech.tohoku.ac.jp が行います。勝敗は、山口研
の GTX580 で `float` での計算速度を計測した結果の SCORE で決定します。
提出方法、テストのしやすさ等は、勝敗判定の上では考慮しません。

SCORE は青木研主催の 2010年と同じで、

        SCORE = (A)/30 + (B)/40 + (C)/50
        (A) = 64×64×64 の Performance (単位 GFlops)
        (B) = 128×128×128 の Performance (単位 GFlops)
        (C) = 256×256×256 の Performance (単位 GFlops)

(A) の比率がかなり高いので注意してください。

賞品はありません。

実装例について
==============

Debian GNU/Linux (sid) 32bit、CUDA SDK 4.0 + driver 270.41.19、g++-4.4、
libboost 1.46.1、C2Q 9550、GeForce 8600 GT でコンパイル、実行を確認済み。

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
       中身が空の GPU 版雛形 - gpu-blank@reference@example.org

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

マクロ: `FLOAT`
-----------

プリプロセッサマクロ `FLOAT` で、計算に使用する浮動小数点型を指定できる
ようにしてある (今のところ `float` か `double` のみ)。`FLOAT` を直接参
照しているのは `main()` 内のみ、としたかったのだが、コンパイル時間短縮
のため solver/gpu@hayasaka@pfsl.mech.tohoku.ac.jp/gpu_impl.h にも参照個
所がある。他の部分からは、テンプレートのパラメータ `Float` として参照さ
れる。

型: `Vec3<T>`
-------------

cuda の `float3` 等のテンプレート化されたラッパとして `Vec3<T>` を定義
している。`T` としてはいまのところ `unsigned int`, `int`, `float`,
`double` のみが使用可能。

型: `Float`
-----------

計算式中に `Float(1.0)` など、定数をキャストしているところがある。例えば、

        Float a; /*...*/ a = 1.0 - cos(a);

ではなく

        Float a; /*...*/ a = Float(1.0) - cos(a);

のように記述している。前者と後者は `Float` が `double` の時は同じ動作だ
が `float` の時はそれぞれ

        前者:
        a = float(1.0 - cos((double)a)); // C の場合
        a = float(1.0 - double(cosf(a))); // C++ の場合

        後者:
        a = 1.0f - cosf(0.5f);

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
