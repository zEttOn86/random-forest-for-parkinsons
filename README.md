
## 目的
　Web上のデータセットを用いて，パーキンソン病であるかないかの2クラス識別問題を解く．

### 実験試料
本実験では，Web上のデータセットを用いた．データセットは，パーキンソン病147名，正常48名，計195名の音声データから求めた特徴量とクラス名から構成される．特徴量には，全部で22個の特徴量が存在する．

[データセットURL](http://archive.ics.uci.edu/ml/datasets/Parkinsons)

### 手法
Random Forestを用いた．

ランダムフォレストのパラメータを選択するにあたり，グリッドサーチを行った．決定木の個数は[50,100,200,300,400,500]，決定木の高さは[3, 5, 10, 15, 20, 25, 30, 40, 50, 100]，seedは[0, 10, 100]の範囲で行った．最適なハイパーパラメータを求めるにあたり，訓練データを用いて，2foldのグリッドサーチを行い，最適なハイパーパラメータを求めた．次に，その最適なパラメータを用いて，訓練データの全てを用いて再学習を行い，テストを行った．

### 結果
コンフュージョンマトリクスを以下に示す．
![Figure1](/results/figure1.png)