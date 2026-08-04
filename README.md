# 競馬複勝予測

## 概要
本リポジトリは、競馬の複勝予測を目的とした機械学習モデルの実装です。

以下の3モデルを実装しています。

- XGBoost
- Random Forest
- Logistic Regression


## ディレクトリ構成

```
horse_racing/
├── data/               # データセット
├── config/             # 実験ごとの設定ファイル
├── learn_model/        # 学習プログラム
├── evaluation/         # 評価プログラム
├── run_scripts/        # 実験実行用シェルスクリプト
├── models/             # 学習済みモデル（自動生成）
└── results/            # 実験結果（自動生成）
```

## セットアップ

### 1. リポジトリを取得

```bash
git clone <リポジトリURL>
cd horse_racing
```

### 2. 必要なライブラリをインストール

```bash
uv sync
```

## データセット

本研究では Kaggle の以下のデータセットを利用しています。<br>
https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset

- 19860105-20210731_race_result.csv
- 19860105-20210731_odds.csv

これらを `data/` ディレクトリへ配置してください。

```
data/
├── 19860105-20210731_race_result.csv
└── 19860105-20210731_odds.csv
```

## 実験の実行方法

### 特徴量作成(最初に1回だけ実行)
```bash
./run_scripts/build_features.sh
```

### 実験1

```bash
./run_scripts/ex1.sh
```

### 実験2

```bash
./run_scripts/ex2.sh
```

### 実験3

```bash
./run_scripts/ex3.sh
```

### 実験4

```bash
./run_scripts/ex4.sh
```

### 実験5

```bash
./run_scripts/ex5.sh
```

### 実験6

実験5と同じ特徴量を用いて回収率を計算します。

```bash
./run_scripts/ex6_1.sh
```

参考実験（実験5 + 単勝オッズ・人気）

```bash
./run_scripts/ex6_2.sh
```

### 実験7

ハイパーパラメータ探索を実行します。

```bash
./run_scripts/ex7.sh
```

## 出力

実行後、以下のディレクトリに結果が保存されます。

### 学習済みモデル

```
models/
```

### 予測結果

```
results/
```
