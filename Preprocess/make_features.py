import pandas as pd
from preprocess import preprocess

import os

print("現在の作業ディレクトリ:", os.getcwd())

# 使用する特徴量
features = [
    "レースID",
    "レース日付",
    "芝・ダート区分",
    "距離(m)",
    "馬齢",
    "馬体重",
    "性別",
    "人気",
    "単勝",
    "過去3走複勝率",
    "過去5走複勝率",
    "前走着順",
    "過去3走平均着順",
    "過去5走平均着順",
    "前走上り",
    "過去3走平均上り",
    "過去5走平均上り",
    "騎手複勝率",
    "調教師複勝率",
    "休養日数",
    "着順",  # preprocess()内で「複勝」を作るために必要
]

# 生データを読み込む
df = pd.read_csv(
    "../data/19860105-20210731_race_result.csv",
    low_memory=False
)

df["レース日付"] = pd.to_datetime(df["レース日付"])

print(df["レース日付"].head())
# 前処理
df = preprocess(df, features)

# 前処理済みデータを保存
df.to_csv("../data/features.csv", index=False)

print("features.csv を作成しました。")

print(df["レース日付"].head())