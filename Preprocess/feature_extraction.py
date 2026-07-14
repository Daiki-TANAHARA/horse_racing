import pandas as pd


# features.csvを読み込む
df = pd.read_csv("../data/features.csv", low_memory=False)


# 学習に使う特徴量
feature_columns = [
   col for col in df.columns
   if col not in ["レースID", "レース日付", "複勝"]
]


print("features = [")
for col in feature_columns:
   print(f'    "{col}",')
print("]")
