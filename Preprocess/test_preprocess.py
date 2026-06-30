import pandas as pd
from preprocess import preprocess

df = pd.read_csv("archive/19860105-20210731_race_result.csv")

features = [
    "レースID",
    "芝・ダート区分",
    "距離(m)",
    "着順",
    "馬齢",
    "馬体重"
]

df2 = preprocess(df, features)


print("=== 前処理後データ ===")
print(df2.head())

print("\n=== dtype確認 ===")
print(df2.dtypes)

print("\n=== 欠損チェック ===")
print(df2.isnull().sum())

print("\n=== shape ===")
print(df2.shape)

print(df["芝・ダート区分"].dtype)