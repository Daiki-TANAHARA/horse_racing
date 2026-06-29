import pandas as pd

# ======================
# 1. 読み込み
# ======================

race_df = pd.read_csv("../data/19860105-20210731_race_result.csv", low_memory=False)

# ======================
# 2. 基本特徴量
# ======================

df = race_df.copy()

# 芝=0 ダート=1
df["芝・ダート区分"] = df["芝・ダート区分"].map({
    "芝": 0,
    "ダート": 1
})

# 性別　牡=0　牝=1　セ=2
df["性別"] = df["性別"].map({
    "牡": 0,
    "牝": 1,
    "セ": 2
})

#馬体重増減
df["馬体重増減"] = df["場体重増減"].fillna(
    df["場体重増減"].median()
)



# ======================
# 3. 目的変数
# ======================

df = df.dropna(subset=["着順"])
df["複勝"] = (df["着順"] <= 3).astype(int)

# ======================
# 4. 使用列
# ======================

columns = [
    "レースID",
    "レース日付",

    "芝・ダート区分",
    "距離(m)",
    "馬齢",
    "馬体重",
    "人気",
    "斤量",
    "性別",
    "単勝",
    "馬体重増減",

    "複勝"
]

missing = set(columns) - set(df.columns)
if missing:
    print("存在しない列:", missing)

features_df = df[columns]

# ======================
# 5. 保存
# ======================

features_df.to_csv(
    "../data/features.csv",
    index=False
)

print("保存完了")

print(features_df.head())