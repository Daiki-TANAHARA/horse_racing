import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# 日本語フォント設定（Mac用）
plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

def build_place_payout_lookup(odds_df):
    records = []

    for i in range(1, 6):
        tmp = odds_df[
            [
                "レースID",
                f"複勝{i}_馬番",
                f"複勝{i}_オッズ"
            ]
        ].copy()

        tmp.columns = [
            "レースID",
            "馬番",
            "複勝払戻金"
        ]

        records.append(tmp)

    payout = pd.concat(records)

    payout = payout.dropna(subset=["馬番"])
    payout["馬番"] = payout["馬番"].astype(int)

    return payout


# =====================
# データ読み込み
# =====================

result_df = pd.read_csv(
    "data/19860105-20210731_race_result.csv",
    low_memory=False
)

odds_df = pd.read_csv(
    "data/19860105-20210731_odds.csv",
    low_memory=False
)


# =====================
# 1番人気を選択
# =====================

favorite = result_df[
    result_df["人気"] == 1
][
    [
        "レースID",
        "馬番",
        "レース日付"
    ]
].copy()


favorite["レース日付"] = pd.to_datetime(
    favorite["レース日付"]
)


# =====================
# 払戻を付与
# =====================

payout = build_place_payout_lookup(odds_df)

favorite = favorite.merge(
    payout,
    on=["レースID", "馬番"],
    how="left"
)

favorite["複勝払戻金"] = (
    favorite["複勝払戻金"]
    .fillna(0)
)


# =====================
# 累積損益計算
# =====================

favorite = favorite.sort_values(
    ["レース日付", "レースID"]
)

favorite["損益"] = (
    favorite["複勝払戻金"] - 100
)

favorite["累積損益"] = (
    favorite["損益"]
    .cumsum()
)


# =====================
# グラフ作成
# =====================

plt.figure(figsize=(12, 6))

plt.plot(
    range(len(favorite)),
    favorite["累積損益"],
    color="blue"
)

plt.axhline(
    0,
    color="gray",
    linestyle="--"
)

plt.title(
    "1番人気の馬の複勝を100円購入した場合の累積損益"
)

plt.xlabel(
    "購入レース数"
)

plt.ylabel(
    "累積損益(円)"
)

plt.grid(alpha=0.3)


roi = (
    favorite["複勝払戻金"].sum()
    /
    (len(favorite) * 100)
    *
    100
)

plt.text(
    0.02,
    0.05,
    f"回収率: {roi:.2f}%\n"
    f"最終損益: {favorite['累積損益'].iloc[-1]:.0f}円",
    transform=plt.gca().transAxes,
    bbox={
        "facecolor": "white",
        "alpha": 0.8
    }
)


Path("results").mkdir(
    exist_ok=True
)

plt.savefig(
    "results/favorite_place_roi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "保存しました: results/favorite_place_roi.png"
)