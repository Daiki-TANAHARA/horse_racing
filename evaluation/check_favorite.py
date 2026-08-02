import pandas as pd


# 読み込み
result_df = pd.read_csv(
    "data/19860105-20210731_race_result.csv",
    low_memory=False
)

odds_df = pd.read_csv(
    "data/19860105-20210731_odds.csv",
    low_memory=False
)


# ① 各レースの1番人気を選ぶ
favorite = result_df[result_df["人気"] == 1][
    ["レースID", "馬番"]
].copy()


# ② 複勝払戻表を作る
payouts = []

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

    payouts.append(tmp)


payout_df = pd.concat(payouts)

payout_df = payout_df.dropna(
    subset=["馬番"]
)

payout_df["馬番"] = payout_df["馬番"].astype(int)


# ③ 払戻を付ける
favorite = favorite.merge(
    payout_df,
    on=["レースID", "馬番"],
    how="left"
)

favorite["複勝払戻金"] = (
    favorite["複勝払戻金"]
    .fillna(0)
)


# ④ 回収率計算
total_bet = len(favorite) * 100
total_return = favorite["複勝払戻金"].sum()

roi = total_return / total_bet * 100


print("購入レース数:", len(favorite))
print("総投資:", total_bet)
print("総払戻:", total_return)
print(f"回収率: {roi:.2f}%")