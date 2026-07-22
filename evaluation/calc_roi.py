"""
複勝回収率(ROI)計算のための共通関数をまとめたファイルです。
XGBoost, RandomForest, Logisticなど、どのモデルの予測結果からも呼び出せます。
"""
import pandas as pd


def build_place_payout_lookup(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    横持ちのoddsデータから「レースID・馬番・複勝払戻金」の対応表を作る関数です。

    odds.csvの「複勝1_馬番」〜「複勝3_馬番」は、実際に複勝(3着以内)した馬の馬番、
    「複勝1_オッズ」〜「複勝3_オッズ」はその馬に100円賭けた場合の払戻金です。
    (7頭以下のレースでは複勝3が存在せず、欠損になります)

    引数:
        odds_df: odds.csvを読み込んだデータフレーム

    戻り値:
        レースID, 馬番, 複勝払戻金 の3列を持つデータフレーム
        (複勝1〜3に登場する馬=複勝的中馬のみを含む)
    """
    place_records = []

    for i in range(1, 4):  # 複勝1〜3
        tmp = odds_df[["レースID", f"複勝{i}_馬番", f"複勝{i}_オッズ"]].copy()
        tmp = tmp.rename(columns={
            f"複勝{i}_馬番": "馬番",
            f"複勝{i}_オッズ": "複勝払戻金",
        })
        place_records.append(tmp)

    place_payout = pd.concat(place_records, ignore_index=True)
    place_payout = place_payout.dropna(subset=["馬番"])
    place_payout["馬番"] = place_payout["馬番"].astype(int)

    return place_payout


def attach_payout(test_result: pd.DataFrame, place_payout: pd.DataFrame) -> pd.DataFrame:
    """
    予測結果データに複勝払戻金を結合する関数です。
    複勝1〜3に載っていない(=複勝しなかった)馬は、払戻金0円として扱います。

    引数:
        test_result: 「レースID」「馬番」列を持つ予測結果データ
        place_payout: build_place_payout_lookup で作成した払戻金対応表

    戻り値:
        「複勝払戻金」列を追加したデータフレーム
    """
    result = test_result.merge(
        place_payout,
        on=["レースID", "馬番"],
        how="left"
    )
    result["複勝払戻金"] = result["複勝払戻金"].fillna(0)

    return result


def select_top_n(
    test_result: pd.DataFrame,
    score_col: str,
    n: int,
    ascending: bool
) -> pd.DataFrame:
    """
    レースごとに、指定したスコア列で上位n頭を選ぶ関数です。

    引数:
        test_result: 予測結果データ(「レースID」列が必要)
        score_col: 順位付けに使う列名(例:「予測確率」「人気」)
        n: 選ぶ頭数
        ascending: True なら値が小さい方が上位(人気など)、
                   False なら値が大きい方が上位(予測確率など)

    戻り値:
        各レースの上位n頭だけを抽出したデータフレーム
    """
    return (
        test_result
        .sort_values(["レースID", score_col], ascending=[True, ascending])
        .groupby("レースID")
        .head(n)
    )


def calc_return_rate(bet_df: pd.DataFrame, bet_per_horse: int = 100) -> float:
    """
    賭けた馬(1行1賭け)についての回収率を計算する関数です。
    複数Foldの結果をまとめて渡すことで、全期間プールでの回収率を計算できます。

    引数:
        bet_df: 「複勝払戻金」列を持つ、賭けた馬のデータフレーム
        bet_per_horse: 1頭あたりの購入金額(デフォルト100円)

    戻り値:
        回収率(%)。100を超えればプラス収支。
    """
    total_bet = len(bet_df) * bet_per_horse
    total_return = bet_df["複勝払戻金"].sum()

    return total_return / total_bet * 100


def evaluate_roi(
    test_result: pd.DataFrame,
    odds_df: pd.DataFrame,
    model_score_col: str
) -> dict:
    """
    モデル1位・人気1位・モデル上位3頭・人気上位3頭、それぞれの複勝回収率を
    まとめて計算する関数です。

    引数:
        test_result: 全Fold分の予測結果を concat した全期間データ
                     (「レースID」「馬番」「人気」「複勝」およびmodel_score_colの列が必要)
        odds_df: odds.csvを読み込んだデータフレーム
        model_score_col: モデルの予測スコア列名(例:「予測確率」)

    戻り値:
        4種類の回収率(%)を格納した辞書
    """
    place_payout = build_place_payout_lookup(odds_df)
    test_result = attach_payout(test_result, place_payout)

    model_top1   = select_top_n(test_result, model_score_col, n=1, ascending=False)
    popular_top1 = select_top_n(test_result, "人気",           n=1, ascending=True)
    model_top3   = select_top_n(test_result, model_score_col, n=3, ascending=False)
    popular_top3 = select_top_n(test_result, "人気",           n=3, ascending=True)

    return {
        "モデル1位回収率":     calc_return_rate(model_top1),
        "人気1位回収率":       calc_return_rate(popular_top1),
        "モデル上位3頭回収率": calc_return_rate(model_top3),
        "人気上位3頭回収率":   calc_return_rate(popular_top3),
    }

def select_by_threshold(
    test_result: pd.DataFrame,
    score_col: str,
    threshold: float
) -> pd.DataFrame:
    """
    予測スコアが閾値以上の馬だけを選ぶ関数です。
    (レースごとに複数該当する場合もあれば、1頭も該当しないレースもありえます)

    引数:
        test_result: 予測結果データ
        score_col: 判定に使うスコア列名(例:「予測確率」)
        threshold: この値以上のスコアを持つ馬だけを選ぶ

    戻り値:
        条件を満たす馬だけのデータフレーム
    """
    return test_result[test_result[score_col] >= threshold]


def evaluate_roi_by_threshold(
    test_result: pd.DataFrame,
    odds_df: pd.DataFrame,
    model_score_col: str,
    thresholds: list[float]
) -> pd.DataFrame:
    """
    複数の閾値について、回収率とカバー率をまとめて計算する関数です。

    引数:
        test_result: 全Fold分の予測結果を concat した全期間データ
        odds_df: odds.csvを読み込んだデータフレーム
        model_score_col: モデルの予測スコア列名(例:「予測確率」)
        thresholds: 試したい閾値のリスト(例:[0.5, 0.6, 0.7, 0.8])

    戻り値:
        閾値ごとの回収率・賭けた頭数・カバーレース数をまとめたデータフレーム
    """
    place_payout = build_place_payout_lookup(odds_df)
    test_result = attach_payout(test_result, place_payout)

    total_races = test_result["レースID"].nunique()
    records = []

    for threshold in thresholds:
        bets = select_by_threshold(test_result, model_score_col, threshold)

        if len(bets) == 0:
            records.append({
                "閾値": threshold,
                "回収率": None,
                "賭けた頭数": 0,
                "カバーレース数": 0,
                "カバー率": 0.0,
            })
            continue

        roi = calc_return_rate(bets)
        n_races_covered = bets["レースID"].nunique()

        records.append({
            "閾値": threshold,
            "回収率": roi,
            "賭けた頭数": len(bets),
            "カバーレース数": n_races_covered,
            "カバー率": n_races_covered / total_races * 100,
        })

    return pd.DataFrame(records)