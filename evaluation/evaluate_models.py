"""
各モデル(XGBoost, RandomForest, Logistic)の予測結果を読み込み、
評価指標(Accuracy, F1, ROC-AUC, Precision, Recall)と
複勝回収率(モデル1位/人気1位/モデル上位3頭/人気上位3頭)をまとめて出力する関数です。
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from calc_roi import evaluate_roi, evaluate_roi_by_threshold, evaluate_roi_top1_by_threshold


def evaluate_classification_metrics(test_result: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    予測確率と正解ラベルから、二値分類の評価指標をまとめて計算する関数です。

    引数:
        test_result: 「複勝」「予測確率」列を持つデータフレーム
        threshold: 陽性と判定する確率のしきい値(デフォルト0.5)

    戻り値:
        Accuracy, F1, ROC-AUC, Precision, Recall を格納した辞書
    """
    y_true = test_result["複勝"]
    y_proba = test_result["予測確率"]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(name: str, result_path: str, odds_df: pd.DataFrame) -> None:
    """
    1つのモデルについて、分類指標と回収率をまとめて表示する関数です。

    引数:
        name: 表示用のモデル名(例:"XGBoost")
        result_path: 予測結果CSVのパス
        odds_df: odds.csvを読み込んだデータフレーム
    """
    test_result = pd.read_csv(result_path)

    print(f"\n{'='*10} {name} {'='*10}")
    
    classification_metrics = evaluate_classification_metrics(test_result)
    print("--- 分類指標 ---")
    for key, value in classification_metrics.items():
        print(f"{key}: {value:.4f}")

    # 全レース参加方式
    roi_result = evaluate_roi(test_result, odds_df, model_score_col="予測確率")
    print("--- 回収率 ---")
    for key, value in roi_result.items():
        print(f"{key}: {value:.2f}%")

    # 閾値方式
    threshold_result = evaluate_roi_by_threshold(
        test_result, odds_df,
        model_score_col="予測確率",
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
    )
    print("--- 回収率(閾値&全頭方式) ---")
    print(threshold_result.to_string(index=False))

    threshold_result = evaluate_roi_top1_by_threshold(
        test_result, odds_df,
        model_score_col="予測確率",
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
    )
    print("--- 回収率(閾値&1位方式) ---")
    print(threshold_result.to_string(index=False))


if __name__ == "__main__":
    odds_df = pd.read_csv("../data/19860105-20210731_odds.csv", low_memory=False)

    model_result_paths = {
        "XGBoost":      "../results/xgboost_test_results.csv",
        "RandomForest": "../results/randomforest_test_results.csv",
        "Logistic":     "../results/logistic_test_results.csv",
    }

    for model_name, path in model_result_paths.items():
        evaluate_model(model_name, path, odds_df)