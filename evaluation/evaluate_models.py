"""
各モデル(XGBoost, RandomForest, Logistic, LightGBM)の予測結果を読み込み、
評価指標(Accuracy, F1, ROC-AUC, Precision, Recall)と
複勝回収率(モデル1位/人気1位/モデル上位3頭/人気上位3頭)をまとめて出力する関数です。

想定ディレクトリ構成:
    horse_racing/
        data/               features.csv, odds.csv など
        learn_model/
            XGBoost.py
        evaluation/
            calc_roi.py
            evaluate_models.py  ← このファイル
            sweep_and_plot.py
        results/            (自動生成)
        models/             (自動生成)

パスはすべて「このファイルの場所」を基準に自動計算するので、
どのディレクトリから実行しても(cdしなくても)動きます。

【変更点】
- 画面出力だけでなく、指標を辞書として返す get_model_metrics() を追加した。
  sweep_and_plot.py から呼び出して、グラフ描画用のデータを集める。
- 存在しない結果CSV(例: XGBoostだけ実験中でRandomForest/Logisticの結果がまだない)
  があってもエラーで止まらないようにした。
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from calc_roi import evaluate_roi, evaluate_roi_by_threshold, evaluate_roi_top1_by_threshold

# horse_racing/evaluation/evaluate_models.py なので、1つ上がプロジェクトルート(horse_racing/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


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
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


def get_model_metrics(name: str, result_path: str, odds_df: pd.DataFrame) -> dict:
    """
    1つのモデルについて、分類指標と回収率(headlineの4種類)をまとめて
    1つのフラットな辞書として返す関数です。
    グラフ化やCSV保存など、プログラムから扱いたい場合はこちらを使います。

    戻り値の例:
        {
            "Accuracy": 0.7069, "F1": 0.4806, "ROC-AUC": 0.7449,
            "Precision": 0.3913, "Recall": 0.6230,
            "モデル1位回収率": 84.14, "人気1位回収率": 83.54,
            "モデル上位3頭回収率": 82.50, "人気上位3頭回収率": 82.11,
        }
    """
    test_result = pd.read_csv(result_path)

    metrics = {}
    metrics.update(evaluate_classification_metrics(test_result))
    metrics.update(evaluate_roi(test_result, odds_df, model_score_col="予測確率"))
    return metrics


def evaluate_model(name: str, result_path: str, odds_df: pd.DataFrame) -> None:
    """
    1つのモデルについて、分類指標と回収率を画面に表示する関数です。
    (これまで通りの手動確認用。中身はget_model_metrics()と同じ計算を再利用)

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


def parse_args():
    parser = argparse.ArgumentParser(description="モデルの予測結果を評価する")
    parser.add_argument("--odds_path", type=str,
                         default=str(PROJECT_ROOT / "data" / "19860105-20210731_odds.csv"))
    parser.add_argument("--xgboost_path", type=str,
                         default=str(PROJECT_ROOT / "results" / "xgboost_test_results.csv"))
    parser.add_argument("--randomforest_path", type=str,
                         default=str(PROJECT_ROOT / "results" / "randomforest_test_results.csv"))
    parser.add_argument("--logistic_path", type=str,
                         default=str(PROJECT_ROOT / "results" / "logistic_test_results.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    odds_df = pd.read_csv(args.odds_path, low_memory=False)

    model_result_paths = {
        "XGBoost": args.xgboost_path,
        # "LightGBM":     "results/lightgbm_test_results.csv",
        "RandomForest": args.randomforest_path,
        "Logistic": args.logistic_path,
    }

    for model_name, path in model_result_paths.items():
        try:
            evaluate_model(model_name, path, odds_df)
        except FileNotFoundError:
            print(f"\n{'='*10} {model_name} {'='*10}")
            print(f"[スキップ] {path} が見つかりません。まだ学習・保存していない場合は無視してください。")