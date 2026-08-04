"""
競馬複勝予測 - XGBoostモデル
評価指標: F1スコア, ROC-AUC, Precision, Recall, Accuracy
"""

import argparse
import sys
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Hiragino Sans"
from xgboost import plot_importance

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))
from feature_sets import EXPERIMENTS

# ─────────────────────────────
# 0. コマンドライン引数
# ─────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment",
    required=True,
    choices=EXPERIMENTS.keys(),
    help="使用する特徴量セットの実験名(config/feature_sets.py参照)",
)
args = parser.parse_args()

# ─────────────────────────────
# 1. データ読込
# ─────────────────────────────
df = pd.read_csv("data/features.csv", low_memory=False)
df["レース日付"] = pd.to_datetime(df["レース日付"])
df = df.sort_values("レース日付")

features = EXPERIMENTS[args.experiment]
target = "複勝"

# ─────────────────────────────
# 2. 時系列クロスバリデーション
# ─────────────────────────────
race_ids = (
    df[["レースID", "レース日付"]]
    .drop_duplicates()
    .sort_values("レース日付")["レースID"]
    .to_numpy()
)
tscv = TimeSeriesSplit(n_splits=5)

results = []
all_test_results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(race_ids), 1):
    train_df = df[df["レースID"].isin(race_ids[train_idx])]
    test_df  = df[df["レースID"].isin(race_ids[test_idx])]

    print(f"\n===== Fold {fold} =====")
    print(
        f"Train : {train_df['レース日付'].min().date()} ～ "
        f"{train_df['レース日付'].max().date()}"
    )
    print(
        f"Test  : {test_df['レース日付'].min().date()} ～ "
        f"{test_df['レース日付'].max().date()}"
    )

    X_train, y_train = train_df[features], train_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

    # ─────────────────────────────
    # 3. モデル定義・学習
    # ─────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=neg / pos,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    model.save_model(f"models/xgboost_{args.experiment}_fold{fold}.json")

    # ─────────────────────────────
    # 4. 評価
    # ─────────────────────────────
    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_label = model.predict(X_test)

    test_result = test_df[["レースID", "レース日付", "馬番", "人気", "複勝"]].copy()
    test_result["予測確率"] = pred_proba
    test_result["Fold"] = fold
    all_test_results.append(test_result)

    results.append({
        "Fold":      fold,
        "Accuracy":  accuracy_score(y_test, pred_label),
        "F1":        f1_score(y_test, pred_label, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, pred_proba),
        "Precision": precision_score(y_test, pred_label, zero_division=0),
        "Recall":    recall_score(y_test, pred_label, zero_division=0),
    })

    print(f"[Fold {fold}] "
          f"Acc={results[-1]['Accuracy']:.4f}  "
          f"F1={results[-1]['F1']:.4f}  "
          f"AUC={results[-1]['ROC-AUC']:.4f}  "
          f"Prec={results[-1]['Precision']:.4f}  "
          f"Rec={results[-1]['Recall']:.4f}")

# ─────────────────────────────
# 5. 予測結果の保存
# ─────────────────────────────
all_test_results = pd.concat(all_test_results, ignore_index=True)

# 実験ごとの結果保存
experiment_result_path = (f"results/xgboost_{args.experiment}_test_results.csv")
all_test_results.to_csv(experiment_result_path,index=False)
print(f"{experiment_result_path} を保存しました。")


# evaluate_models.py 用の固定ファイル保存
# 常に最新実験結果で上書きする
latest_result_path = "results/xgboost_test_results.csv"
all_test_results.to_csv(latest_result_path,index=False)
print(f"{latest_result_path} を更新しました。")

# ─────────────────────────────
# 6. 集計
# ─────────────────────────────
results_df = pd.DataFrame(results).set_index("Fold")
print(f"\n=== XGBoost({args.experiment}) 平均スコア ===")
print(results_df.mean().to_string())

# ─────────────────────────────
# 7. 特徴量重要度
# ─────────────────────────────
importance = pd.Series(
    model.feature_importances_, index=features
).sort_values(ascending=False)
print("\n=== 特徴量重要度 ===")
print(importance.to_string())