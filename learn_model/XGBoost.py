"""
競馬複勝予測 - XGBoostモデル
評価指標: F1スコア, ROC-AUC, Precision, Recall, Accuracy

想定ディレクトリ構成:
    horse_racing/
        data/               features.csv, odds.csv など
        learn_model/
            XGBoost.py      ← このファイル
        evaluation/
            calc_roi.py
            evaluate_models.py
            sweep_and_plot.py
        results/            (自動生成)
        models/             (自動生成)

パスはすべて「このファイル(XGBoost.py)の場所」を基準に自動計算するので、
どのディレクトリから実行しても(cdしなくても)動きます。

【変更点】
パラメータ調整をシェルスクリプトから自動で回せるように、
主要なハイパーパラメータをコマンドライン引数で指定できるようにした。
(引数を何も指定しなければ、これまでと全く同じ設定で動く)

例:
    python XGBoost.py                              # デフォルト設定
    python XGBoost.py --max_depth 6                # max_depthだけ変更
    python XGBoost.py --max_depth 6 --learning_rate 0.05
    python XGBoost.py --scale_pos_weight 1.5        # neg/pos の代わりに固定値
    python XGBoost.py --scale_pos_weight auto       # neg/pos を毎foldで計算(デフォルト)
    python XGBoost.py --reg_lambda 2 --reg_alpha 0.5
    python XGBoost.py --output results/xgboost_test_results_depth6.csv
"""

import argparse
import os
from pathlib import Path

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
import matplotlib
matplotlib.use("Agg")  # 画面がない環境(サーバー/CI)でも落ちないように
import matplotlib.pyplot as plt
try:
    plt.rcParams["font.family"] = "Hiragino Sans"
except Exception:
    pass
from xgboost import plot_importance

# horse_racing/learn_model/XGBoost.py なので、1つ上がプロジェクトルート(horse_racing/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ─────────────────────────────
# 0. コマンドライン引数
# ─────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="XGBoostで複勝予測モデルを学習する")

    parser.add_argument("--data_path", type=str,
                         default=str(PROJECT_ROOT / "data" / "features.csv"))
    parser.add_argument("--output", type=str,
                         default=str(PROJECT_ROOT / "results" / "xgboost_test_results.csv"),
                         help="fold横断の予測結果CSVの保存先")
    parser.add_argument("--model_dir", type=str,
                         default=str(PROJECT_ROOT / "models"))

    # ── チューニング対象のハイパーパラメータ ──
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument(
        "--scale_pos_weight", type=str, default="auto",
        help='"auto"ならfoldごとにneg/posを計算。数値(例: 1.5)を渡すと固定値を使う。'
    )
    parser.add_argument("--reg_lambda", type=float, default=1.0, help="L2正則化係数")
    parser.add_argument("--reg_alpha", type=float, default=0.0, help="L1正則化係数")

    parser.add_argument("--quiet", action="store_true", help="fold毎の詳細出力を抑制する")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    # 同じ実行でmodelファイルが上書きされないよう、出力ファイル名からtagを作る
    run_tag = os.path.splitext(os.path.basename(args.output))[0]

    # ─────────────────────────────
    # 1. データ読込
    # ─────────────────────────────
    df = pd.read_csv(args.data_path, low_memory=False)
    df["レース日付"] = pd.to_datetime(df["レース日付"])
    df = df.sort_values("レース日付")

    features = [
        "距離(m)",
        "馬齢",
        "馬体重",
        "斤量",
        "馬体重増減",
        "枠番",
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
        "芝・ダート別複勝率",
        "競馬場別複勝率",
        "距離帯別複勝率",
        "過去3走平均着順順位率",
        "過去5走平均着順順位率",
        "過去3走平均上り順位率",
        "過去5走平均上り順位率",
        "過去3走複勝率順位率",
        "過去5走複勝率順位率",
        "騎手複勝率順位率",
        "調教師複勝率順位率",
        "芝・ダート別複勝率順位率",
        "距離帯別複勝率順位率",
        "競馬場別複勝率順位率",
        "過去3走平均着順標準化",
        "過去5走平均着順標準化",
        "過去3走平均上り標準化",
        "過去5走平均上り標準化",
        "過去3走複勝率標準化",
        "過去5走複勝率標準化",
        "騎手複勝率標準化",
        "調教師複勝率標準化",
        "芝・ダート別複勝率標準化",
        "距離帯別複勝率標準化",
        "競馬場別複勝率標準化",
        "斤量順位率",
        "馬齢順位率",
        "馬体重順位率",
        "馬体重増減順位率",
        "休養日数順位率",
        "斤量標準化",
        "馬齢標準化",
        "馬体重標準化",
        "馬体重増減標準化",
        "休養日数標準化",
        "芝・ダート区分_ダート",
        "芝・ダート区分_芝",
        "性別_セ",
        "性別_牝",
        "性別_牡",
        "競馬場名_中京",
        "競馬場名_中山",
        "競馬場名_京都",
        "競馬場名_函館",
        "競馬場名_小倉",
        "競馬場名_新潟",
        "競馬場名_札幌",
        "競馬場名_東京",
        "競馬場名_福島",
        "競馬場名_阪神",
        "馬場状態1_不良",
        "馬場状態1_稍重",
        "馬場状態1_良",
        "馬場状態1_重",
        "右左回り・直線区分_右",
        "右左回り・直線区分_左",
        "右左回り・直線区分_直線",
        "天候_小雨",
        "天候_小雪",
        "天候_晴",
        "天候_曇",
        "天候_雨",
        "天候_雪",
        "距離帯_マイル",
        "距離帯_中距離",
        "距離帯_中長距離",
        "距離帯_短距離",
        "距離帯_長距離",
        "レースクラス_1勝クラス",
        "レースクラス_2勝クラス",
        "レースクラス_3勝クラス",
        "レースクラス_オープン",
        "レースクラス_新馬",
        "レースクラス_未勝利",
        "レースグレード_G",
        "レースグレード_G1",
        "レースグレード_G2",
        "レースグレード_G3",
        "レースグレード_L",
        "レースグレード_なし",
    ]
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
    model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(race_ids), 1):
        train_df = df[df["レースID"].isin(race_ids[train_idx])]
        test_df = df[df["レースID"].isin(race_ids[test_idx])]

        if not args.quiet:
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
        X_test, y_test = test_df[features], test_df[target]

        # ─────────────────────────────
        # 3. モデル定義・学習
        # ─────────────────────────────
        if args.scale_pos_weight == "auto":
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            spw = neg / pos
        else:
            spw = float(args.scale_pos_weight)

        model = XGBClassifier(
            objective="binary:logistic",  # 何を予測するか（今回は2値分類）を指定
            eval_metric="logloss",  # 予測確率が実際の結果にどれだけ近いかを測る指標
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            scale_pos_weight=spw,  # クラス不均衡の補正
            reg_lambda=args.reg_lambda,  # L2正則化
            reg_alpha=args.reg_alpha,  # L1正則化
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )
        model.save_model(f"{args.model_dir}/xgboost_{run_tag}_fold{fold}.json")

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
            "Fold": fold,
            "Accuracy": accuracy_score(y_test, pred_label),
            "F1": f1_score(y_test, pred_label, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, pred_proba),
            "Precision": precision_score(y_test, pred_label, zero_division=0),
            "Recall": recall_score(y_test, pred_label, zero_division=0),
        })

        if not args.quiet:
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
    all_test_results.to_csv(args.output, index=False)
    print(f"{args.output} を保存しました。")

    # ─────────────────────────────
    # 6. 集計
    # ─────────────────────────────
    results_df = pd.DataFrame(results).set_index("Fold")
    print("\n=== XGBoost 平均スコア ===")
    print(results_df.mean().to_string())

    # ─────────────────────────────
    # 7. 特徴量重要度
    # ─────────────────────────────
    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)
    if not args.quiet:
        print("\n=== 特徴量重要度 ===")
        print(importance.to_string())


if __name__ == "__main__":
    main()