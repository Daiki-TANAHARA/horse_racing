"""
競馬複勝予測 LightGBM学習スクリプト

前提フォルダ構成:
horse_racing/
├── data/
│   └── 19860105-20210731_race_result.csv
├── Preprocess/
│   └── preprocess.py
└── learn_model/
    └── LightGBM.py   ← このファイル

実行方法(learn_model フォルダの中から):
    uv run LightGBM.py
"""

import os
import sys

# ---- Preprocess フォルダを import パスに追加 ----
# このファイル(learn_model/LightGBM.py)から見て、
# 一つ上の階層(horse_racing/)にある Preprocess フォルダを指す
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(CURRENT_DIR, "..", "Preprocess")
sys.path.append(PREPROCESS_DIR)

from claude_preprocess import read_race_csv, preprocess_train_valid  # noqa: E402

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # 画面表示なしでファイル保存するための設定
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# ---- 日本語(特徴量名)が図で文字化けしないようフォントを設定 ----
# 環境に日本語フォントがあれば使う。無ければ警告だけ出して英字のまま進める。
_JP_FONT_CANDIDATES = ["Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans", "Meiryo", "MS Gothic"]
for _font_name in _JP_FONT_CANDIDATES:
    if any(_font_name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _font_name
        break
else:
    print("警告: 日本語フォントが見つからないため、図中の特徴量名が文字化けする可能性があります。")
    print("      日本語表示にしたい場合は `pip install japanize-matplotlib` を検討してください。")


# ============================================================
# 1. 設定
# ============================================================

# CSVファイルのパス(learn_model から見て ../data/ 配下)
CSV_PATH = os.path.join(CURRENT_DIR, "..", "data", "19860105-20210731_race_result.csv")

# 検証データの境界日(この日付以降をvalidデータとする)
SPLIT_DATE = "2019-01-01"

# 学習データの開始日(この日付より前のデータは学習に使わない)
# JRAは2001年度から馬齢の数え方を「数え年」→「満年齢」に変更しており、
# それ以前と以降ではデータの前提が異なるため、2001年以降だけを使う。
# 全期間を使いたい場合は None にする。
CUTOFF_DATE = "2001-01-01"

# 使用する特徴量(「着順」は目的変数作成に必要なので必ず含める)
FEATURES = [
    "レースID", "レース日付", "着順",
    "距離(m)", "斤量", "馬齢", "馬体重", "枠番", "馬番",
    "騎手", "調教師", "芝・ダート区分", "競馬場名", "天候", "馬場状態1",
    "past3_hukusho_rate", "past5_hukusho_rate", "past9_hukusho_rate",
    "past5_agari_mean", "past5_chakujun_mean", "career_count",
    "days_since_last_race", "prev_chakujun", "prev_agari",
    "past_dist_hukusho_rate", "past_track_hukusho_rate",
    "jockey_hukusho_rate", "trainer_hukusho_rate", "jockey_course_rate",
    "past5_hukusho_rate_race_z", "past5_agari_mean_race_z",
    "jockey_hukusho_rate_race_z", "past5_chakujun_mean_race_z",
    "馬体重_race_z",
]

# 学習・評価結果や図の出力先
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. データ読み込み・前処理
# ============================================================

def load_and_preprocess():
    import pandas as pd

    print(f"[1/4] CSV読み込み中... ({CSV_PATH})")
    df = read_race_csv(CSV_PATH, FEATURES)
    print(f"      読み込み完了: {df.shape}")

    if CUTOFF_DATE is not None:
        df["レース日付"] = pd.to_datetime(df["レース日付"])
        before = len(df)
        df = df[df["レース日付"] >= CUTOFF_DATE].copy()
        print(f"      学習データを {CUTOFF_DATE} 以降に絞り込み: {before:,}行 → {len(df):,}行")

    print("[2/4] 特徴量エンジニアリング・前処理中...")
    train_processed, valid_processed, preprocessor = preprocess_train_valid(
        df, FEATURES, split_date=SPLIT_DATE
    )
    print(f"      train: {train_processed.shape}, valid: {valid_processed.shape}")

    return train_processed, valid_processed, preprocessor


# ============================================================
# 3. LightGBM学習
# ============================================================

def train_lightgbm(train_processed, valid_processed):
    print("[3/4] LightGBM学習中...")

    # 学習に使わない列(識別子・日付・目的変数)を除外
    drop_cols = ["複勝", "レースID", "レース日付"]
    feature_cols = [c for c in train_processed.columns if c not in drop_cols]

    X_train = train_processed[feature_cols]
    y_train = train_processed["複勝"]
    X_valid = valid_processed[feature_cols]
    y_valid = valid_processed["複勝"]

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    return model, X_valid, y_valid, feature_cols


# ============================================================
# 4. 評価・可視化
# ============================================================

def evaluate_model(model, X_valid, y_valid):
    print("[4/4] 評価中...")

    pred_proba = model.predict(X_valid, num_iteration=model.best_iteration)
    pred_label = (pred_proba >= 0.7).astype(int)

    auc = roc_auc_score(y_valid, pred_proba)
    acc = accuracy_score(y_valid, pred_label)
    precision = precision_score(y_valid, pred_label)
    recall = recall_score(y_valid, pred_label)

    print("\n===== 評価結果(検証データ) =====")
    print(f"AUC       : {auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}  (複勝と予測した中で実際に当たった割合)")
    print(f"Recall    : {recall:.4f}  (実際の複勝馬を捕捉できた割合)")

    # 結果をテキストファイルに保存
    result_path = os.path.join(OUTPUT_DIR, "evaluation_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("===== 評価結果(検証データ) =====\n")
        f.write(f"AUC       : {auc:.4f}\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
    print(f"\n評価結果を保存しました: {result_path}")

    return pred_proba


def plot_feature_importance(model, feature_cols):
    importance = model.feature_importance(importance_type="gain")
    imp_df = (
        __import__("pandas")
        .DataFrame({"feature": feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(8, 8))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.xlabel("Importance (gain)")
    plt.title("LightGBM Feature Importance (Top 20)")
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"特徴量重要度の図を保存しました: {fig_path}")


def save_model(model):
    model_path = os.path.join(OUTPUT_DIR, "lgb_model.txt")
    model.save_model(model_path)
    print(f"モデルを保存しました: {model_path}")


# ============================================================
# メイン処理
# ============================================================

def main():
    train_processed, valid_processed, preprocessor = load_and_preprocess()
    model, X_valid, y_valid, feature_cols = train_lightgbm(train_processed, valid_processed)
    evaluate_model(model, X_valid, y_valid)
    plot_feature_importance(model, feature_cols)
    save_model(model)


if __name__ == "__main__":
    main()