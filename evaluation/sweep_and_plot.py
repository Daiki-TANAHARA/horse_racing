"""
XGBoostのハイパーパラメータを1項目ずつスイープして、
「値を変えると分類指標・回収率がどう変化するか」をグラフにするスクリプトです。

想定ディレクトリ構成:
    horse_racing/
        data/               features.csv, odds.csv など
        learn_model/
            XGBoost.py
        evaluation/
            calc_roi.py
            evaluate_models.py
            sweep_and_plot.py   ← このファイル
        results/            (自動生成)
        models/             (自動生成)

パスはすべて「このファイルの場所」を基準に自動計算する(=learn_model/XGBoost.py も
自動で見つける)ので、どのディレクトリから実行しても(cdしなくても)動きます。

やっていること:
    1. 候補値ごとに XGBoost.py をサブプロセスで実行 → 予測結果CSVを保存
    2. evaluate_models.py の get_model_metrics() で分類指標・回収率を計算
    3. 候補値 × 指標 の一覧表(CSV)と、折れ線グラフ(PNG)を保存

使い方の例(第2項目: max_depthを4〜10で試す):
    python sweep_and_plot.py --param max_depth --values 4,5,6,7,8,9,10

第3項目(learning_rate)。すでに決めたmax_depth=6を固定しつつ試す場合:
    python sweep_and_plot.py --param learning_rate --values 0.01,0.03,0.05,0.07,0.1 \
        --fixed '{"max_depth": 6}'

第4項目(scale_pos_weight)。"auto"はneg/posを毎foldで自動計算する元の設定:
    python sweep_and_plot.py --param scale_pos_weight --values auto,1.5,1.2,1.0 \
        --fixed '{"max_depth": 6, "learning_rate": 0.05}'

第5項目(L2 = reg_lambda):
    python sweep_and_plot.py --param reg_lambda --values 0,0.5,1,2,5 \
        --fixed '{"max_depth": 6, "learning_rate": 0.05, "scale_pos_weight": 1.2}'

第6項目(L1 = reg_alpha):
    python sweep_and_plot.py --param reg_alpha --values 0,0.5,1,2,5 \
        --fixed '{"max_depth": 6, "learning_rate": 0.05, "scale_pos_weight": 1.2, "reg_lambda": 1}'
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    plt.rcParams["font.family"] = "Hiragino Sans"
except Exception:
    pass

from evaluate_models import get_model_metrics

# horse_racing/evaluation/sweep_and_plot.py なので、1つ上がプロジェクトルート(horse_racing/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# 引数ごとの型(scale_pos_weightだけ"auto"を許すので文字列のまま扱う)
NUMERIC_TYPE = {
    "n_estimators": int,
    "max_depth": int,
    "learning_rate": float,
    "scale_pos_weight": str,
    "reg_lambda": float,
    "reg_alpha": float,
}

CLASSIFICATION_KEYS = ["Accuracy", "F1", "ROC-AUC", "Precision", "Recall"]
ROI_KEYS = ["モデル1位回収率", "人気1位回収率", "モデル上位3頭回収率", "人気上位3頭回収率"]


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoostのパラメータをスイープしてグラフ化する")
    parser.add_argument("--param", type=str, required=True,
                         choices=list(NUMERIC_TYPE.keys()),
                         help="スイープするパラメータ名")
    parser.add_argument("--values", type=str, required=True,
                         help='カンマ区切りの候補値。例: "4,5,6,7,8,9,10" や "auto,1.5,1.2,1.0"')
    parser.add_argument("--fixed", type=str, default="{}",
                         help='他パラメータを固定する場合のJSON文字列。例: \'{"max_depth": 6}\'')
    parser.add_argument("--data_path", type=str,
                         default=str(PROJECT_ROOT / "data" / "features.csv"))
    parser.add_argument("--odds_path", type=str,
                         default=str(PROJECT_ROOT / "data" / "19860105-20210731_odds.csv"))
    parser.add_argument("--xgboost_script", type=str,
                         default=str(PROJECT_ROOT / "learn_model" / "XGBoost.py"))
    parser.add_argument("--outdir", type=str,
                         default=str(PROJECT_ROOT / "results" / "sweep"))
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def sanitize(value: str) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def run_one(python_exe, script, data_path, output_path, param, value, fixed: dict, quiet=False):
    cmd = [
        python_exe, script,
        f"--data_path={data_path}",
        f"--output={output_path}",
    ]
    for k, v in fixed.items():
        cmd.append(f"--{k}={v}")
    cmd.append(f"--{param}={value}")
    if quiet:
        cmd.append("--quiet")

    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    value_type = NUMERIC_TYPE[args.param]
    raw_values = [v.strip() for v in args.values.split(",") if v.strip() != ""]
    values = [v if value_type is str else value_type(v) for v in raw_values]

    fixed = json.loads(args.fixed)

    odds_df = pd.read_csv(args.odds_path, low_memory=False)

    records = []
    for value in values:
        tag = f"{args.param}_{sanitize(value)}"
        output_path = os.path.join(args.outdir, f"xgboost_{tag}.csv")

        run_one(
            python_exe=args.python,
            script=args.xgboost_script,
            data_path=args.data_path,
            output_path=output_path,
            param=args.param,
            value=value,
            fixed=fixed,
            quiet=True,
        )

        metrics = get_model_metrics("XGBoost", output_path, odds_df)
        metrics[args.param] = value
        records.append(metrics)
        print(f"[{args.param}={value}] "
              + "  ".join(f"{k}={metrics[k]:.4f}" for k in CLASSIFICATION_KEYS)
              + "  |  "
              + "  ".join(f"{k}={metrics[k]:.2f}%" for k in ROI_KEYS))

    summary = pd.DataFrame(records)
    summary_path = os.path.join(args.outdir, f"sweep_{args.param}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n一覧表を保存しました: {summary_path}")

    plot_path = plot_summary(summary, args.param, args.outdir)
    print(f"グラフを保存しました: {plot_path}")

    best_idx = summary["モデル上位3頭回収率"].astype(float).idxmax()
    best_value = summary.loc[best_idx, args.param]  # 元のvaluesリストの型のまま取得
    best_roi = summary.loc[best_idx, "モデル上位3頭回収率"]
    print(f"\n『モデル上位3頭回収率』が最も高いのは {args.param}={best_value} "
          f"({best_roi:.2f}%) でした。")


def plot_summary(summary: pd.DataFrame, param: str, outdir: str) -> str:
    x_labels = summary[param].astype(str).tolist()
    x = range(len(x_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    for key in CLASSIFICATION_KEYS:
        ax1.plot(x, summary[key], marker="o", label=key)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel(param)
    ax1.set_ylabel("score")
    ax1.set_title(f"分類指標 vs {param}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for key in ROI_KEYS:
        ax2.plot(x, summary[key], marker="o", label=key)
    ax2.axhline(100, color="gray", linestyle="--", linewidth=1, label="収支分岐(100%)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel(param)
    ax2.set_ylabel("回収率(%)")
    ax2.set_title(f"回収率 vs {param}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(outdir, f"sweep_{param}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


if __name__ == "__main__":
    main()