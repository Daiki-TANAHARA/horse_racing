#!/bin/bash
# XGBoostパラメータ調整を1項目ずつ実行するスクリプト。
# 置き場所: horse_racing/run_scripts/run_sweep.sh
# (実行対象の sweep_and_plot.py は horse_racing/new_evaluation/ にあります)
# 各ステップを終えるごとに、最善値を確認して次のステップの --fixed に反映してください。
# (グラフとCSVは horse_racing/results/sweep/ に保存されます)
set -e

# どこから実行しても、このファイル自身がある場所(run_scripts/)を基準に
# sweep_and_plot.pyがあるディレクトリ(new_evaluation/)に移動してから実行する
cd "$(dirname "$0")/../new_evaluation"

# ── ステップ1: 木の本数 100 -> 1000 ──
# XGBoost.py のデフォルトを既に1000に変更済みなので、このステップは実行不要です。

# ── ステップ2: 木の深さ(max_depth) 4〜10 ──
#uv run sweep_and_plot.py \
#    --param max_depth \
#    --values 4,5,6,7,8,9,10

# ここでグラフ(results/sweep/sweep_max_depth.png)を見て、最善のmax_depthを決めてください。
# 決まったら、下のステップ3以降の --fixed '{"max_depth": ...}' を書き換えてから実行します。

# ── ステップ3: learning_rate (0.1以下) ──
#
#     --param learning_rate \
#     --values 0.01,0.03,0.05,0.07,0.1 \
#     --fixed '{"max_depth": 7}'

# ── ステップ4: scale_pos_weight ──
# "auto" は元の設定(neg/posをfoldごとに自動計算)です。
#uv run sweep_and_plot.py \
#      --param scale_pos_weight \
#      --values auto,1.5,1.2,1.0 \
#      --fixed '{"max_depth": 7, "learning_rate": 0.07}'

# ── ステップ5: L2正則化 (reg_lambda) ──
# candidate値は例です。必要に応じて増減してください。
#uv run sweep_and_plot.py \
#      --param reg_lambda \
#      --values 0,0.5,1.0,5 \
#      --fixed '{"max_depth": 7, "learning_rate": 0.07, "scale_pos_weight": "auto"}'

# ── ステップ6: L1正則化 (reg_alpha) ──
uv run sweep_and_plot.py \
     --param reg_alpha \
     --values 0,0.5,1.0,5 \
     --fixed '{"max_depth": 7, "learning_rate": 0.07, "scale_pos_weight": "auto", "reg_lambda": 1}'