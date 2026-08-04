#!/bin/bash
# 参考実験:実験5 + 単勝オッズ・人気で回収率を計算する

set -e

EXPERIMENT="exp5_with_odds"

echo "===== ${EXPERIMENT}: XGBoost ====="
uv run python learn_model/XGBoost.py --experiment "${EXPERIMENT}"

echo "===== ${EXPERIMENT}: Random Forest ====="
uv run python learn_model/RandomForest.py --experiment "${EXPERIMENT}"

echo "===== ${EXPERIMENT}: ロジスティック回帰 ====="
uv run python learn_model/Logistic.py --experiment "${EXPERIMENT}"

echo "=================================="
echo "3モデル結果"
echo "=================================="
uv run evaluation/evaluate_models.py

echo "===== ${EXPERIMENT} 完了 ====="