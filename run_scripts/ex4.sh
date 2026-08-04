#!/bin/bash
# 実験4:過去成績特徴量を追加して3モデルで実行する

set -e

EXPERIMENT="exp4"

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