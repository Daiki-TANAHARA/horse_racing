#!/bin/bash
# 実験1:ベースライン(馬齢・馬体重・距離・芝ダート区分のみ)を3モデルで実行する

set -e  # いずれかのコマンドが失敗したら即座に停止する

EXPERIMENT="exp1"

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