#!/bin/bash

set -e


echo "=================================="
echo "2. Logistic Regression"
echo "=================================="
uv run learn_model/Logistic.py

echo "=================================="
echo "3. Random Forest"
echo "=================================="
uv run learn_model/RandomForest.py

echo "=================================="
echo "4. XGBoost"
echo "=================================="
uv run learn_model/XGBoost.py

echo "=================================="
echo "5. LightGBM 前処理"
echo "=================================="
uv run extra_LightGBM/claude_preprocess.py

echo "=================================="
echo "6. LightGBM"
echo "=================================="
uv run extra_LightGBM/LightGBM.py

echo "=================================="
echo "7. 評価"
echo "=================================="
uv run evaluation/evaluate_models.py

echo "=================================="
echo "すべて完了しました！"
echo "=================================="