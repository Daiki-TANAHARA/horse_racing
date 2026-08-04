#!/bin/bash
# run_scripts/build_features.sh - 生データから features.csv を作成する

set -e

echo "===== 特徴量作成:make_features.py を実行 ====="
uv run python Preprocess/make_features.py

echo "===== data/features.csv を作成しました ====="