"""
競馬複勝予測 前処理モジュール(完全版)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from pandas.api.types import is_numeric_dtype


# ============================================================
# 1. 特徴量エンジニアリング
# ============================================================

def _grouped_rolling_mean(df, group_col, value_col, window, min_periods=1):
    """
    df.groupby(group_col)[value_col] を shift(1) してから rolling(window).mean() を
    高速に計算するヘルパー関数。
    lambdaをtransformに渡す方式(グループ数ぶんPython関数を呼ぶため遅い)ではなく、
    GroupBy.rolling を直接使うことで大幅に高速化しています。
    """
    shifted = df.groupby(group_col)[value_col].shift(1)
    result = (
        shifted.groupby(df[group_col])
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return result.reindex(df.index)


def _grouped_expanding_mean(df, group_col, value_col, min_periods=1):
    """
    groupby + shift(1) + expanding().mean() を高速に計算するヘルパー関数。
    group_col は単一列名、または複数列のリストを受け付ける。
    """
    shifted = df.groupby(group_col)[value_col].shift(1)
    if isinstance(group_col, list):
        grouper = [df[c] for c in group_col]
        n_levels = len(group_col)
    else:
        grouper = df[group_col]
        n_levels = 1
    result = (
        shifted.groupby(grouper)
        .expanding(min_periods=min_periods)
        .mean()
        .reset_index(level=list(range(n_levels)), drop=True)
    )
    return result.reindex(df.index)


def add_horse_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["馬名", "レース日付"]).copy()
    df["複勝_tmp"] = (df["着順"] <= 3).astype("int8")
    g = df.groupby("馬名", sort=False)

    for n in [3, 5, 9]:
        df[f"past{n}_hukusho_rate"] = _grouped_rolling_mean(
            df, "馬名", "複勝_tmp", n, min_periods=1
        )

    df["past5_agari_mean"] = _grouped_rolling_mean(df, "馬名", "上り", 5, min_periods=1)
    df["past5_chakujun_mean"] = _grouped_rolling_mean(df, "馬名", "着順", 5, min_periods=1)
    df["career_count"] = g.cumcount()

    df["prev_date"] = g["レース日付"].shift(1)
    df["days_since_last_race"] = (df["レース日付"] - df["prev_date"]).dt.days

    df["prev_chakujun"] = g["着順"].shift(1)
    df["prev_agari"] = g["上り"].shift(1)

    df = df.drop(columns=["複勝_tmp", "prev_date"])
    return df


def add_condition_affinity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["複勝_tmp"] = (df["着順"] <= 3).astype("int8")

    df["距離区分"] = pd.cut(
        df["距離(m)"],
        bins=[0, 1400, 1800, 2200, 9999],
        labels=["sprint", "mile", "middle", "long"],
    )

    df["past_dist_hukusho_rate"] = _grouped_expanding_mean(
        df, ["馬名", "距離区分"], "複勝_tmp", min_periods=1
    )
    df["past_track_hukusho_rate"] = _grouped_expanding_mean(
        df, ["馬名", "芝・ダート区分"], "複勝_tmp", min_periods=1
    )

    df = df.drop(columns=["複勝_tmp"])
    return df


def add_jockey_trainer_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("レース日付").copy()
    df["複勝_tmp"] = (df["着順"] <= 3).astype("int8")

    df["jockey_hukusho_rate"] = _grouped_expanding_mean(
        df, "騎手", "複勝_tmp", min_periods=10
    )
    df["trainer_hukusho_rate"] = _grouped_expanding_mean(
        df, "調教師", "複勝_tmp", min_periods=10
    )
    df["jockey_course_rate"] = _grouped_expanding_mean(
        df, ["騎手", "競馬場名"], "複勝_tmp", min_periods=5
    )

    df = df.drop(columns=["複勝_tmp"])
    df = df.sort_values(["馬名", "レース日付"])
    return df


def add_race_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース内での相対的な強さ(Zスコア)を追加する関数です。
    groupby(...).transform(lambda ...) はグループ数ぶんPython関数を呼ぶため遅いので、
    組み込みの transform('mean') / transform('std')(内部でベクトル化されている)を使い、
    Zスコアの計算式だけ自前で行うことで高速化しています。
    """
    df = df.copy()
    rel_cols = [
        "past5_hukusho_rate",
        "past5_agari_mean",
        "jockey_hukusho_rate",
        "past5_chakujun_mean",
        "馬体重",
    ]
    g = df.groupby("レースID")
    for col in rel_cols:
        if col in df.columns:
            race_mean = g[col].transform("mean")
            race_std = g[col].transform("std")
            df[f"{col}_race_z"] = (df[col] - race_mean) / (race_std + 1e-6)
    return df


# 特徴量エンジニアリングの計算に必要な「生データの列」一覧。
# 66列すべてを持ったまま処理するとメモリを圧迫するため、
# build_features に渡す前にこの列 + 最終的に使いたい features 列だけに絞り込みます。
RAW_COLUMNS_FOR_FEATURE_ENGINEERING = [
    "レースID", "レース日付", "着順", "馬名",
    "距離(m)", "芝・ダート区分", "騎手", "調教師", "競馬場名",
    "上り", "馬体重",
]


def read_race_csv(path: str, features: list, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """
    result.csv を読み込む際に、特徴量エンジニアリングに必要な列 +
    最終的に使いたい features 列だけに最初から絞り込んで読み込む関数です。

    66列すべてを読み込むと(文字列列が多いため)メモリを大きく消費するので、
    pd.read_csv の usecols を使い、読み込み時点で必要な列だけに絞ります。

    引数:
        path: result.csv のパス
        features: 使用する特徴量のリスト

    戻り値:
        必要な列だけに絞ったデータフレーム
    """
    # ヘッダー行だけ読んで実在する列名を確認(BOM対策のためencodingを揃える)
    header = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()

    needed = set(RAW_COLUMNS_FOR_FEATURE_ENGINEERING) | set(features)
    usecols = [c for c in header if c in needed]

    df = pd.read_csv(path, encoding=encoding, usecols=usecols, low_memory=False)
    return df


def slim_raw_columns(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    特徴量エンジニアリングに必要な列 + 最終的に使いたい features 列
    (生データに実在するもののみ)だけを残してメモリを削減する関数です。
    """
    needed = set(RAW_COLUMNS_FOR_FEATURE_ENGINEERING) | (set(features) & set(df.columns))
    cols = [c for c in df.columns if c in needed]
    return df[cols].copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["馬名", "レース日付"])
    df = add_horse_history_features(df)
    df = add_condition_affinity(df)
    df = add_jockey_trainer_stats(df)
    df = add_race_relative_features(df)
    return df


# ============================================================
# 2. 列選択・目的変数作成
# ============================================================

def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df2 = df[features].copy()
    return df2


def create_target(df2: pd.DataFrame) -> pd.DataFrame:
    df2 = df2.dropna(subset=["着順"]).copy()
    df2["複勝"] = (df2["着順"] <= 3).astype(int)
    df2 = df2.drop(columns=["着順"])
    return df2


# ============================================================
# 3. 欠損補完・カテゴリエンコーディング
# ============================================================

class Preprocessor:
    def __init__(self, exclude_cols=("複勝", "レースID", "レース日付")):
        self.exclude_cols = list(exclude_cols)
        self.medians_ = {}
        self.modes_ = {}
        self.onehot_cols_ = []
        self.ordinal_cols_ = []
        self.onehot_encoder_ = None
        self.ordinal_encoder_ = None

    def fit(self, df2: pd.DataFrame):
        for col in df2.columns:
            if col in self.exclude_cols:
                continue
            if is_numeric_dtype(df2[col]):
                self.medians_[col] = df2[col].median()
            else:
                self.modes_[col] = df2[col].mode()[0]

        for col in df2.columns:
            if col in self.exclude_cols:
                continue
            if not is_numeric_dtype(df2[col]):
                n_unique = df2[col].nunique()
                if n_unique <= 20:
                    self.onehot_cols_.append(col)
                else:
                    self.ordinal_cols_.append(col)

        df_filled = self._fill_missing(df2)

        if self.onehot_cols_:
            self.onehot_encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=int
            )
            self.onehot_encoder_.fit(df_filled[self.onehot_cols_].astype(str))

        if self.ordinal_cols_:
            self.ordinal_encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self.ordinal_encoder_.fit(df_filled[self.ordinal_cols_].astype(str))

        return self

    def _fill_missing(self, df2: pd.DataFrame) -> pd.DataFrame:
        df2 = df2.copy()
        for col, val in self.medians_.items():
            if col in df2.columns:
                df2[col] = df2[col].fillna(val)
        for col, val in self.modes_.items():
            if col in df2.columns:
                df2[col] = df2[col].fillna(val)
        return df2

    def transform(self, df2: pd.DataFrame) -> pd.DataFrame:
        df2 = self._fill_missing(df2)

        keep_cols = [
            c for c in df2.columns
            if c in self.exclude_cols
            or c not in self.onehot_cols_ + self.ordinal_cols_
        ]
        result = df2[keep_cols].copy()

        if self.onehot_encoder_ is not None:
            onehot_arr = self.onehot_encoder_.transform(
                df2[self.onehot_cols_].astype(str)
            )
            onehot_df = pd.DataFrame(
                onehot_arr,
                columns=self.onehot_encoder_.get_feature_names_out(self.onehot_cols_),
                index=df2.index,
            )
            result = pd.concat([result, onehot_df], axis=1)

        if self.ordinal_encoder_ is not None:
            ordinal_arr = self.ordinal_encoder_.transform(
                df2[self.ordinal_cols_].astype(str)
            )
            ordinal_df = pd.DataFrame(
                ordinal_arr, columns=self.ordinal_cols_, index=df2.index
            )
            result = pd.concat([result, ordinal_df], axis=1)

        return result

    def fit_transform(self, df2: pd.DataFrame) -> pd.DataFrame:
        self.fit(df2)
        return self.transform(df2)


# ============================================================
# 4. 全体をまとめる関数
# ============================================================

def preprocess_train_valid(df: pd.DataFrame, features: list, split_date: str):
    df = df.copy()
    df["レース日付"] = pd.to_datetime(df["レース日付"])

    df = slim_raw_columns(df, features)
    df = build_features(df)

    df2 = select_features(df, features)
    df2 = create_target(df2)

    train = df2[df2["レース日付"] < split_date].copy()
    valid = df2[df2["レース日付"] >= split_date].copy()

    preprocessor = Preprocessor()
    train_processed = preprocessor.fit_transform(train)
    valid_processed = preprocessor.transform(valid)

    return train_processed, valid_processed, preprocessor