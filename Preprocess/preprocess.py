"""前処理に必要な関数をまとめたファイルです。"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pandas.api.types import is_numeric_dtype
import numpy as np

def create_last5_place_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去5走の複勝率を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「過去5走複勝率」列を追加したデータフレーム
    """

    # 複勝フラグ作成（3着以内なら1、それ以外は0）
    df["複勝フラグ"] = (df["着順"] <= 3).astype(int)

    # 過去5走複勝率
    df["過去5走複勝率"] = (
        df.groupby("馬名")["複勝フラグ"]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # 作業用列を削除
    df = df.drop(columns=["複勝フラグ"])

    return df

def create_last3_place_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去3走の複勝率を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「過去3走複勝率」列を追加したデータフレーム
    """

    # 複勝フラグ作成（3着以内なら1、それ以外は0）
    df["複勝フラグ"] = (df["着順"] <= 3).astype(int)

    # 過去3走複勝率
    df["過去3走複勝率"] = (
        df.groupby("馬名")["複勝フラグ"]
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # 作業用列を削除
    df = df.drop(columns=["複勝フラグ"])

    return df

def create_previous_rank(df:pd.DataFrame) -> pd.DataFrame:
    """
    前走着順を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「前走着順」列を追加したデータフレーム
    """

    df["前走着順"] = (
        df.groupby("馬名")["着順"]
          .shift(1)
    )

    return df

def create_last5_average_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去5走の平均着順を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「過去5走平均着順」列を追加したデータフレーム
    """

    df["過去5走平均着順"] = (
        df.groupby("馬名")["着順"]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    return df

def create_last3_average_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去3走の平均着順を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「過去3走平均着順」列を追加したデータフレーム
    """

    df["過去3走平均着順"] = (
        df.groupby("馬名")["着順"]
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    return df

def create_previous_last3f(df: pd.DataFrame) -> pd.DataFrame:
    """
    前走上りを作成する関数です。

    引数:
        データフレーム

    戻り値:
        「前走上り」列を追加したデータフレーム
    """

    df["前走上り"] = (
        df.groupby("馬名")["上り"]
          .shift(1)
    )

    return df

def create_last3_average_last3f(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去3走の平均上りを作成する関数です。

    引数:
        データフレーム

    戻り値:
        「過去3走平均上り」列を追加したデータフレーム
    """

    df["過去3走平均上り"] = (
        df.groupby("馬名")["上り"]
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    return df

def create_last5_average_last3f(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去5走の平均上りを作成する関数です。

    引数:
        データフレーム

    戻り値:
        「過去5走平均上り」列を追加したデータフレーム
    """

    df["過去5走平均上り"] = (
        df.groupby("馬名")["上り"]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    return df

def create_jockey_place_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    騎手の通算複勝率を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「騎手複勝率」列を追加したデータフレーム
    """

    # 複勝フラグ作成（3着以内なら1、それ以外は0）
    df["複勝フラグ"] = (df["着順"] <= 3).astype(int)

    # 騎手ごとの通算複勝率（現在のレースは含めない）
    df["騎手複勝率"] = (
        df.groupby("騎手")["複勝フラグ"]
          .transform(lambda x: x.shift(1).expanding().mean())
    )

    # 作業用列を削除
    df = df.drop(columns=["複勝フラグ"])

    return df

def create_trainer_place_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    調教師の通算複勝率を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「調教師複勝率」列を追加したデータフレーム
    """

    # 複勝フラグ作成（3着以内なら1、それ以外は0）
    df["複勝フラグ"] = (df["着順"] <= 3).astype(int)

    # 調教師ごとの通算複勝率（現在のレースは含めない）
    df["調教師複勝率"] = (
        df.groupby("調教師")["複勝フラグ"]
          .transform(lambda x: x.shift(1).expanding().mean())
    )

    # 作業用列を削除
    df = df.drop(columns=["複勝フラグ"])

    return df

def create_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    前走からの休養日数を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「休養日数」列を追加したデータフレーム
    """

    previous_date = (
        df.groupby("馬名")["レース日付"]
          .shift(1)
    )

    df["休養日数"] = (
        df["レース日付"] - previous_date
    ).dt.days

    return df

def create_surface_place_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    芝・ダート別通算複勝率を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「芝・ダート別複勝率」列を追加したデータフレーム
    """

    place = (df["着順"] <= 3).astype(int)

    df["芝・ダート別複勝率"] = (
        place.groupby([df["馬名"], df["芝・ダート区分"]])
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    return df

def create_course_place_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    競馬場別通算複勝率を作成する関数です。

    引数:
        データフレーム

    戻り値:
        「競馬場別複勝率」列を追加したデータフレーム
    """

    place = (df["着順"] <= 3).astype(int)

    df["競馬場別複勝率"] = (
        place.groupby([df["馬名"], df["競馬場名"]])
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    return df

def create_distance_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース距離から距離帯を作成する関数です。

    距離帯の区分
        短距離   : 1400m以下
        マイル   : 1401～1800m
        中距離   : 1801～2200m
        中長距離 : 2201～2800m
        長距離   : 2801m～
    """

    distance_conditions = [
        df["距離(m)"] <= 1400,
        df["距離(m)"] <= 1800,
        df["距離(m)"] <= 2200,
        df["距離(m)"] <= 2800,
    ]

    distance_choices = [
        "短距離",
        "マイル",
        "中距離",
        "中長距離",
    ]

    df["距離帯"] = np.select(
        distance_conditions,
        distance_choices,
        default="長距離"
    )

    return df

def create_distance_place_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごとの距離帯別通算複勝率を作成する関数です。

    例:
        短距離のレースでは短距離での過去複勝率
        マイルのレースではマイルでの過去複勝率
        を特徴量として使用します。
    """

    place = (df["着順"] <= 3).astype(int)

    df["距離帯別複勝率"] = (
        place
        .groupby([df["馬名"], df["距離帯"]])
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    return df

def create_race_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    競争条件からレースクラスを作成する関数です。

    区分:
        新馬
        未勝利
        1勝クラス
        2勝クラス
        3勝クラス
        オープン
    """

    def convert_class(x):

        if "新馬" in x or "未出走" in x:
            return "新馬"

        elif "未勝利" in x:
            return "未勝利"

        elif any(word in x for word in [
            "1400万下",
            "1500万下",
            "1600万下",
            "3勝クラス"
        ]):
            return "3勝クラス"

        elif any(word in x for word in [
            "600万下",
            "700万下",
            "800万下",
            "900万下",
            "1000万下",
            "2勝クラス"
        ]):
            return "2勝クラス"

        elif any(word in x for word in [
            "300万下",
            "400万下",
            "500万下",
            "1勝クラス"
        ]):
            return "1勝クラス"

        elif "オープン" in x:
            return "オープン"

        else:
            return "その他"

    df["レースクラス"] = df["競争条件"].apply(convert_class)

    return df

def create_race_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    リステッド・重賞競走からレースグレードを作成する関数です。

    区分:
        G1
        G2
        G3
        L
        G
        なし
    """

    df["レースグレード"] = df["リステッド・重賞競走"].fillna("なし")

    return df

def create_rank_features(
    df: pd.DataFrame,
    rank_features: dict[str, bool]
) -> pd.DataFrame:
    """
    同一レース内でパーセンタイル順位を作成する関数です。

    引数:
        df : データフレーム
        rank_features :
            {"特徴量名": ascending} の辞書
            True  -> 小さいほど良い
            False -> 大きいほど良い
    """

    for feature, ascending in rank_features.items():
        df[f"{feature}順位率"] = (
            df.groupby("レースID")[feature]
              .rank(
                  method="min",
                  ascending=ascending,
                  pct=True
              )
        )

    return df

def zscore(x):
    std = x.std()

    if std == 0:
        return pd.Series(0, index=x.index)

    return (x - x.mean()) / std


def create_standardize_features(
    df: pd.DataFrame,
    standardize_features: list[str]
) -> pd.DataFrame:
    """
    同一レース内で標準化した特徴量を作成する関数です。
    標準偏差が0の場合は0を設定します。
    """

    for feature in standardize_features:
        df[f"{feature}標準化"] = (
            df.groupby("レースID")[feature]
              .transform(zscore)
        )

    return df

def select_features(df:pd.DataFrame ,features:list[str])-> pd.DataFrame:
    """
    使用する特徴量だけを抽出する関数です。

    引数:
    データフレーム
    特徴量リスト

    戻り値:
    必要な列だけのデータフレーム
    """
    df2 = df[features].copy()
    return df2

def create_target(df2:pd.DataFrame)-> pd.DataFrame:
    """
    「着順」列から目的変数「複勝」列を作る関数です。

    引数:
    必要な列だけのデータフレーム

    戻り値:
    目的変数を追加したデータフレーム

    """
    # 着順欠損を除去
    df2 = df2.dropna(subset=["着順"])

    # 目的変数(複勝)作成
    df2["複勝"] = (df2["着順"] <= 3).astype(int)

    # 着順削除
    df2 = df2.drop(columns=["着順"])
    return df2



def handle_missing(df2: pd.DataFrame) -> pd.DataFrame:
    """
    欠損がある場合は、数値列は中央値補完、カテゴリ列は最頻値補完を行う関数です。
    引数:
    必要な列だけのデータフレーム
    戻り値:
    補完したデータフレーム

    """

    for col in df2.columns:
        if is_numeric_dtype(df2[col]): #df2[col]が数値列の場合
            df2[col] = df2[col].fillna(df2[col].median())

        else:
            df2[col] = df2[col].fillna(df2[col].mode()[0])

    return df2




def encode_categorical(df2: pd.DataFrame) -> pd.DataFrame:
    """
    カテゴリデータを数値データに変換する関数です。
    カテゴリ数が20種類以下ならOne-Hotエンコーディングを行い、それ以上ならOridinalEncoderを行います。

    引数:
        データフレーム

    戻り値:
        カテゴリ変数を数値化したデータフレーム
    """

    label_cols = []
    onehot_cols = []

    for col in df2.columns:

        if not is_numeric_dtype(df2[col]): #df2[col]が数値型じゃない場合
            
            if col in ["複勝", "レースID", "レース日付"]:
                continue

            #カテゴリ数を数える
            n_unique = df2[col].nunique()
            
            if n_unique <= 20:
                onehot_cols.append(col)
            else:
                label_cols.append(col)

    #One-Hot Encoding
    df2 = pd.get_dummies(
        df2,
        columns = onehot_cols,
        dtype = int
    )

    #Oridinal Encoding
    for col in label_cols:
        le = OrdinalEncoder()
        df2[col] = le.fit_transform(df2[[col]].astype(str))

    return df2



def preprocess(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    上記の前処理をまとめて実行する関数です。
    =============つまりこれをやれば前処理は全部終わり！==============

    引数:
        csvを読み込んだ生データのデータフレーム
        特徴量リスト

    戻り値:
        前処理後のデータフレーム
    """

    # 障害レースを除外
    df = df[df["障害区分"].isna()].copy()
    
    # 列名を分かりやすく変更
    df = df.rename(columns={
        "場体重増減": "馬体重増減"
    })

    df["天候"] = df["天候"].str.strip()

    df = create_distance_category(df)

    df = df.sort_values(["馬名", "レース日付", "発走時刻"])
    df = create_last5_place_rate(df)
    df = create_last3_place_rate(df)
    df = create_previous_rank(df)
    df = create_last5_average_rank(df)
    df = create_last3_average_rank(df)
    df = create_previous_last3f(df)
    df = create_last3_average_last3f(df)
    df = create_last5_average_last3f(df)
    df = create_rest_days(df)
    df = create_surface_place_rate(df)
    df = create_course_place_rate(df)
    df = create_distance_place_rate(df)

    df = create_race_class(df)
    df = create_race_grade(df)

    df = df.sort_values(["騎手", "レース日付", "発走時刻"])
    df = create_jockey_place_rate(df)

    df = df.sort_values(["調教師", "レース日付", "発走時刻"])
    df = create_trainer_place_rate(df)

    rank_features = {
        "過去3走平均着順": True,      # 小さいほど良い(着順なので)
        "過去5走平均着順": True,
        "過去3走平均上り": True,      # 上りタイムも小さいほど良い
        "過去5走平均上り": True,
        "過去3走複勝率": False,       # 大きいほど良い
        "過去5走複勝率": False,
        "騎手複勝率": False,
        "調教師複勝率": False,
        "芝・ダート別複勝率": False,
        "距離帯別複勝率": False,
        "競馬場別複勝率": False,
    }

    df = create_rank_features(df, rank_features)

    zscore_features = [
        "過去3走平均着順",
        "過去5走平均着順",
        "過去3走平均上り",
        "過去5走平均上り",
        "過去3走複勝率",
        "過去5走複勝率",
        "騎手複勝率",
        "調教師複勝率",
        "芝・ダート別複勝率",
        "距離帯別複勝率",
        "競馬場別複勝率",
    ]

    df = create_standardize_features(df, zscore_features)

    df2 = select_features(df, features)
    df2 = create_target(df2)
    df2 = handle_missing(df2)
    df2 = encode_categorical(df2)

    return df2