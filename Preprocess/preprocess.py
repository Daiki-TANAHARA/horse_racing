"""前処理に必要な関数をまとめたファイルです。"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pandas.api.types import is_numeric_dtype

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

    df = df.sort_values(["馬名", "レース日付", "発走時刻"])
    df = create_last5_place_rate(df)
    df = create_last3_place_rate(df)
    df = create_previous_rank(df)
    df = create_last5_average_rank(df)
    df = create_last3_average_rank(df)
    df = create_previous_last3f(df)
    df = create_last3_average_last3f(df)
    df = create_last5_average_last3f(df)

    df = df.sort_values(["騎手", "レース日付", "発走時刻"])
    df = create_jockey_place_rate(df)

    df = df.sort_values(["調教師", "レース日付", "発走時刻"])
    df = create_trainer_place_rate(df)

    df2 = select_features(df, features)
    df2 = create_target(df2)
    df2 = handle_missing(df2)
    df2 = encode_categorical(df2)

    return df2