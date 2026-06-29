"""前処理に必要な関数をまとめたファイルです。"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pandas.api.types import is_numeric_dtype


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
    カテゴリ数が20種類以下ならOne-Hotエンコーディングを行い、それ以上ならLabelEncoderを行います。

    引数:
        データフレーム

    戻り値:
        カテゴリ変数を数値化したデータフレーム
    """

    label_cols = []
    onehot_cols = []

    for col in df2.columns:

        if not is_numeric_dtype(df2[col]): #df2[col]が数値型じゃない場合
            
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

    #Label Encoding
    for col in label_cols:
        le = OrdinalEncoder()
        df2[col] = le.fit_transform(df2[col].astype(str))

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

    df2 = select_features(df, features)
    df2 = create_target(df2)
    df2 = handle_missing(df2)
    df2 = encode_categorical(df2)

    return df2