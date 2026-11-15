"""
讀資料 + 建特徵。train_lgbm.py 和 notebook 都用這裡的函式，確保特徵一致。

特徵分兩組：
1. 原始 metadata：train 和 test 都有的欄位（train 獨有的診斷欄位會洩漏答案，一律排除）
2. 病人內標準化（ugly duckling）：這顆病灶跟同一個人的其他病灶比起來有多特別
"""

import pandas as pd

from config import DROP_COLS, NO_PATIENT_NORM, TEST_META_CSV, TRAIN_META_CSV


def load_data():
    train = pd.read_csv(TRAIN_META_CSV, low_memory=False)
    test = pd.read_csv(TEST_META_CSV) if TEST_META_CSV.exists() else None
    return train, test


def _add_patient_feats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("patient_id")[cols]
    z = (df[cols] - g.transform("mean")) / (g.transform("std") + 1e-6)
    z.columns = [c + "_pnorm" for c in cols]
    df = pd.concat([df, z], axis=1)
    df["patient_n_lesions"] = df.groupby("patient_id")["isic_id"].transform("count")
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame | None = None):
    """回傳 (train, test, features)。test 可以是 None（只做 CV 時）。"""
    ref = test if test is not None else train
    common = [c for c in ref.columns if c in train.columns and c not in DROP_COLS]
    cat_cols = [c for c in common if not pd.api.types.is_numeric_dtype(train[c])]
    num_cols = [c for c in common if c not in cat_cols]

    # 類別欄位：test 用 train 的類別表，避免編碼對不上
    for c in cat_cols:
        cats = pd.Categorical(train[c]).categories
        train[c] = pd.Categorical(train[c], categories=cats)
        if test is not None:
            test[c] = pd.Categorical(test[c], categories=cats)

    pn_cols = [c for c in num_cols if c not in NO_PATIENT_NORM]
    train = _add_patient_feats(train, pn_cols)
    if test is not None:
        test = _add_patient_feats(test, pn_cols)

    features = num_cols + cat_cols + [c + "_pnorm" for c in pn_cols] + ["patient_n_lesions"]
    return train, test, features
