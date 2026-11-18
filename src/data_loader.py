



import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"

NA_MEANS_NONE = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
]
NUMERIC_WITH_NA = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]


def load_raw_data():
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    return df


def clean_na_values(df):
    df = df.copy()
    has_basement = df["TotalBsmtSF"] > 0
    for col in ["BsmtExposure", "BsmtFinType2"]:
        df.loc[has_basement & (df[col] == "NA"), col] = np.nan
    df[NA_MEANS_NONE] = df[NA_MEANS_NONE].replace("NA", "None")
    df = df.replace("NA", np.nan)
    for col in NUMERIC_WITH_NA:
        df[col] = pd.to_numeric(df[col])
    return df


if __name__ == "__main__":
    print("pandas:", pd.__version__)
    df = clean_na_values(load_raw_data())
    missing = df.isnull().sum()
    print(missing[missing > 0])




def data_quality_report(df):
    missing = df.isnull().sum()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing": missing[missing > 0],
    }







