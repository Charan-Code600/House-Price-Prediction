
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    RANDOM_STATE, TEST_SIZE, OUTLIER_IDS,
    ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
)


def split_data(df):
    X = df.drop(columns=["SalePrice"])
    y = np.log1p(df["SalePrice"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    keep = ~X_train["Id"].isin(OUTLIER_IDS)
    X_train, y_train = X_train[keep], y_train[keep]

    X_train = X_train.drop(columns=["Id"])
    X_test = X_test.drop(columns=["Id"])
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            categories=list(ORDINAL_COLUMNS.values()),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLUMNS),
        ("ord", ordinal_pipe, list(ORDINAL_COLUMNS)),
        ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
    ])





