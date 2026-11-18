



import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict_price(model, house):
    row = pd.DataFrame([house]).reindex(columns=FEATURE_COLUMNS)
    return float(np.expm1(model.predict(row))[0])

