

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, RANDOM_STATE
from src.preprocess import build_preprocessor


def get_models():
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10),
        "Lasso": Lasso(alpha=0.001, max_iter=20000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def compare_models(X_train, y_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = -cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        rows.append({
            "Model": name,
            "CV RMSE (mean)": round(scores.mean(), 4),
            "CV RMSE (std)": round(scores.std(), 4),
        })
    return pd.DataFrame(rows).sort_values("CV RMSE (mean)").reset_index(drop=True)




def get_search_spaces():
    return {
        "Lasso": (
            Lasso(max_iter=20000),
            {"model__alpha": [0.0003, 0.0005, 0.001, 0.002, 0.005]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(subsample=0.8, random_state=RANDOM_STATE),
            {
                "model__n_estimators": [300, 500],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def tune_models(X_train, y_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, best_models = [], {}
    for name, (model, grid) in get_search_spaces().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        search = GridSearchCV(
            pipe, grid, cv=cv, n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        search.fit(X_train, y_train)
        train_rmse = root_mean_squared_error(y_train, search.predict(X_train))
        rows.append({
            "Model": name,
            "Best params": search.best_params_,
            "Train RMSE": round(train_rmse, 4),
            "CV RMSE": round(-search.best_score_, 4),
        })
        best_models[name] = search.best_estimator_
    return pd.DataFrame(rows), best_models


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)



