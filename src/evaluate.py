



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.eda import FIGURES_DIR


def evaluate_on_test(model, X_test, y_test):
    pred_log = model.predict(X_test)
    actual = np.expm1(y_test)
    predicted = np.expm1(pred_log)

    return {
        "RMSE (log price)": round(root_mean_squared_error(y_test, pred_log), 4),
        "MAE ($)": round(mean_absolute_error(actual, predicted)),
        "R2 (price in $)": round(r2_score(actual, predicted), 4),
    }


def worst_errors(model, X_test, y_test, n=10):
    predicted = np.expm1(model.predict(X_test))
    actual = np.expm1(y_test)

    table = pd.DataFrame({
        "Neighborhood": X_test["Neighborhood"],
        "GrLivArea": X_test["GrLivArea"],
        "Actual": actual.round(),
        "Predicted": predicted.round(),
    })
    table["Error ($)"] = table["Predicted"] - table["Actual"]
    table["Error (%)"] = (table["Error ($)"] / table["Actual"] * 100).round(1)
    order = table["Error ($)"].abs().sort_values(ascending=False).index
    return table.loc[order].head(n)




def plot_actual_vs_predicted(model, X_test, y_test):
    predicted = np.expm1(model.predict(X_test))
    actual = np.expm1(y_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, s=15, alpha=0.6)
    limit = [actual.min(), actual.max()]
    ax.plot(limit, limit, color="red", linestyle="--", label="Perfect prediction")
    ax.set_title("Actual vs Predicted price (test data)")
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.legend()

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=120)
    plt.close(fig)


def plot_feature_importance(model, top_n=10):
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["model"].coef_, index=names)

    top = coefs[coefs.abs().sort_values(ascending=False).head(top_n).index]
    top = top.sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.index, top.values)
    ax.set_title(f"Top {top_n} features by Lasso coefficient")
    ax.set_xlabel("Coefficient (effect on log price)")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


    