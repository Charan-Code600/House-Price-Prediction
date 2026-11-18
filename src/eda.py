

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_raw_data, clean_na_values

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def plot_saleprice_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["SalePrice"], bins=40)
    axes[0].set_title("SalePrice (original)")
    axes[0].set_xlabel("Price ($)")

    axes[1].hist(np.log1p(df["SalePrice"]), bins=40)
    axes[1].set_title("SalePrice after log1p")
    axes[1].set_xlabel("log(1 + Price)")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "saleprice_distribution.png", dpi=120)
    plt.close(fig)





def plot_area_vs_price(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["GrLivArea"], df["SalePrice"], s=12, alpha=0.6)

    big = df[df["GrLivArea"] > 4000]
    ax.scatter(big["GrLivArea"], big["SalePrice"], s=40, color="red")
    for _, row in big.iterrows():
        ax.annotate(f"Id {row['Id']}", (row["GrLivArea"], row["SalePrice"]),
                    textcoords="offset points", xytext=(6, 6))

    ax.set_title("Living area vs SalePrice")
    ax.set_xlabel("GrLivArea (sq ft)")
    ax.set_ylabel("SalePrice ($)")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "area_vs_price.png", dpi=120)
    plt.close(fig)





def plot_top_correlations(df, top_n=10):
    numeric = df.select_dtypes(include="number").drop(columns=["Id"])
    corr = numeric.corr()["SalePrice"].drop("SalePrice")
    top = corr.abs().sort_values(ascending=False).head(top_n)
    top = corr[top.index].sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values)
    ax.set_title(f"Top {top_n} features correlated with SalePrice")
    ax.set_xlabel("Correlation")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "top_correlations.png", dpi=120)
    plt.close(fig)





def plot_neighborhood_prices(df):
    medians = df.groupby("Neighborhood")["SalePrice"].median().sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(medians.index, medians.values)
    ax.set_title("Median SalePrice by Neighborhood")
    ax.set_xlabel("Median price ($)")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "neighborhood_prices.png", dpi=120)
    plt.close(fig)



