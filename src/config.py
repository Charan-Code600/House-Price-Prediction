

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTLIER_IDS = [524, 1299]

QUALITY_ORDER = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

ORDINAL_COLUMNS = {
    "ExterQual": QUALITY_ORDER,
    "ExterCond": QUALITY_ORDER,
    "BsmtQual": QUALITY_ORDER,
    "BsmtCond": QUALITY_ORDER,
    "HeatingQC": QUALITY_ORDER,
    "KitchenQual": QUALITY_ORDER,
    "FireplaceQu": QUALITY_ORDER,
    "GarageQual": QUALITY_ORDER,
    "GarageCond": QUALITY_ORDER,
    "PoolQC": QUALITY_ORDER,
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
}

CATEGORICAL_COLUMNS = [
    "MSSubClass", "MoSold",
    "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir",
    "Electrical", "Functional", "GarageType", "PavedDrive", "Fence",
    "MiscFeature", "SaleType", "SaleCondition",
]

NUMERIC_COLUMNS = [
    "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt",
    "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
    "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr",
    "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars",
    "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
    "ScreenPorch", "PoolArea", "MiscVal", "YrSold",
]




FEATURE_COLUMNS = list(ORDINAL_COLUMNS) + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS




NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]





