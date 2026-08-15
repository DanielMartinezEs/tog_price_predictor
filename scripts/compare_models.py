import warnings

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import BaggingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
)


# ============================================================
# CONFIG
# ============================================================

with open("params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

np.random.seed(config["base"]["numpy_seed"])


# ============================================================
# LOAD CURRENT PIPELINE DATA
# ============================================================

train_df = pd.read_csv(
    config["split"]["train_data_path"]
)

test_df = pd.read_csv(
    config["split"]["test_data_path"]
)

scaler_Y = joblib.load(
    config["artifacts"]["scaler_y_path"]
)

X_train = train_df.drop(
    columns=["price"]
).values

y_train = train_df[
    "price"
].values

X_test = test_df.drop(
    columns=["price"]
).values

y_test = test_df[
    "price"
].values


print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 1. LINEAR REGRESSION
# ============================================================

linear = LinearRegression()

linear.fit(
    X_train,
    y_train,
)


# ============================================================
# 2. DECISION TREE
# ============================================================

tree = DecisionTreeRegressor(
    random_state=0,
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)

tree.fit(
    X_train,
    y_train,
)


# ============================================================
# 3. BAGGING TREES
# Historical grid from Phase 1
# ============================================================

bagging_grid = GridSearchCV(
    estimator=BaggingRegressor(
        random_state=42
    ),
    param_grid={
        "n_estimators": [40, 50, 60],
        "max_samples": [0.70, 0.75, 0.80],
    },
    scoring="r2",
    cv=5,
    n_jobs=-1,
)

bagging_grid.fit(
    X_train,
    y_train,
)

bagging = bagging_grid.best_estimator_


# ============================================================
# 4. NEURAL NETWORK - MLP
# Historical Phase 1 configuration
# ============================================================

mlp = MLPRegressor(
    hidden_layer_sizes=(6,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.005,
    max_iter=1,
    warm_start=True,
    random_state=33,
)

for _ in range(200):
    mlp.fit(
        X_train,
        y_train,
    )


# ============================================================
# 5. RIDGE
# Current pipeline configuration
# ============================================================

folds = KFold(
    n_splits=config["train"]["cv_splits"],
    shuffle=True,
    random_state=config["train"]["cv_random_state"],
)

ridge_grid = GridSearchCV(
    estimator=Ridge(),
    param_grid={
        "alpha": config["train"]["alphas"]
    },
    scoring=config["train"]["scoring"],
    cv=folds,
    n_jobs=config["train"]["n_jobs"],
)

ridge_grid.fit(
    X_train,
    y_train,
)

ridge = ridge_grid.best_estimator_


# ============================================================
# 6. LASSO
# Same alpha grid used during Phase 1 comparison
# ============================================================

lasso_grid = GridSearchCV(
    estimator=Lasso(
        max_iter=10000
    ),
    param_grid={
        "alpha": config["train"]["alphas"]
    },
    scoring="r2",
    cv=folds,
    n_jobs=-1,
)

lasso_grid.fit(
    X_train,
    y_train,
)

lasso = lasso_grid.best_estimator_


# ============================================================
# METRICS IN ORIGINAL PESO SCALE
# ============================================================

def evaluate_model(
    name,
    model,
):
    pred_train_scaled = model.predict(
        X_train
    )

    pred_test_scaled = model.predict(
        X_test
    )

    y_train_real = scaler_Y.inverse_transform(
        y_train.reshape(-1, 1)
    ).ravel()

    y_test_real = scaler_Y.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    pred_train_real = scaler_Y.inverse_transform(
        pred_train_scaled.reshape(-1, 1)
    ).ravel()

    pred_test_real = scaler_Y.inverse_transform(
        pred_test_scaled.reshape(-1, 1)
    ).ravel()

    r2_train = r2_score(
        y_train_real,
        pred_train_real,
    )

    r2_test = r2_score(
        y_test_real,
        pred_test_real,
    )

    return {
        "Modelo": name,
        "R2_train": r2_train,
        "R2_test": r2_test,
        "RMSE_train_pesos": root_mean_squared_error(
            y_train_real,
            pred_train_real,
        ),
        "RMSE_test_pesos": root_mean_squared_error(
            y_test_real,
            pred_test_real,
        ),
        "MAE_train_pesos": mean_absolute_error(
            y_train_real,
            pred_train_real,
        ),
        "MAE_test_pesos": mean_absolute_error(
            y_test_real,
            pred_test_real,
        ),
        "Gap_R2_train_test": (
            r2_train - r2_test
        ),
    }


models = [
    ("Linear Regression", linear),
    ("Decision Tree", tree),
    ("Bagging Trees (tuned)", bagging),
    ("Neural Network (MLP)", mlp),
    ("Ridge Regression", ridge),
    ("Lasso Regression", lasso),
]

results = pd.DataFrame(
    [
        evaluate_model(name, model)
        for name, model in models
    ]
)

results = results.sort_values(
    by="R2_test",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# OUTPUT
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)

print("\n============================================")
print("COMPARACION FINAL - DATASET CORREGIDO")
print("============================================\n")

print(
    results.to_string(
        index=False,
        formatters={
            "R2_train": "{:.5f}".format,
            "R2_test": "{:.5f}".format,
            "RMSE_train_pesos": "{:,.2f}".format,
            "RMSE_test_pesos": "{:,.2f}".format,
            "MAE_train_pesos": "{:,.2f}".format,
            "MAE_test_pesos": "{:,.2f}".format,
            "Gap_R2_train_test": "{:.5f}".format,
        },
    )
)

print("\nMejor Bagging:")
print(bagging_grid.best_params_)

print("\nMejor Ridge:")
print(ridge_grid.best_params_)
print(
    "R2 CV Ridge:",
    ridge_grid.best_score_,
)

print("\nMejor Lasso:")
print(lasso_grid.best_params_)
print(
    "R2 CV Lasso:",
    lasso_grid.best_score_,
)

results.to_csv(
    "reports/model_comparison.csv",
    index=False,
)

print(
    "\nResultados guardados en "
    "reports/model_comparison.csv"
)