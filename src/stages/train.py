# TRAIN MODEL STAGE
# Lee train_data.csv, entrena el modelo Ridge y guarda el modelo final.

from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import joblib

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge


# ============================================================
# CONFIG
# ============================================================

PARAMS_PATH = "params.yaml"

with open(PARAMS_PATH, "r", encoding="utf-8") as conf_file:
    config = yaml.safe_load(conf_file)

np.random.seed(config["base"]["numpy_seed"])


# ============================================================
# LOAD TRAIN DATA
# ============================================================

train_data_path = config["split"]["train_data_path"]

train_df = pd.read_csv(train_data_path)

print(f"Train data loaded from: {train_data_path}")
print(f"Train data shape: {train_df.shape}")


# ============================================================
# X / Y
# ============================================================

X_train = train_df.drop(["price"], axis="columns").values
y_train = train_df["price"].values


# ============================================================
# TRAIN MODEL
# ============================================================

folds = KFold(
    n_splits=config["train"]["cv_splits"],
    shuffle=True,
    random_state=config["train"]["cv_random_state"]
)

ridge_grid = GridSearchCV(
    estimator=Ridge(),
    param_grid={"alpha": config["train"]["alphas"]},
    scoring=config["train"]["scoring"],
    cv=folds,
    n_jobs=config["train"]["n_jobs"]
)

ridge_grid.fit(X_train, y_train)

modelo_final = ridge_grid.best_estimator_


# ============================================================
# SAVE MODEL
# ============================================================

model_path = config["artifacts"]["model_path"]

Path(model_path).parent.mkdir(parents=True, exist_ok=True)

joblib.dump(modelo_final, model_path)

print("Modelo entrenado correctamente.")
print(f"Mejores hiperparámetros: {ridge_grid.best_params_}")
print(f"R2 CV train: {ridge_grid.best_score_}")
print(f"Modelo guardado en: {model_path}")