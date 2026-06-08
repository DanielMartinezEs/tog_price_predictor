# EVALUATE MODEL STAGE
# Lee train/test, carga modelo y scaler_Y, calcula métricas y guarda reports/metrics.json.

from pathlib import Path

import pandas as pd
import yaml
import joblib
import json

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


# ============================================================
# CONFIG
# ============================================================

PARAMS_PATH = "params.yaml"

with open(PARAMS_PATH, "r", encoding="utf-8") as conf_file:
    config = yaml.safe_load(conf_file)


# ============================================================
# LOAD DATASETS
# ============================================================

train_data_path = config["split"]["train_data_path"]
test_data_path = config["split"]["test_data_path"]

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

print(f"Train data loaded from: {train_data_path}")
print(f"Test data loaded from: {test_data_path}")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model_path = config["artifacts"]["model_path"]
scaler_y_path = config["artifacts"]["scaler_y_path"]

modelo_final = joblib.load(model_path)
scaler_Y = joblib.load(scaler_y_path)

print(f"Model loaded from: {model_path}")
print(f"Scaler Y loaded from: {scaler_y_path}")


# ============================================================
# X / Y
# ============================================================

xtrain = train_df.drop(["price"], axis="columns").values
ytrain = train_df["price"].values

xtest = test_df.drop(["price"], axis="columns").values
ytest = test_df["price"].values


# ============================================================
# PREDICTIONS
# ============================================================

y_pred_train = modelo_final.predict(xtrain)
y_pred_test = modelo_final.predict(xtest)


# ============================================================
# INVERSE SCALING TO PESOS
# ============================================================

ytrain_real = scaler_Y.inverse_transform(ytrain.reshape(-1, 1)).ravel()
ytest_real = scaler_Y.inverse_transform(ytest.reshape(-1, 1)).ravel()

y_pred_train_real = scaler_Y.inverse_transform(
    y_pred_train.reshape(-1, 1)
).ravel()

y_pred_test_real = scaler_Y.inverse_transform(
    y_pred_test.reshape(-1, 1)
).ravel()


# ============================================================
# METRICS
# ============================================================

metrics = {
    "R2_train": float(r2_score(ytrain_real, y_pred_train_real)),
    "R2_test": float(r2_score(ytest_real, y_pred_test_real)),
    "RMSE_train_pesos": float(root_mean_squared_error(ytrain_real, y_pred_train_real)),
    "RMSE_test_pesos": float(root_mean_squared_error(ytest_real, y_pred_test_real)),
    "MAE_train_pesos": float(mean_absolute_error(ytrain_real, y_pred_train_real)),
    "MAE_test_pesos": float(mean_absolute_error(ytest_real, y_pred_test_real))
}

print("Metrics:")
print(metrics)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = config["metrics"]["metrics_path"]

Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print(f"Metrics saved to: {metrics_path}")