from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURACION
# ============================================================

PARAMS_PATH = Path("params.yaml")

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


TEST_DATA_PATH = Path(
    config["split"]["test_data_path"]
)

MODEL_PATH = Path(
    config["artifacts"]["model_path"]
)

SCALER_Y_PATH = Path(
    config["artifacts"]["scaler_y_path"]
)

FEATURE_NAMES_PATH = Path(
    config["artifacts"]["feature_names_path"]
)

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGA DE ARTEFACTOS
# ============================================================

test_df = pd.read_csv(TEST_DATA_PATH)

model = joblib.load(MODEL_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

with open(
    FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8",
) as f:
    feature_names = json.load(f)


# ============================================================
# DATOS DE PRUEBA
# ============================================================

X_test = test_df[feature_names].astype(float)

y_test_scaled = (
    test_df["price"]
    .astype(float)
    .to_numpy()
    .reshape(-1, 1)
)

y_pred_scaled = (
    model.predict(X_test)
    .reshape(-1, 1)
)


# ============================================================
# REGRESO DE PRECIOS A MXN
# ============================================================

y_test = scaler_y.inverse_transform(
    y_test_scaled
).ravel()

y_pred = scaler_y.inverse_transform(
    y_pred_scaled
).ravel()


# ============================================================
# METRICAS
# ============================================================

r2 = r2_score(
    y_test,
    y_pred,
)

mae = mean_absolute_error(
    y_test,
    y_pred,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred,
    )
)

residuals = y_test - y_pred


print("\nMetricas reproducidas:\n")
print(f"R2   = {r2:.5f}")
print(f"MAE  = ${mae:,.2f}")
print(f"RMSE = ${rmse:,.2f}")


# ============================================================
# FORMATO EN MILLONES DE MXN
# ============================================================

millions_formatter = FuncFormatter(
    lambda value, position: f"{value / 1_000_000:.1f}"
)


# ============================================================
# FIGURA 1: PRECIO REAL VS PRECIO ESTIMADO
# ============================================================

minimum = min(
    y_test.min(),
    y_pred.min(),
)

maximum = max(
    y_test.max(),
    y_pred.max(),
)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

ax.scatter(
    y_test,
    y_pred,
    alpha=0.70,
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
)

ax.xaxis.set_major_formatter(
    millions_formatter
)

ax.yaxis.set_major_formatter(
    millions_formatter
)

ax.set_xlabel(
    "Precio real (millones de MXN)"
)

ax.set_ylabel(
    "Precio estimado (millones de MXN)"
)

ax.set_title(
    "Precio real vs. precio estimado - Ridge\n"
    f"R² en prueba = {r2:.4f}"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

actual_vs_predicted_path = (
    OUTPUT_DIR
    / "ridge_real_vs_estimado.png"
)

fig.savefig(
    actual_vs_predicted_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURA 2: RESIDUOS
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)

ax.scatter(
    y_pred,
    residuals,
    alpha=0.70,
)

ax.axhline(
    y=0,
    linestyle="--",
)

ax.xaxis.set_major_formatter(
    millions_formatter
)

ax.yaxis.set_major_formatter(
    millions_formatter
)

ax.set_xlabel(
    "Precio estimado (millones de MXN)"
)

ax.set_ylabel(
    "Residual: precio real - estimado "
    "(millones de MXN)"
)

ax.set_title(
    "Residuos del modelo Ridge "
    "sobre el conjunto de prueba"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

residuals_path = (
    OUTPUT_DIR
    / "ridge_residuos_test.png"
)

fig.savefig(
    residuals_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# DATOS PARA AUDITORIA
# ============================================================

diagnostics_df = pd.DataFrame(
    {
        "precio_real": y_test,
        "precio_estimado": y_pred,
        "residual": residuals,
    }
)

diagnostics_path = Path(
    "reports/ridge_test_predictions.csv"
)

diagnostics_df.to_csv(
    diagnostics_path,
    index=False,
)


# ============================================================
# RESULTADOS
# ============================================================

print(
    f"\nFigura 1: {actual_vs_predicted_path}"
)

print(
    f"Figura 2: {residuals_path}"
)

print(
    f"Datos: {diagnostics_path}"
)