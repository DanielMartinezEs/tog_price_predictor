from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path("data/processed/prepared_data.csv")
OUTPUT_PATH = Path("reports/figures/figura_15_correlacion_spearman.png")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(INPUT_PATH)


# ============================================================
# VARIABLES TO DISPLAY
# ============================================================

columns = [
    "classification",
    "sqm",
    "terrace",
    "bhk",
    "park_u",
    "levels",
    "price",
    "months_in_sale",
    "master_plan_units",
    "total_units",
    "inventory",
    "months_to_delivery",
]

labels = {
    "classification": "Clasificación",
    "sqm": "Superficie interior",
    "terrace": "Terraza",
    "bhk": "Recámaras",
    "park_u": "Estacionamientos",
    "levels": "Niveles",
    "price": "Precio actual",
    "months_in_sale": "Meses en venta",
    "master_plan_units": "Unidades proyectadas",
    "total_units": "Unidades totales",
    "inventory": "Inventario",
    "months_to_delivery": "Meses para entrega",
}


# ============================================================
# CORRELATION
# ============================================================

correlation = (
    data[columns]
    .corr(method="spearman")
    .rename(index=labels, columns=labels)
)


# ============================================================
# HEATMAP
# ============================================================

mask = np.triu(
    np.ones_like(correlation, dtype=bool),
    k=1,
)

plt.figure(figsize=(13, 10))

ax = sns.heatmap(
    correlation,
    mask=mask,
    annot=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"label": "Rho de Spearman"},
)

ax.set_title(
    "Matriz de correlación de Spearman",
    fontsize=15,
    pad=16,
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\nCorrelación con Precio actual:\n")
print(
    correlation["Precio actual"]
    .sort_values(ascending=False)
    .round(3)
    .to_string()
)

print(f"\nFigura guardada en: {OUTPUT_PATH}")