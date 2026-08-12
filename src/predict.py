# PREDICT SCRIPT
# Carga artefactos entrenados y genera una predicción individual de precio.

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import yaml


# ============================================================
# CONFIG
# ============================================================

PARAMS_PATH = "params.yaml"

with open(PARAMS_PATH, "r", encoding="utf-8") as conf_file:
    config = yaml.safe_load(conf_file)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

model_path = config["artifacts"]["model_path"]
scaler_x_path = config["artifacts"]["scaler_x_path"]
scaler_y_path = config["artifacts"]["scaler_y_path"]
feature_names_path = config["artifacts"]["feature_names_path"]

required_files = [
    model_path,
    scaler_x_path,
    scaler_y_path,
    feature_names_path,
]

for file_path in required_files:
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"No existe {file_path}. Ejecuta primero: dvc repro"
        )

model = joblib.load(model_path)
scaler_X = joblib.load(scaler_x_path)
scaler_Y = joblib.load(scaler_y_path)

with open(feature_names_path, "r", encoding="utf-8") as f:
    feature_names = json.load(f)


# ============================================================
# INPUT BUILDER
# ============================================================

def build_input(
    town: str,
    classification: str,
    sqm: float,
    terrace: float,
    bhk: int,
    park_u: int,
    levels: int,
    months_in_sale: float,
    total_units: float,
    master_plan_units: float,
    inventory: float,
    months_to_delivery: float,
) -> pd.DataFrame:
    """
    Construye un dataframe con exactamente las columnas esperadas por el modelo.
    """

    # Inicializamos todas las variables en cero
    row = {feature: 0.0 for feature in feature_names}

    # Clasificación SOFTEC
    classification_mapping = config["prepare"]["classification_mapping"]

    if isinstance(classification, str):
        classification_value = classification_mapping[classification]
    else:
        classification_value = classification

    # Variables numéricas base
    input_values = {
        "classification": classification_value,
        "sqm": sqm,
        "terrace": terrace,
        "bhk": bhk,
        "park_u": park_u,
        "levels": levels,
        "months_in_sale": months_in_sale,
        "total_units": total_units,
        "master_plan_units": master_plan_units,
        "inventory": inventory,
        "months_to_delivery": months_to_delivery,
    }

    for key, value in input_values.items():
        if key in row:
            row[key] = float(value)

    # Variable dummy de municipio
    if town in row:
        row[town] = 1.0
    else:
        print(f"Advertencia: el municipio '{town}' no existe en feature_names.")
        print("Municipios/columnas disponibles:")
        print(
            [
                col
                for col in feature_names
                if col not in input_values.keys()
            ]
        )

    input_df = pd.DataFrame(
        [row],
        columns=feature_names,
    )

    return input_df


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict_price(input_df: pd.DataFrame) -> float:
    """
    Escala variables, predice precio estandarizado
    y regresa precio en pesos.
    """

    x_scaled = scaler_X.transform(input_df.values)

    y_scaled_pred = model.predict(x_scaled)

    price_pred = scaler_Y.inverse_transform(
        np.array(y_scaled_pred).reshape(-1, 1)
    ).ravel()[0]

    return float(price_pred)


# ============================================================
# OPTIONS FOR VISUAL APP
# ============================================================

def get_classification_options() -> list[str]:
    """
    Regresa las clasificaciones disponibles
    según params.yaml.
    """

    return list(
        config["prepare"]["classification_mapping"].keys()
    )


def get_available_towns() -> list[str]:
    """
    Regresa los municipios disponibles a partir
    de las columnas utilizadas por el modelo.
    """

    non_town_features = {
        "classification",
        "sqm",
        "terrace",
        "bhk",
        "park_u",
        "levels",
        "months_in_sale",
        "total_units",
        "master_plan_units",
        "inventory",
        "months_to_delivery",
    }

    towns = [
        feature
        for feature in feature_names
        if feature not in non_town_features
    ]

    return sorted(towns)


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    example_input = build_input(
        town="Zapopan",
        classification="R",
        sqm=85,
        terrace=0,
        bhk=2,
        park_u=1,
        levels=8,
        months_in_sale=12,
        total_units=100,
        master_plan_units=100,
        inventory=30,
        months_to_delivery=12,
    )

    prediction = predict_price(example_input)

    print("Input utilizado:")
    print(example_input)

    print("\nPrecio estimado:")
    print(f"${prediction:,.2f} MXN")