# VALIDATION UTILITIES
# Validaciones de dominio para entradas del TOG Price Predictor.

from functools import lru_cache
from pathlib import Path
import json

import joblib
import pandas as pd
import yaml


# ============================================================
# CONFIG
# ============================================================

PARAMS_PATH = Path("params.yaml")

with open(PARAMS_PATH, "r", encoding="utf-8") as conf_file:
    config = yaml.safe_load(conf_file)


TRAIN_DATA_PATH = Path(
    config["split"]["train_data_path"]
)

SCALER_X_PATH = Path(
    config["artifacts"]["scaler_x_path"]
)

FEATURE_NAMES_PATH = Path(
    config["artifacts"]["feature_names_path"]
)

SQM_MAX = float(
    config["prepare"]["sqm_max"]
)


# ============================================================
# FEATURES USED FOR RANGE VALIDATION
# ============================================================

NUMERIC_FEATURES = [
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
]


FEATURE_LABELS = {
    "sqm": "Superficie interior",
    "terrace": "Superficie de terraza",
    "bhk": "Número de recámaras",
    "park_u": "Cajones de estacionamiento",
    "levels": "Número de niveles",
    "months_in_sale": "Meses en venta",
    "total_units": "Unidades totales",
    "master_plan_units": "Unidades del master plan",
    "inventory": "Inventario disponible",
    "months_to_delivery": "Meses para entrega",
}


# Tolerancia para pequeñas diferencias numéricas producidas
# al aplicar inverse_transform.
EPSILON = 1e-6


# ============================================================
# TRAINING RANGES
# ============================================================

@lru_cache(maxsize=1)
def get_training_ranges() -> dict:
    """
    Obtiene los valores mínimo y máximo en escala ORIGINAL
    observados en el conjunto de entrenamiento.

    train_data.csv contiene las variables ya escaladas, por lo que
    primero se aplica inverse_transform utilizando el scaler_X
    entrenado.

    Los rangos resultantes se utilizan para detectar posibles
    extrapolaciones del modelo.
    """

    required_files = [
        TRAIN_DATA_PATH,
        SCALER_X_PATH,
        FEATURE_NAMES_PATH,
    ]

    for file_path in required_files:

        if not file_path.exists():

            return {}

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    # --------------------------------------------------------
    # LOAD FEATURE NAMES
    # --------------------------------------------------------

    with open(
        FEATURE_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        feature_names = json.load(f)

    # --------------------------------------------------------
    # VERIFY FEATURES
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in train_df.columns
    ]

    if missing_features:

        return {}

    # --------------------------------------------------------
    # LOAD SCALER
    # --------------------------------------------------------

    scaler_X = joblib.load(
        SCALER_X_PATH
    )

    # --------------------------------------------------------
    # RETURN TRAINING FEATURES TO ORIGINAL SCALE
    # --------------------------------------------------------

    x_scaled = train_df[
        feature_names
    ].astype(float)

    x_original = scaler_X.inverse_transform(
        x_scaled.values
    )

    original_df = pd.DataFrame(
        x_original,
        columns=feature_names,
    )

    # --------------------------------------------------------
    # CALCULATE ORIGINAL RANGES
    # --------------------------------------------------------

    ranges = {}

    for feature in NUMERIC_FEATURES:

        if feature not in original_df.columns:
            continue

        values = pd.to_numeric(
            original_df[feature],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        ranges[feature] = {
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return ranges


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_prediction_inputs(
    input_values: dict,
) -> tuple[list[str], list[str]]:
    """
    Valida las entradas antes de enviarlas al modelo.

    Regresa:

    - errors:
      problemas que bloquean la predicción.

    - warnings:
      valores fuera del rango ORIGINAL observado durante
      entrenamiento que representan extrapolación.
    """

    errors = []
    warnings = []

    # --------------------------------------------------------
    # LOGICAL VALIDATIONS
    # --------------------------------------------------------

    if input_values["inventory"] > input_values["total_units"]:

        errors.append(
            "El inventario disponible no puede ser mayor "
            "que las unidades totales del desarrollo."
        )

    # --------------------------------------------------------
    # EXPLICIT MODEL DOMAIN
    # --------------------------------------------------------

    if input_values["sqm"] >= SQM_MAX:

        errors.append(
            f"La superficie interior debe ser menor a "
            f"{SQM_MAX:,.0f} m² para este modelo. "
            f"El pipeline fue preparado con ese límite máximo."
        )

    # --------------------------------------------------------
    # TRAINING RANGE WARNINGS
    # --------------------------------------------------------

    training_ranges = get_training_ranges()

    for feature in NUMERIC_FEATURES:

        if feature not in training_ranges:
            continue

        value = float(
            input_values[feature]
        )

        minimum = training_ranges[
            feature
        ]["min"]

        maximum = training_ranges[
            feature
        ]["max"]

        outside_range = (
            value < minimum - EPSILON
            or
            value > maximum + EPSILON
        )

        if outside_range:

            label = FEATURE_LABELS[
                feature
            ]

            warnings.append(
                f"{label}: el valor capturado "
                f"({value:,.1f}) está fuera del "
                f"rango observado durante entrenamiento "
                f"({minimum:,.1f} a {maximum:,.1f})."
            )

    return errors, warnings