# TOG PRICE PREDICTOR - STREAMLIT APP
# Primera versión visual de la herramienta de predicción.

from pathlib import Path
import json

import streamlit as st

from src.predict import (
    build_input,
    predict_price,
    get_available_towns,
    get_classification_options,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TOG Price Predictor",
    page_icon="🏢",
    layout="centered",
)


# ============================================================
# HEADER
# ============================================================

st.title("🏢 TOG Price Predictor")

st.write(
    """
    Herramienta para estimar el precio de un departamento
    en la Zona Metropolitana de Guadalajara.
    """
)

st.caption(
    "Modelo predictivo: Ridge Regression"
)


# ============================================================
# AVAILABLE OPTIONS
# ============================================================

towns = get_available_towns()
classifications = get_classification_options()


# ============================================================
# INPUT FORM
# ============================================================

with st.form("prediction_form"):

    st.subheader("Características del departamento")

    town = st.selectbox(
        "Municipio",
        options=towns,
    )

    classification = st.selectbox(
        "Clasificación SOFTEC",
        options=classifications,
    )

    sqm = st.number_input(
        "Superficie interior (m²)",
        min_value=1.0,
        value=85.0,
        step=1.0,
    )

    terrace = st.number_input(
        "Terraza (m²)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    bhk = st.number_input(
        "Recámaras",
        min_value=0,
        value=2,
        step=1,
    )

    park_u = st.number_input(
        "Cajones de estacionamiento",
        min_value=0,
        value=1,
        step=1,
    )

    levels = st.number_input(
        "Niveles del desarrollo",
        min_value=1,
        value=8,
        step=1,
    )

    months_in_sale = st.number_input(
        "Meses en venta",
        min_value=0.0,
        value=12.0,
        step=1.0,
    )

    total_units = st.number_input(
        "Unidades totales",
        min_value=1.0,
        value=100.0,
        step=1.0,
    )

    master_plan_units = st.number_input(
        "Unidades del master plan",
        min_value=1.0,
        value=100.0,
        step=1.0,
    )

    inventory = st.number_input(
        "Inventario disponible",
        min_value=0.0,
        value=30.0,
        step=1.0,
    )

    months_to_delivery = st.number_input(
        "Meses para entrega",
        min_value=0.0,
        value=12.0,
        step=1.0,
    )

    submitted = st.form_submit_button(
        "Estimar precio",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    try:

        input_df = build_input(
            town=town,
            classification=classification,
            sqm=sqm,
            terrace=terrace,
            bhk=bhk,
            park_u=park_u,
            levels=levels,
            months_in_sale=months_in_sale,
            total_units=total_units,
            master_plan_units=master_plan_units,
            inventory=inventory,
            months_to_delivery=months_to_delivery,
        )

        prediction = predict_price(input_df)

        st.divider()

        st.subheader("Resultado")

        st.metric(
            label="Precio estimado",
            value=f"${prediction:,.0f} MXN",
        )

        st.caption(
            "El resultado es una estimación estadística basada "
            "en las características capturadas y en los datos "
            "utilizados para entrenar el modelo."
        )

    except Exception as error:

        st.error(
            f"No fue posible generar la predicción: {error}"
        )


# ============================================================
# MODEL METRICS
# ============================================================

st.divider()

st.subheader("Desempeño del modelo")

metrics_path = Path("reports/metrics.json")

if metrics_path.exists():

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "R² Test",
            f"{metrics.get('R2_test', 0):.3f}",
        )

    with col2:
        st.metric(
            "MAE",
            f"${metrics.get('MAE_test_pesos', 0):,.0f}",
        )

    with col3:
        st.metric(
            "RMSE",
            f"${metrics.get('RMSE_test_pesos', 0):,.0f}",
        )

    st.caption(
        "MAE y RMSE están expresados en pesos mexicanos."
    )

else:

    st.warning(
        "No se encontró reports/metrics.json. "
        "Ejecuta primero: dvc repro"
    )