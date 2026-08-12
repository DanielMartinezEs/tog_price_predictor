# TOG PRICE PREDICTOR - STREAMLIT APP
# Versión visual robusta previa a integración de IA generativa.

from pathlib import Path
import json

import streamlit as st

from src.predict import (
    build_input,
    predict_price,
    get_available_towns,
    get_classification_options,
)

from src.validation import (
    get_training_ranges,
    validate_prediction_inputs,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TOG Price Predictor",
    page_icon="🏢",
    layout="wide",
)


# ============================================================
# LOAD MODEL METRICS
# ============================================================

metrics_path = Path("reports/metrics.json")

metrics = {}

if metrics_path.exists():

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)


# ============================================================
# HEADER
# ============================================================

st.title("🏢 TOG Price Predictor")

st.write(
    """
    Estima el precio de un departamento en la Zona Metropolitana
    de Guadalajara a partir de sus características y las del
    desarrollo.
    """
)

st.caption(
    "La estimación es generada por un modelo de Machine Learning "
    "Ridge Regression."
)

st.divider()


# ============================================================
# AVAILABLE OPTIONS
# ============================================================

towns = get_available_towns()
classifications = get_classification_options()


# ============================================================
# INPUT FORM
# ============================================================

st.subheader("1. Captura las características")

st.write(
    "Completa los datos del departamento y del desarrollo."
)

with st.form("prediction_form"):

    col_unit, col_development = st.columns(2)

    # --------------------------------------------------------
    # UNIT DATA
    # --------------------------------------------------------

    with col_unit:

        st.markdown("### 🏠 Departamento")

        town = st.selectbox(
            "Municipio",
            options=towns,
            help=(
                "Municipio donde se ubica el departamento."
            ),
        )

        classification = st.selectbox(
            "Clasificación SOFTEC (código)",
            options=classifications,
            help=(
                "Código de clasificación utilizado en el dataset "
                "original y por el modelo."
            ),
        )

        sqm = st.number_input(
            "Superficie interior del departamento (m²)",
            min_value=1.0,
            value=85.0,
            step=1.0,
            format="%.1f",
            help=(
                "Superficie interior habitable, sin incluir terraza."
            ),
        )

        terrace = st.number_input(
            "Superficie de terraza (m²)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.1f",
            help=(
                "Si el departamento no tiene terraza, captura 0."
            ),
        )

        bhk = st.number_input(
            "Número de recámaras",
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

    # --------------------------------------------------------
    # DEVELOPMENT DATA
    # --------------------------------------------------------

    with col_development:

        st.markdown("### 🏗️ Desarrollo")

        levels = st.number_input(
            "Número de niveles del desarrollo",
            min_value=1,
            value=8,
            step=1,
        )

        months_in_sale = st.number_input(
            "Meses en venta",
            min_value=0,
            value=12,
            step=1,
            help=(
                "Meses que el desarrollo lleva "
                "en comercialización."
            ),
        )

        total_units = st.number_input(
            "Unidades totales del desarrollo",
            min_value=1,
            value=100,
            step=1,
        )

        master_plan_units = st.number_input(
            "Unidades del master plan",
            min_value=1,
            value=100,
            step=1,
        )

        inventory = st.number_input(
            "Inventario disponible",
            min_value=0,
            value=30,
            step=1,
        )

        months_to_delivery = st.number_input(
            "Meses para entrega",
            min_value=0,
            value=12,
            step=1,
        )

    st.write("")

    submitted = st.form_submit_button(
        "Calcular precio estimado",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    input_values = {
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

    validation_errors, validation_warnings = (
        validate_prediction_inputs(
            input_values
        )
    )

    # --------------------------------------------------------
    # BLOCKING ERRORS
    # --------------------------------------------------------

    if validation_errors:

        st.divider()

        st.error(
            "Revisa los datos capturados antes de continuar."
        )

        for validation_error in validation_errors:

            st.write(
                f"- {validation_error}"
            )

    else:

        # ----------------------------------------------------
        # EXTRAPOLATION WARNINGS
        # ----------------------------------------------------

        if validation_warnings:

            st.warning(
                "Algunos valores están fuera de los rangos "
                "observados durante el entrenamiento. "
                "La predicción representa una extrapolación "
                "y debe interpretarse con mayor precaución."
            )

            with st.expander(
                "Ver advertencias de rango"
            ):

                for warning in validation_warnings:

                    st.write(
                        f"- {warning}"
                    )

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

            prediction = predict_price(
                input_df
            )

            price_per_sqm = (
                prediction / sqm
            )

            st.divider()

            st.subheader(
                "2. Resultado de la estimación"
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    label="Precio estimado",
                    value=f"${prediction:,.0f} MXN",
                )

            with result_col2:

                st.metric(
                    label="Precio estimado por m² interior",
                    value=(
                        f"${price_per_sqm:,.0f} MXN/m²"
                    ),
                )

            st.success(
                "Predicción generada correctamente."
            )

            # ------------------------------------------------
            # SIMPLE INTERPRETATION
            # ------------------------------------------------

            st.markdown(
                "### ¿Cómo interpretar este resultado?"
            )

            st.write(
                f"""
                Para un departamento de **{sqm:,.0f} m² interiores**
                ubicado en **{town}**, el modelo estima un precio de
                **${prediction:,.0f} MXN**.
                """
            )

            st.write(
                f"""
                Esto equivale aproximadamente a
                **${price_per_sqm:,.0f} MXN por m² interior**.
                """
            )

            if validation_warnings:

                st.warning(
                    "Esta estimación utiliza uno o más valores "
                    "fuera del rango observado durante el "
                    "entrenamiento del modelo."
                )

            if "MAE_test_pesos" in metrics:

                mae = metrics[
                    "MAE_test_pesos"
                ]

                st.info(
                    f"Como referencia, en el conjunto de prueba "
                    f"el modelo tuvo un error absoluto promedio "
                    f"(MAE) de ${mae:,.0f} MXN."
                )

                st.caption(
                    "El MAE describe el error promedio observado "
                    "durante la evaluación del modelo. "
                    "No representa un intervalo de confianza "
                    "para esta predicción individual."
                )

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            with st.expander(
                "Ver datos utilizados para esta estimación"
            ):

                summary_col1, summary_col2 = (
                    st.columns(2)
                )

                with summary_col1:

                    st.write(
                        f"**Municipio:** {town}"
                    )

                    st.write(
                        f"**Clasificación SOFTEC:** "
                        f"{classification}"
                    )

                    st.write(
                        f"**Superficie interior:** "
                        f"{sqm:,.1f} m²"
                    )

                    st.write(
                        f"**Terraza:** "
                        f"{terrace:,.1f} m²"
                    )

                    st.write(
                        f"**Recámaras:** {bhk}"
                    )

                    st.write(
                        f"**Estacionamientos:** "
                        f"{park_u}"
                    )

                with summary_col2:

                    st.write(
                        f"**Niveles:** {levels}"
                    )

                    st.write(
                        f"**Meses en venta:** "
                        f"{months_in_sale}"
                    )

                    st.write(
                        f"**Unidades totales:** "
                        f"{total_units}"
                    )

                    st.write(
                        f"**Unidades master plan:** "
                        f"{master_plan_units}"
                    )

                    st.write(
                        f"**Inventario:** {inventory}"
                    )

                    st.write(
                        f"**Meses para entrega:** "
                        f"{months_to_delivery}"
                    )

        except Exception as error:

            st.error(
                f"No fue posible generar la predicción: "
                f"{error}"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "📊 Información técnica del modelo"
):

    st.write(
        """
        El modelo actual es **Ridge Regression**.
        Las métricas corresponden al conjunto de prueba.
        """
    )

    if metrics:

        metric_col1, metric_col2, metric_col3 = (
            st.columns(3)
        )

        with metric_col1:

            st.metric(
                "R² Test",
                f"{metrics.get('R2_test', 0):.3f}",
            )

        with metric_col2:

            st.metric(
                "MAE",
                f"${metrics.get('MAE_test_pesos', 0):,.0f}",
            )

        with metric_col3:

            st.metric(
                "RMSE",
                f"${metrics.get('RMSE_test_pesos', 0):,.0f}",
            )

        st.write(
            """
            **R²:** proporción de la variabilidad del precio que
            el modelo logra explicar en los datos de prueba.

            **MAE:** diferencia absoluta promedio entre el precio
            real y el precio estimado.

            **RMSE:** medida del error que penaliza con mayor fuerza
            las predicciones con errores grandes.
            """
        )

    else:

        st.warning(
            "No se encontró reports/metrics.json. "
            "Ejecuta primero: dvc repro"
        )


# ============================================================
# TRAINING DOMAIN INFORMATION
# ============================================================

with st.expander(
    "🔎 Rangos observados durante el entrenamiento"
):

    training_ranges = get_training_ranges()

    if training_ranges:

        st.write(
            """
            Estos rangos corresponden a los valores mínimo y máximo
            observados en el conjunto utilizado para entrenar el
            modelo. Una entrada fuera de estos rangos implica que
            el modelo está realizando una extrapolación.
            """
        )

        range_labels = {
            "sqm": "Superficie interior (m²)",
            "terrace": "Terraza (m²)",
            "bhk": "Recámaras",
            "park_u": "Estacionamientos",
            "levels": "Niveles",
            "months_in_sale": "Meses en venta",
            "total_units": "Unidades totales",
            "master_plan_units": "Unidades master plan",
            "inventory": "Inventario",
            "months_to_delivery": "Meses para entrega",
        }

        for feature, values in training_ranges.items():

            st.write(
                f"**{range_labels.get(feature, feature)}:** "
                f"{values['min']:,.1f} a "
                f"{values['max']:,.1f}"
            )

    else:

        st.warning(
            "No fue posible leer los rangos del conjunto "
            "de entrenamiento."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TOG Price Predictor · La predicción es una estimación "
    "estadística y no sustituye una valuación inmobiliaria formal."
)