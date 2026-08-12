# TOG PRICE PREDICTOR - STREAMLIT APP
# V1.2 - Interfaz amigable, ayudas de captura y validaciones.

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
    de Guadalajara a partir de sus características y las del desarrollo.
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
    "Completa los datos del departamento y del desarrollo. "
    "Los campos incluyen ayudas para facilitar la captura."
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
                "Municipio de la Zona Metropolitana de Guadalajara "
                "donde se ubica el departamento."
            ),
        )

        classification = st.selectbox(
            "Clasificación SOFTEC (código)",
            options=classifications,
            help=(
                "Código de clasificación utilizado en el dataset "
                "original. Se conserva el código SOFTEC para no "
                "alterar la variable que utiliza el modelo."
            ),
        )

        sqm = st.number_input(
            "Superficie interior del departamento (m²)",
            min_value=1.0,
            value=85.0,
            step=1.0,
            format="%.1f",
            help=(
                "Superficie interior habitable del departamento, "
                "sin incluir la terraza."
            ),
        )

        terrace = st.number_input(
            "Superficie de terraza (m²)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.1f",
            help=(
                "Superficie de terraza reportada para la unidad. "
                "Si no tiene terraza, captura 0."
            ),
        )

        bhk = st.number_input(
            "Número de recámaras",
            min_value=0,
            value=2,
            step=1,
            help="Número de recámaras del departamento.",
        )

        park_u = st.number_input(
            "Cajones de estacionamiento",
            min_value=0,
            value=1,
            step=1,
            help=(
                "Número de cajones de estacionamiento asignados "
                "a la unidad."
            ),
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
            help=(
                "Número de niveles reportados para el desarrollo."
            ),
        )

        months_in_sale = st.number_input(
            "Meses en venta",
            min_value=0,
            value=12,
            step=1,
            help=(
                "Número de meses que el desarrollo lleva "
                "en comercialización."
            ),
        )

        total_units = st.number_input(
            "Unidades totales del desarrollo",
            min_value=1,
            value=100,
            step=1,
            help=(
                "Número total de unidades reportadas para "
                "el desarrollo."
            ),
        )

        master_plan_units = st.number_input(
            "Unidades del master plan",
            min_value=1,
            value=100,
            step=1,
            help=(
                "Número de unidades reportadas en el master plan "
                "del proyecto."
            ),
        )

        inventory = st.number_input(
            "Inventario disponible",
            min_value=0,
            value=30,
            step=1,
            help=(
                "Número de unidades reportadas actualmente como "
                "inventario disponible."
            ),
        )

        months_to_delivery = st.number_input(
            "Meses para entrega",
            min_value=0,
            value=12,
            step=1,
            help=(
                "Número de meses reportados hasta la entrega "
                "del desarrollo."
            ),
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

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    validation_errors = []

    if inventory > total_units:
        validation_errors.append(
            "El inventario disponible no puede ser mayor "
            "que las unidades totales del desarrollo."
        )

    if validation_errors:

        st.divider()

        st.error("Revisa los datos capturados antes de continuar.")

        for validation_error in validation_errors:
            st.write(f"- {validation_error}")

    else:

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

            price_per_sqm = prediction / sqm

            st.divider()

            st.subheader("2. Resultado de la estimación")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    label="Precio estimado",
                    value=f"${prediction:,.0f} MXN",
                )

            with result_col2:

                st.metric(
                    label="Precio estimado por m² interior",
                    value=f"${price_per_sqm:,.0f} MXN/m²",
                )

            st.success(
                "Predicción generada correctamente."
            )

            # ------------------------------------------------
            # SIMPLE EXPLANATION
            # ------------------------------------------------

            st.markdown("### ¿Cómo interpretar este resultado?")

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

            if "MAE_test_pesos" in metrics:

                mae = metrics["MAE_test_pesos"]

                st.info(
                    f"Como referencia, en el conjunto de prueba el "
                    f"modelo tuvo un error absoluto promedio (MAE) "
                    f"de ${mae:,.0f} MXN."
                )

                st.caption(
                    "El MAE describe el error promedio observado "
                    "durante la evaluación del modelo. No representa "
                    "un intervalo de confianza para esta predicción "
                    "individual."
                )

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            with st.expander(
                "Ver datos utilizados para esta estimación"
            ):

                summary_col1, summary_col2 = st.columns(2)

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
                        f"**Estacionamientos:** {park_u}"
                    )

                with summary_col2:

                    st.write(
                        f"**Niveles:** {levels}"
                    )

                    st.write(
                        f"**Meses en venta:** {months_in_sale}"
                    )

                    st.write(
                        f"**Unidades totales:** {total_units}"
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
                f"No fue posible generar la predicción: {error}"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("📊 Información técnica del modelo"):

    st.write(
        """
        El modelo actual es **Ridge Regression**.
        Las siguientes métricas corresponden al conjunto de prueba.
        """
    )

    if metrics:

        metric_col1, metric_col2, metric_col3 = st.columns(3)

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
            **R²:** proporción de la variabilidad del precio que el
            modelo logra explicar en los datos de prueba.

            **MAE:** diferencia absoluta promedio entre el precio real
            y el precio estimado.

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
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TOG Price Predictor · La predicción es una estimación "
    "estadística y no sustituye una valuación inmobiliaria formal."
)