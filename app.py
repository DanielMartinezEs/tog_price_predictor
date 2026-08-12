# TOG PRICE PREDICTOR - STREAMLIT APP
# Version visual con integracion opcional de IA generativa.
#
# IMPORTANTE:
# La IA generativa solo interpreta texto y extrae variables.
# La prediccion de precio sigue siendo responsabilidad del
# modelo Ridge Regression.

from pathlib import Path
import json

import streamlit as st

from src.ai_layer import interpret_property_query
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
    "Ridge Regression. La IA generativa es una capa opcional para "
    "interpretar descripciones en lenguaje natural."
)

st.divider()


# ============================================================
# AVAILABLE OPTIONS
# ============================================================

towns = get_available_towns()
classifications = get_classification_options()

if not towns or not classifications:
    st.error(
        "No fue posible cargar las opciones necesarias para la "
        "aplicación. Verifica los artefactos del modelo."
    )
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

default_input_values = {
    "town": towns[0],
    "classification": classifications[0],
    "sqm": 85.0,
    "terrace": 0.0,
    "bhk": 2,
    "park_u": 1,
    "levels": 8,
    "months_in_sale": 12,
    "total_units": 100,
    "master_plan_units": 100,
    "inventory": 30,
    "months_to_delivery": 12,
}

for field, default_value in default_input_values.items():
    if field not in st.session_state:
        st.session_state[field] = default_value

if "ai_query" not in st.session_state:
    st.session_state["ai_query"] = ""

if "ai_last_extraction" not in st.session_state:
    st.session_state["ai_last_extraction"] = None

if "ai_applied_fields" not in st.session_state:
    st.session_state["ai_applied_fields"] = []

if "ai_missing_fields" not in st.session_state:
    st.session_state["ai_missing_fields"] = []

if "ai_skipped_fields" not in st.session_state:
    st.session_state["ai_skipped_fields"] = []

if "ai_last_error" not in st.session_state:
    st.session_state["ai_last_error"] = None

if "ai_missing_confirmed" not in st.session_state:
    st.session_state["ai_missing_confirmed"] = False


# ============================================================
# AI HELPERS
# ============================================================

field_labels = {
    "town": "Municipio",
    "classification": "Clasificación SOFTEC",
    "sqm": "Superficie interior",
    "terrace": "Superficie de terraza",
    "bhk": "Recámaras",
    "park_u": "Estacionamientos",
    "levels": "Niveles del desarrollo",
    "months_in_sale": "Meses en venta",
    "total_units": "Unidades totales",
    "master_plan_units": "Unidades del master plan",
    "inventory": "Inventario disponible",
    "months_to_delivery": "Meses para entrega",
}

integer_fields = {
    "bhk",
    "park_u",
    "levels",
    "months_in_sale",
    "total_units",
    "master_plan_units",
    "inventory",
    "months_to_delivery",
}

minimum_values = {
    "sqm": 1.0,
    "terrace": 0.0,
    "bhk": 0,
    "park_u": 0,
    "levels": 1,
    "months_in_sale": 0,
    "total_units": 1,
    "master_plan_units": 1,
    "inventory": 0,
    "months_to_delivery": 0,
}


def apply_ai_extraction(
    extracted: dict,
) -> tuple[list[str], list[str], list[str]]:
    """
    Aplica al formulario únicamente valores válidos extraídos por la IA.

    Los campos no encontrados conservan el valor que ya tenía el
    formulario para que el usuario pueda revisarlos o completarlos.
    """

    applied_fields = []
    missing_fields = []
    skipped_fields = []

    for field in default_input_values:

        value = extracted.get(field)

        if value is None:
            missing_fields.append(field)
            continue

        if field == "town":

            if value in towns:
                st.session_state[field] = value
                applied_fields.append(field)
            else:
                skipped_fields.append(field)

            continue

        if field == "classification":

            if value in classifications:
                st.session_state[field] = value
                applied_fields.append(field)
            else:
                skipped_fields.append(field)

            continue

        try:
            numeric_value = float(value)

        except (TypeError, ValueError):
            skipped_fields.append(field)
            continue

        if numeric_value < minimum_values[field]:
            skipped_fields.append(field)
            continue

        if field in integer_fields:

            if not numeric_value.is_integer():
                skipped_fields.append(field)
                continue

            st.session_state[field] = int(
                numeric_value
            )

        else:

            st.session_state[field] = float(
                numeric_value
            )

        applied_fields.append(field)

    return (
        applied_fields,
        missing_fields,
        skipped_fields,
    )


def labels_for(fields: list[str]) -> str:

    return ", ".join(
        field_labels.get(field, field)
        for field in fields
    )


# ============================================================
# OPTIONAL GENERATIVE AI
# ============================================================

st.subheader(
    "1. Opcional: describe el inmueble con IA"
)

st.write(
    """
    Puedes escribir una descripción en lenguaje natural y la IA
    intentará extraer únicamente los datos que estén explícitamente
    presentes. Después podrás revisar y modificar todos los campos
    antes de calcular el precio.
    """
)

st.caption(
    "La IA no calcula el precio. Si no deseas utilizarla o la API "
    "no está disponible, puedes completar el formulario manualmente."
)

with st.form("ai_interpretation_form"):

    ai_query = st.text_area(
        "Descripción del departamento y del desarrollo",
        key="ai_query",
        height=130,
        placeholder=(
            "Ejemplo: Departamento en Zapopan, clasificación SOFTEC R, "
            "85 m² interiores, 2 recámaras y 1 estacionamiento..."
        ),
    )

    interpret_submitted = st.form_submit_button(
        "🤖 Interpretar con IA",
        use_container_width=True,
    )


if interpret_submitted:

    st.session_state[
        "ai_last_error"
    ] = None

    # Cada nueva interpretación requiere una nueva revisión
    # de posibles campos faltantes.
    st.session_state[
        "ai_missing_confirmed"
    ] = False

    if not ai_query.strip():

        st.session_state[
            "ai_last_extraction"
        ] = None

        st.session_state[
            "ai_applied_fields"
        ] = []

        st.session_state[
            "ai_missing_fields"
        ] = []

        st.session_state[
            "ai_skipped_fields"
        ] = []

        st.warning(
            "Escribe una descripción antes de utilizar la IA."
        )

    else:

        try:

            with st.spinner(
                "Interpretando la descripción..."
            ):

                extracted = interpret_property_query(
                    ai_query
                )

            (
                applied_fields,
                missing_fields,
                skipped_fields,
            ) = apply_ai_extraction(
                extracted
            )

            st.session_state[
                "ai_last_extraction"
            ] = extracted

            st.session_state[
                "ai_applied_fields"
            ] = applied_fields

            st.session_state[
                "ai_missing_fields"
            ] = missing_fields

            st.session_state[
                "ai_skipped_fields"
            ] = skipped_fields

        except Exception as error:

            st.session_state[
                "ai_last_error"
            ] = str(error)

            st.session_state[
                "ai_last_extraction"
            ] = None

            st.session_state[
                "ai_applied_fields"
            ] = []

            st.session_state[
                "ai_missing_fields"
            ] = []

            st.session_state[
                "ai_skipped_fields"
            ] = []


# ============================================================
# AI FEEDBACK
# ============================================================

if st.session_state["ai_last_error"]:

    st.error(
        "La interpretación con IA no está disponible en este "
        "momento. Puedes continuar utilizando el formulario "
        "manualmente."
    )

    with st.expander(
        "Ver detalle técnico del error"
    ):

        st.code(
            st.session_state[
                "ai_last_error"
            ]
        )


elif (
    st.session_state[
        "ai_last_extraction"
    ]
    is not None
):

    applied_fields = st.session_state[
        "ai_applied_fields"
    ]

    missing_fields = st.session_state[
        "ai_missing_fields"
    ]

    skipped_fields = st.session_state[
        "ai_skipped_fields"
    ]

    if applied_fields:

        st.success(
            "La descripción fue interpretada. "
            "Revisa los campos autocompletados antes de predecir."
        )

        st.write(
            "**Campos autocompletados:** "
            f"{labels_for(applied_fields)}."
        )

    else:

        st.warning(
            "La IA no encontró datos que pudiera aplicar "
            "automáticamente al formulario."
        )

    if missing_fields:

        st.warning(
            "La descripción no contiene toda la información "
            "requerida por el modelo. "
            f"La IA identificó "
            f"{len(applied_fields)} de "
            f"{len(default_input_values)} variables."
        )

        st.write(
            "**Campos que deben revisarse o completarse "
            "manualmente:** "
            f"{labels_for(missing_fields)}."
        )

        st.caption(
            "Los campos no identificados conservaron los valores "
            "que ya tenía el formulario. Esos valores NO fueron "
            "inferidos por la IA."
        )

    if skipped_fields:

        st.warning(
            "Algunos valores extraídos no pudieron aplicarse "
            "porque no cumplen el formato básico del formulario: "
            f"{labels_for(skipped_fields)}."
        )

    with st.expander(
        "Ver extracción estructurada de la IA"
    ):

        st.json(
            st.session_state[
                "ai_last_extraction"
            ]
        )


st.divider()


# ============================================================
# INPUT FORM
# ============================================================

st.subheader(
    "2. Revisa y completa las características"
)

st.write(
    """
    Verifica todos los datos antes de calcular el precio.
    Los campos pueden capturarse o modificarse manualmente
    aunque hayas utilizado la IA.
    """
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
            key="town",
            help=(
                "Municipio donde se ubica el departamento."
            ),
        )

        classification = st.selectbox(
            "Clasificación SOFTEC (código)",
            options=classifications,
            key="classification",
            help=(
                "Código de clasificación utilizado en el dataset "
                "original y por el modelo."
            ),
        )

        sqm = st.number_input(
            "Superficie interior del departamento (m²)",
            min_value=1.0,
            step=1.0,
            format="%.1f",
            key="sqm",
            help=(
                "Superficie interior habitable, sin incluir terraza."
            ),
        )

        terrace = st.number_input(
            "Superficie de terraza (m²)",
            min_value=0.0,
            step=1.0,
            format="%.1f",
            key="terrace",
            help=(
                "Si el departamento no tiene terraza, captura 0."
            ),
        )

        bhk = st.number_input(
            "Número de recámaras",
            min_value=0,
            step=1,
            key="bhk",
        )

        park_u = st.number_input(
            "Cajones de estacionamiento",
            min_value=0,
            step=1,
            key="park_u",
        )

    # --------------------------------------------------------
    # DEVELOPMENT DATA
    # --------------------------------------------------------

    with col_development:

        st.markdown("### 🏗️ Desarrollo")

        levels = st.number_input(
            "Número de niveles del desarrollo",
            min_value=1,
            step=1,
            key="levels",
        )

        months_in_sale = st.number_input(
            "Meses en venta",
            min_value=0,
            step=1,
            key="months_in_sale",
            help=(
                "Meses que el desarrollo lleva "
                "en comercialización."
            ),
        )

        total_units = st.number_input(
            "Unidades totales del desarrollo",
            min_value=1,
            step=1,
            key="total_units",
        )

        master_plan_units = st.number_input(
            "Unidades del master plan",
            min_value=1,
            step=1,
            key="master_plan_units",
        )

        inventory = st.number_input(
            "Inventario disponible",
            min_value=0,
            step=1,
            key="inventory",
        )

        months_to_delivery = st.number_input(
            "Meses para entrega",
            min_value=0,
            step=1,
            key="months_to_delivery",
        )

    st.write("")

    ai_requires_confirmation = (
        st.session_state["ai_last_extraction"] is not None
        and bool(
            st.session_state["ai_missing_fields"]
        )
    )

    if ai_requires_confirmation:

        st.warning(
            "La IA recibió una descripción incompleta. "
            "Antes de calcular el precio, revisa manualmente "
            "los campos que no fueron identificados."
        )

        st.checkbox(
            "Confirmo que revisé y completé los campos "
            "que la IA no pudo identificar.",
            key="ai_missing_confirmed",
        )

    submitted = st.form_submit_button(
        "Calcular precio estimado",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    ai_requires_confirmation = (
        st.session_state["ai_last_extraction"] is not None
        and bool(
            st.session_state["ai_missing_fields"]
        )
    )

    if (
        ai_requires_confirmation
        and not st.session_state[
            "ai_missing_confirmed"
        ]
    ):

        st.error(
            "Antes de calcular el precio, confirma que "
            "revisaste y completaste manualmente los campos "
            "que la IA no pudo identificar."
        )

        st.stop()

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
                "3. Resultado de la estimación"
            )

            result_col1, result_col2 = (
                st.columns(2)
            )

            with result_col1:

                st.metric(
                    label="Precio estimado",
                    value=(
                        f"${prediction:,.0f} MXN"
                    ),
                )

            with result_col2:

                st.metric(
                    label=(
                        "Precio estimado por m² interior"
                    ),
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
                (
                    f"${metrics.get('MAE_test_pesos', 0):,.0f}"
                ),
            )

        with metric_col3:

            st.metric(
                "RMSE",
                (
                    f"${metrics.get('RMSE_test_pesos', 0):,.0f}"
                ),
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

        for feature, values in (
            training_ranges.items()
        ):

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