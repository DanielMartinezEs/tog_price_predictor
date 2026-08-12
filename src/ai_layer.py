# AI LAYER
# Convierte consultas inmobiliarias en lenguaje natural
# a variables estructuradas para TOG Price Predictor.
#
# IMPORTANTE:
# La IA NO calcula precios.
# El precio continúa siendo calculado por el modelo Ridge Regression.

import json
import os

from openai import OpenAI
from pydantic import BaseModel

from src.predict import (
    get_available_towns,
    get_classification_options,
)


# ============================================================
# CONFIG
# ============================================================

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.4-nano",
)


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class PropertyExtraction(BaseModel):
    """
    Variables que la IA debe extraer.

    Todos los campos deben existir en la respuesta.
    Si el usuario no proporciona un valor,
    el campo debe regresar como None.
    """

    town: str | None
    classification: str | None

    sqm: float | None
    terrace: float | None

    bhk: int | None
    park_u: int | None

    levels: int | None
    months_in_sale: float | None

    total_units: float | None
    master_plan_units: float | None

    inventory: float | None
    months_to_delivery: float | None


# ============================================================
# NATURAL LANGUAGE INTERPRETATION
# ============================================================

def interpret_property_query(
    query: str,
) -> dict:
    """
    Convierte una descripción inmobiliaria en lenguaje natural
    a las variables estructuradas utilizadas por el modelo.

    Esta función únicamente extrae información.
    No calcula ni estima precios.
    """

    if not query.strip():

        raise ValueError(
            "La consulta no puede estar vacía."
        )

    towns = get_available_towns()

    classifications = (
        get_classification_options()
    )

    client = OpenAI()

    instructions = f"""
Eres una capa de extracción de datos para una aplicación
de Machine Learning inmobiliario.

Tu única función es convertir el texto del usuario en
variables estructuradas.

REGLAS OBLIGATORIAS:

1. Extrae únicamente información proporcionada explícitamente
   por el usuario.

2. Nunca inventes valores faltantes.

3. Cuando un dato no esté presente, devuelve null.

4. No calcules precios.

5. No estimes precios.

6. No hagas supuestos sobre características del inmueble.

7. Municipios válidos:
   {towns}

8. Si el municipio escrito por el usuario corresponde claramente
   a uno de los municipios válidos, normalízalo exactamente al
   nombre utilizado en la lista.

9. Clasificaciones SOFTEC válidas:
   {classifications}

10. Solo extrae classification cuando el usuario proporcione
    explícitamente el código SOFTEC.

11. Nunca deduzcas classification a partir de ubicación,
    tamaño, descripción o precio.

DEFINICIÓN DE VARIABLES:

town:
municipio.

classification:
código de clasificación SOFTEC.

sqm:
superficie interior del departamento en metros cuadrados.

terrace:
superficie de terraza en metros cuadrados.

bhk:
número de recámaras.

park_u:
número de cajones de estacionamiento.

levels:
número de niveles del desarrollo.

months_in_sale:
meses que el desarrollo lleva en venta.

total_units:
unidades totales del desarrollo.

master_plan_units:
unidades del master plan.

inventory:
inventario disponible.

months_to_delivery:
meses restantes para entrega.
"""

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        text_format=PropertyExtraction,
    )

    parsed = None

    for output in response.output:

        if output.type != "message":
            continue

        for content in output.content:

            if (
                content.type == "output_text"
                and content.parsed is not None
            ):

                parsed = content.parsed
                break

        if parsed is not None:
            break

    if parsed is None:

        raise RuntimeError(
            "La IA no devolvió una extracción estructurada."
        )

    extracted = parsed.model_dump()

    # ========================================================
    # DETERMINISTIC POST-VALIDATION
    # ========================================================

    town_lookup = {
        town.casefold(): town
        for town in towns
    }

    if extracted["town"] is not None:

        normalized_town = town_lookup.get(
            extracted["town"].strip().casefold()
        )

        extracted["town"] = normalized_town

    if extracted["classification"] is not None:

        classification = (
            extracted["classification"]
            .strip()
            .upper()
        )

        if classification in classifications:

            extracted[
                "classification"
            ] = classification

        else:

            extracted[
                "classification"
            ] = None

    return extracted


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    example_query = (
        "Tengo un departamento de 85 metros cuadrados "
        "en Zapopan, con 2 recámaras y 1 cajón "
        "de estacionamiento."
    )

    result = interpret_property_query(
        example_query
    )

    print("Consulta:")
    print(example_query)

    print("\nDatos extraídos:")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )