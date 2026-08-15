from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ============================================================
# SALIDA
# ============================================================

OUTPUT_PATH = Path(
    "reports/figures/arquitectura_tog_price_predictor.png"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def add_box(ax, x, y, width, height, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
        },
    )


# ============================================================
# LIENZO
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 9)
)

ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")


# ============================================================
# TITULO
# ============================================================

ax.text(
    8,
    9.5,
    "Arquitectura general del TOG Price Predictor",
    ha="center",
    va="center",
    fontsize=17,
    fontweight="bold",
)


# ============================================================
# FLUJO DE PREPARACION Y ENTRENAMIENTO
# ============================================================

ax.text(
    0.6,
    8.6,
    "Preparación y entrenamiento reproducible",
    fontsize=13,
    fontweight="bold",
)

box_w = 2.05
box_h = 1.05
y_train = 6.9

training_boxes = [
    (0.5, "DIME Guadalajara\n4Q22"),
    (3.0, "Preparación\nde datos"),
    (5.5, "División\nTrain / Test"),
    (8.0, "Entrenamiento\nRidge"),
    (10.5, "Evaluación\nR², MAE, RMSE"),
    (13.0, "Modelo, scalers\ny variables"),
]

for x, text in training_boxes:
    add_box(
        ax,
        x,
        y_train,
        box_w,
        box_h,
        text,
    )

for i in range(len(training_boxes) - 1):
    add_arrow(
        ax,
        training_boxes[i][0] + box_w,
        y_train + box_h / 2,
        training_boxes[i + 1][0],
        y_train + box_h / 2,
    )

ax.text(
    7.9,
    6.45,
    "Pipeline reproducible administrado con DVC",
    ha="center",
    fontsize=9,
    fontstyle="italic",
)


# ============================================================
# FLUJO DE INFERENCIA
# ============================================================

ax.text(
    0.6,
    5.35,
    "Inferencia y uso de la solución",
    fontsize=13,
    fontweight="bold",
)

y_inf = 3.75

inference_boxes = [
    (0.5, "Usuario"),
    (3.0, "Formulario\nStreamlit"),
    (5.5, "Validación\nde entradas"),
    (8.0, "Construcción y\nescalado de variables"),
    (10.5, "Modelo\nRidge"),
    (13.0, "Precio estimado\nMXN"),
]

for x, text in inference_boxes:
    add_box(
        ax,
        x,
        y_inf,
        box_w,
        box_h,
        text,
    )

for i in range(len(inference_boxes) - 1):
    add_arrow(
        ax,
        inference_boxes[i][0] + box_w,
        y_inf + box_h / 2,
        inference_boxes[i + 1][0],
        y_inf + box_h / 2,
    )


# ============================================================
# CAPA OPCIONAL DE IA GENERATIVA
# ============================================================

ai_x = 1.35
ai_y = 1.35
ai_w = 4.0
ai_h = 1.05

add_box(
    ax,
    ai_x,
    ai_y,
    ai_w,
    ai_h,
    "IA generativa opcional\n"
    "Descripción en lenguaje natural → datos estructurados",
)

add_arrow(
    ax,
    1.5,
    y_inf,
    ai_x + 1.0,
    ai_y + ai_h,
)

add_arrow(
    ax,
    ai_x + ai_w,
    ai_y + ai_h / 2,
    4.0,
    y_inf,
)


# ============================================================
# CONEXION DE ARTEFACTOS ENTRENADOS CON INFERENCIA
# ============================================================

add_arrow(
    ax,
    14.0,
    y_train,
    11.5,
    y_inf + box_h,
)


# ============================================================
# NOTA
# ============================================================

ax.text(
    8,
    0.45,
    "La IA generativa apoya únicamente la captura de información; "
    "la estimación del precio es realizada por el modelo Ridge.",
    ha="center",
    va="center",
    fontsize=10,
    fontstyle="italic",
)


# ============================================================
# GUARDAR
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"Figura guardada en: {OUTPUT_PATH}"
)