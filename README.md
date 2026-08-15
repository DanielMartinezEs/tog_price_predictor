# TOG Price Predictor

Proyecto de Machine Learning desarrollado como Trabajo de Obtención de Grado de la Maestría en Ciencia de Datos del ITESO.

La solución estima el precio de departamentos a partir de características físicas, comerciales y de ubicación utilizando información correspondiente a la **plaza Guadalajara de DIME**.

El proyecto evolucionó desde un prototipo analítico hasta una solución reproducible que integra:

- Modelo final Ridge Regression.
- Pipeline de Machine Learning con DVC.
- Preparación, entrenamiento, evaluación e inferencia por componentes.
- Aplicación visual desarrollada con Streamlit.
- Validaciones de entrada y advertencias de extrapolación.
- Capa opcional de inteligencia artificial generativa para interpretar descripciones en lenguaje natural.

> La IA generativa únicamente facilita la captura de información. La estimación del precio es realizada exclusivamente por el modelo Ridge Regression.

---

## Resultados principales

El modelo final fue evaluado sobre el conjunto de prueba obtenido mediante una partición reproducible de los datos.

| Métrica | Resultado |
|---|---:|
| R² test | 0.85886 |
| RMSE test | $723,726.15 MXN |
| MAE test | $550,099.67 MXN |
| Ridge alpha | 20 |
| R² promedio de validación cruzada | 0.85367 |

El conjunto preparado contiene 536 observaciones. La solución recibe 12 características conceptuales que se transforman internamente en 17 variables de entrada para el modelo.

Las métricas completas pueden consultarse después de reproducir el pipeline mediante:

```bash
dvc metrics show
```

---

## Arquitectura

El flujo reproducible de Machine Learning es:

```text
DIME Guadalajara 4Q22
        ↓
prepare_data
        ↓
split_data
        ↓
train
        ↓
evaluate
        ↓
Modelo y artefactos
        ↓
Inferencia
        ↓
Aplicación Streamlit
        ↓
Usuario
```

La captura asistida mediante IA es opcional:

```text
Descripción en lenguaje natural
        ↓
IA generativa
        ↓
Variables estructuradas
        ↓
Revisión del usuario
        ↓
Validaciones
        ↓
Ridge Regression
        ↓
Precio estimado
```

---

## Estructura principal

```text
.
├── app.py
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
│
├── src/
│   ├── ai_layer.py
│   ├── predict.py
│   ├── validation.py
│   └── stages/
│       ├── prepare_data.py
│       ├── split_data.py
│       ├── train.py
│       └── evaluate.py
│
├── scripts/
│   ├── compare_models.py
│   ├── generate_architecture_diagram.py
│   ├── generate_correlation_heatmap.py
│   └── generate_model_diagnostics.py
│
├── data/
│   └── raw/
│       └── README.md
│
├── fase1_prototipo_monolito_abr2026.py
└── step-*.ipynb
```

Los directorios `data/processed/`, `models/` y los resultados de `reports/` se generan localmente durante la ejecución y no contienen el dataset original dentro del repositorio público.

---

## Datos

El proyecto utiliza como fuente original:

```text
data/raw/Guadalajara 4Q22.xlsx
```

El archivo corresponde a información DIME de Softec adquirida por la empresa donde se desarrolló el proyecto.

**El dataset original no se distribuye ni forma parte de este repositorio público.**

La reproducción completa desde los datos originales requiere contar previamente con acceso autorizado al archivo y colocarlo con el nombre esperado en:

```text
data/raw/Guadalajara 4Q22.xlsx
```

Los datos preparados, conjuntos de entrenamiento y prueba tampoco se publican en GitHub.

---

## Instalación y reproducción

El proyecto fue desarrollado en Python y requiere las dependencias definidas en `requirements.txt`.

### 1. Crear un entorno virtual

En Windows:

```bash
python -m venv entorno_rspp
entorno_rspp\Scripts\activate
```

### 2. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

### 3. Incorporar el dataset autorizado

Colocar el archivo:

```text
data/raw/Guadalajara 4Q22.xlsx
```

### 4. Reproducir el pipeline

```bash
dvc repro
```

El pipeline ejecutará:

```text
prepare_data → split_data → train → evaluate
```

Para verificar su estado:

```bash
dvc status
```

Para consultar las métricas:

```bash
dvc metrics show
```

---

## Ejecutar la aplicación

Después de generar los artefactos del modelo:

```bash
python -m streamlit run app.py
```

La aplicación permite capturar las características del departamento y del desarrollo, validar la información y generar una estimación de precio.

También muestra:

- Precio estimado en MXN.
- Precio estimado por metro cuadrado interior.
- R², MAE y RMSE del modelo.
- Rangos observados durante entrenamiento.
- Advertencias cuando una entrada implica extrapolación.

La herramienta representa una **referencia estadística de apoyo** y no sustituye una valuación inmobiliaria formal ni un estudio de mercado.

---

## IA generativa opcional

La aplicación puede utilizarse completamente de forma manual.

Opcionalmente, `src/ai_layer.py` permite interpretar una descripción del inmueble mediante la API de OpenAI y extraer las 12 características requeridas por la aplicación.

Para utilizar esta funcionalidad debe existir una variable de entorno:

```text
OPENAI_API_KEY
```

La credencial:

- no se almacena en el código;
- no forma parte del repositorio;
- no es necesaria para utilizar Ridge Regression;
- no es necesaria para utilizar el formulario manual.

Cuando la IA no identifica alguna característica, el usuario debe revisar y completar la información antes de ejecutar la estimación.

---

## Validaciones

Antes de realizar una predicción se aplican reglas determinísticas de consistencia.

Entre ellas:

- el inventario no puede superar las unidades totales;
- la superficie interior debe permanecer dentro del límite definido para el proyecto;
- el municipio debe pertenecer a las categorías reconocidas por el modelo.

Además, las variables numéricas se comparan con los rangos observados durante el entrenamiento.

Una entrada fuera de esos rangos genera una **advertencia de extrapolación**, pero no necesariamente bloquea la predicción.

---

## Limitaciones

Los resultados deben interpretarse dentro del alcance del ejercicio:

- El modelo fue entrenado con información de **DIME Guadalajara 4Q22**.
- La ubicación se representa a nivel de municipio.
- El desempeño obtenido no garantiza el mismo comportamiento para periodos posteriores.
- El modelo no debe utilizarse directamente para otras plazas sin nuevo entrenamiento y evaluación.
- El MAE representa un error promedio del conjunto de prueba y no un intervalo de confianza para una predicción individual.
- La herramienta proporciona una referencia cuantitativa y no una valuación inmobiliaria formal.

---

## Autor

**Daniel Martínez Escobosa**  
Maestría en Ciencia de Datos  
ITESO

Trabajo de Obtención de Grado:  
**Estimador de precios de productos inmobiliarios (Escenario Departamentos)**