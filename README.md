# TOG Price Predictor

Proyecto de Machine Learning para estimación de precios de departamentos en la Zona Metropolitana de Guadalajara.

El proyecto incluye:

- Prototipo analítico desarrollado en Fase 1.
- Pipeline reproducible de Machine Learning con DVC.
- Configuración centralizada en `params.yaml`.
- Scripts reutilizables para preparación de datos, split, entrenamiento, evaluación e inferencia.
- Métricas reproducibles del modelo ganador.
- Aplicación visual desarrollada con Streamlit.
- Inferencia directa desde la aplicación utilizando el modelo entrenado.

---

## 1. Estructura principal

```text
tog_dme/
│
├── app.py
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── Guadalajara 4Q22.xlsx
│   │   └── README.md
│   └── processed/
│
├── models/
│
├── reports/
│   └── metrics.json
│
└── src/
    ├── predict.py
    │
    └── stages/
        ├── prepare_data.py
        ├── split_data.py
        ├── train.py
        └── evaluate.py
```

---

## 2. Datos privados

El archivo original de datos no se versiona en GitHub por confidencialidad.

El pipeline espera encontrar el archivo:

```text
data/raw/Guadalajara 4Q22.xlsx
```

Para ejecutar el proyecto, solicitar el archivo privado al autor y colocarlo exactamente en esa ruta.

Más detalles en:

```text
data/raw/README.md
```

---

## 3. Instalación

Crear entorno virtual:

```bash
python -m venv entorno_rspp
```

Activar entorno en Windows:

```bash
entorno_rspp\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## 4. Ejecutar pipeline completo

Con el archivo raw colocado en:

```text
data/raw/Guadalajara 4Q22.xlsx
```

ejecutar:

```bash
dvc repro
```

Para forzar la reproducción completa:

```bash
dvc repro -f
```

El pipeline ejecuta las siguientes etapas:

```text
prepare_data
    ↓
split_data
    ↓
train
    ↓
evaluate
```

---

## 5. Revisar estado del pipeline

Para comprobar si existen cambios pendientes en el pipeline:

```bash
dvc status
```

Si todo está actualizado, DVC deberá indicar que los datos y el pipeline están al día.

Para visualizar el DAG:

```bash
dvc dag
```

---

## 6. Revisar métricas

Ejecutar:

```bash
dvc metrics show
```

Métricas esperadas del modelo actual:

```text
R2_test ≈ 0.85867
RMSE_test_pesos ≈ 724,228.73
MAE_test_pesos ≈ 549,476.61
```

---

## 7. Modelo actual

El modelo seleccionado después de comparar diferentes algoritmos fue:

```text
Ridge Regression
```

Métricas principales:

```text
R2_test = 0.858668
RMSE_test_pesos = 724,228.73
MAE_test_pesos = 549,476.61
```

El modelo explica aproximadamente el 85.9 % de la variabilidad observada en el conjunto de prueba.

---

## 8. Ejecutar predicción individual

Después de ejecutar el pipeline, es posible realizar una predicción directamente desde terminal:

```bash
python src\predict.py
```

El script:

1. Carga el modelo entrenado.
2. Carga los scalers utilizados durante entrenamiento.
3. Recupera las columnas esperadas por el modelo.
4. Construye el registro de entrada.
5. Ejecuta la transformación de variables.
6. Genera la predicción.
7. Convierte el resultado nuevamente a pesos mexicanos.

Ejemplo de entrada:

```text
Municipio: Zapopan
Clasificación: R
Superficie: 85 m²
Terraza: 0 m²
Recámaras: 2
Estacionamientos: 1
Niveles: 8
Meses en venta: 12
Unidades totales: 100
Unidades master plan: 100
Inventario: 30
Meses para entrega: 12
```

Resultado aproximado:

```text
$3,903,813.61 MXN
```

---

## 9. Aplicación visual con Streamlit

El proyecto incluye una aplicación visual desarrollada con Streamlit para permitir que un usuario no técnico utilice el modelo sin interactuar directamente con los scripts de Python.

La aplicación se encuentra en:

```text
app.py
```

La interfaz utiliza directamente las funciones de inferencia definidas en:

```text
src/predict.py
```

La arquitectura actual es:

```text
Usuario
   ↓
Streamlit
   ↓
app.py
   ↓
src/predict.py
   ↓
Ridge Regression
   ↓
Precio estimado
```

Streamlit funciona únicamente como capa visual y no reemplaza el modelo estadístico.

---

## 10. Ejecutar la aplicación

Antes de iniciar la aplicación deben existir los artefactos generados por el pipeline.

Si es necesario, ejecutar primero:

```bash
dvc repro
```

Después, desde la raíz del proyecto:

```bash
python -m streamlit run app.py
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```

---

## 11. Funcionalidad actual de la aplicación

La primera versión permite:

- Seleccionar municipio.
- Seleccionar clasificación SOFTEC.
- Capturar superficie interior.
- Capturar superficie de terraza.
- Capturar número de recámaras.
- Capturar cajones de estacionamiento.
- Capturar niveles del desarrollo.
- Capturar meses en venta.
- Capturar unidades totales.
- Capturar unidades del master plan.
- Capturar inventario disponible.
- Capturar meses para entrega.
- Ejecutar una predicción.
- Mostrar el precio estimado en pesos mexicanos.
- Mostrar R², MAE y RMSE del modelo.

---

## 12. Validación de la aplicación

La aplicación fue validada utilizando los mismos valores del ejemplo de inferencia:

```text
Municipio: Zapopan
Clasificación: R
Superficie: 85 m²
Terraza: 0 m²
Recámaras: 2
Estacionamientos: 1
Niveles: 8
Meses en venta: 12
Unidades totales: 100
Unidades master plan: 100
Inventario: 30
Meses para entrega: 12
```

La aplicación genera:

```text
$3,903,814 MXN
```

La diferencia respecto a la salida de terminal:

```text
$3,903,813.61 MXN
```

se debe únicamente al redondeo visual de la aplicación.

Métricas mostradas:

```text
R² Test: 0.859
MAE: $549,477 MXN
RMSE: $724,229 MXN
```

---

## 13. Fases del proyecto

### Fase 1 — Prototipo

Se desarrolló un prototipo monolítico y se compararon diferentes algoritmos de Machine Learning.

Modelo ganador:

```text
Ridge Regression
```

### Fase 2 — Industrialización

Se reorganizó el proyecto utilizando:

- DVC.
- Pipeline reproducible.
- Scripts reutilizables.
- Configuración centralizada.
- Tracking de métricas.
- Inferencia independiente.

### Fase 3 — Aplicación visual e IA generativa

Actualmente se encuentra implementada la primera versión de la aplicación visual con Streamlit.

La siguiente etapa contempla mejorar la experiencia para usuarios no técnicos y posteriormente incorporar una capa opcional de IA generativa.

---

## 14. Próximos pasos

Las siguientes mejoras previstas son:

1. Mejorar la experiencia visual de la aplicación.
2. Simplificar y explicar los campos para usuarios no técnicos.
3. Mejorar la presentación e interpretación del precio estimado.
4. Agregar información complementaria como precio por metro cuadrado.
5. Incorporar explicaciones sencillas del resultado.
6. Incorporar opcionalmente IA generativa para interpretar consultas en lenguaje natural.
7. Convertir consultas de lenguaje natural a las variables estructuradas utilizadas por el modelo.
8. Utilizar IA generativa para explicar resultados sin sustituir la predicción estadística.

La arquitectura futura propuesta es:

```text
Consulta del usuario
        ↓
IA generativa
        ↓
Extracción de variables
        ↓
Modelo Ridge Regression
        ↓
Predicción estadística
        ↓
IA generativa
        ↓
Explicación del resultado
```

El modelo de Machine Learning continuará siendo responsable de la predicción de precio. La IA generativa funcionará únicamente como capa de interpretación e interacción.