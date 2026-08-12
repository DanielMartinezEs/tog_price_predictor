# TOG Price Predictor

Proyecto de Machine Learning para estimación de precios de departamentos en la Zona Metropolitana de Guadalajara.

El proyecto incluye:

- Prototipo analítico de Fase 1.
- Pipeline reproducible con DVC.
- Configuración centralizada en `params.yaml`.
- Scripts reutilizables para preparación de datos, split, entrenamiento, evaluación e inferencia.
- Métricas reproducibles del modelo ganador.

---

## 1. Estructura principal

```text
data/
  raw/
  processed/

models/

reports/

src/
  stages/
    prepare_data.py
    split_data.py
    train.py
    evaluate.py
  predict.py

params.yaml
dvc.yaml
dvc.lock
requirements.txt
README.md
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

Con el archivo raw colocado en `data/raw/`, ejecutar:

```bash
dvc repro
```

Para forzar la reproducción completa:

```bash
dvc repro -f
```

---

## 5. Revisar métricas

```bash
dvc metrics show
```

Métricas esperadas:

```text
R2_test ≈ 0.85867
RMSE_test_pesos ≈ 724,228.73
MAE_test_pesos ≈ 549,476.61
```

---

## 6. Ejecutar predicción individual

Después de correr el pipeline:

```bash
python src\predict.py
```

Este script carga los artefactos entrenados y genera una predicción de ejemplo.

---

## 7. Pipeline DVC

El pipeline está definido en:

```text
dvc.yaml
```

Stages:

```text
prepare_data
split_data
train
evaluate
```

Flujo:

```text
raw data
→ prepared data
→ train/test split
→ model training
→ evaluation metrics
```

---

## 8. Modelo actual

Modelo ganador:

```text
Ridge Regression
```

Métricas principales:

```text
R2_test = 0.858668
RMSE_test_pesos = 724,228.73
MAE_test_pesos = 549,476.61
```

---

## 9. Próxima fase

La siguiente fase consiste en construir una interfaz visual para usuarios no técnicos, probablemente con Streamlit, y agregar una capa opcional de IA generativa para explicar resultados e interpretar consultas en lenguaje natural.