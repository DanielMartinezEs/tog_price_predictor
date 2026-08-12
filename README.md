# TOG Price Predictor

Proyecto de Machine Learning para la estimación de precios de departamentos en la Zona Metropolitana de Guadalajara.

El proyecto evolucionó desde un prototipo analítico hasta una solución reproducible y visual que permite a usuarios no técnicos capturar las características de un departamento y obtener una estimación de precio mediante un modelo de Machine Learning.

La versión estable disponible en la rama `main` incluye:

* Prototipo y comparación de modelos.
* Modelo final Ridge Regression.
* Pipeline reproducible con DVC.
* Configuración centralizada mediante `params.yaml`.
* Scripts reutilizables para preparación, entrenamiento, evaluación e inferencia.
* Tracking de métricas.
* Aplicación visual desarrollada con Streamlit.
* Validaciones de entrada.
* Detección de valores fuera del dominio observado durante entrenamiento.
* Presentación del precio estimado y precio por metro cuadrado.
* Explicación sencilla de resultados para usuarios no técnicos.

---

## 1. Arquitectura general

```text
Datos privados
     ↓
Preparación
     ↓
Train / Test Split
     ↓
Ridge Regression
     ↓
Evaluación
     ↓
Artefactos entrenados
     ↓
src/predict.py
     ↓
Streamlit
     ↓
Usuario
```

La aplicación visual funciona únicamente como capa de interacción.

La predicción del precio continúa siendo responsabilidad del modelo estadístico entrenado.

---

## 2. Estructura principal

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
    ├── validation.py
    │
    └── stages/
        ├── prepare_data.py
        ├── split_data.py
        ├── train.py
        └── evaluate.py
```

El archivo original de datos no forma parte del repositorio debido a restricciones de confidencialidad.

---

## 3. Dataset privado

El pipeline espera encontrar el archivo:

```text
data/raw/Guadalajara 4Q22.xlsx
```

Este archivo no se versiona en GitHub.

Para reproducir el proyecto:

1. Solicitar el archivo al autor.
2. Colocarlo exactamente en:

```text
data/raw/Guadalajara 4Q22.xlsx
```

3. Consultar instrucciones adicionales en:

```text
data/raw/README.md
```

---

## 4. Preparar el entorno

Se recomienda utilizar un entorno virtual exclusivo para el proyecto.

Desde la raíz del repositorio:

```bash
python -m venv entorno_rspp
```

Activar en Windows CMD:

```bash
entorno_rspp\Scripts\activate
```

Después instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

La versión de `scikit-learn` está fijada en:

```text
scikit-learn==1.4.2
```

Esto permite mantener compatibilidad con los artefactos serializados del modelo.

El entorno virtual debe crearse localmente y no debe versionarse en Git.

---

## 5. Verificar el entorno

Con el entorno virtual activado puede comprobarse el intérprete utilizado con:

```bash
python -c "import sys; print(sys.executable)"
```

La ruta debe apuntar al Python ubicado dentro de:

```text
tog_dme\entorno_rspp\Scripts\python.exe
```

Para comprobar la integridad de las dependencias:

```bash
python -m pip check
```

Una instalación correcta debe indicar:

```text
No broken requirements found.
```

---

## 6. Pipeline reproducible con DVC

El pipeline está definido en:

```text
dvc.yaml
```

y contiene las siguientes etapas:

```text
prepare_data
     ↓
split_data
     ↓
train
     ↓
evaluate
```

Con el dataset privado disponible, ejecutar:

```bash
dvc repro
```

Para forzar la ejecución completa:

```bash
dvc repro -f
```

---

## 7. Revisar estado del pipeline

Para comprobar si el pipeline está actualizado:

```bash
dvc status
```

El resultado esperado cuando no existen cambios es:

```text
Data and pipelines are up to date.
```

Para visualizar el flujo:

```bash
dvc dag
```

---

## 8. Modelo seleccionado

Durante la etapa de prototipado se compararon distintos algoritmos de Machine Learning.

El modelo seleccionado fue:

```text
Ridge Regression
```

Las métricas reproducidas en el conjunto de prueba son aproximadamente:

```text
R2_test = 0.858668
RMSE_test_pesos = 724,228.73
MAE_test_pesos = 549,476.61
```

Por lo tanto, el modelo explica aproximadamente el 85.9 % de la variabilidad del precio observada en el conjunto de prueba.

---

## 9. Consultar métricas con DVC

Ejecutar:

```bash
dvc metrics show
```

Salida esperada aproximada:

```text
R2_test             0.85867
R2_train            0.87228
MAE_test_pesos      549476.61
MAE_train_pesos     598614.59
RMSE_test_pesos     724228.73
RMSE_train_pesos    820764.76
```

Las métricas también se encuentran almacenadas en:

```text
reports/metrics.json
```

---

## 10. Inferencia desde terminal

La lógica de inferencia está centralizada en:

```text
src/predict.py
```

Después de contar con los artefactos entrenados, ejecutar:

```bash
python src\predict.py
```

El script:

1. Carga el modelo Ridge entrenado.
2. Carga los scalers utilizados durante entrenamiento.
3. Recupera los nombres de las variables esperadas.
4. Construye el registro de entrada.
5. Aplica las transformaciones correspondientes.
6. Ejecuta la predicción.
7. Convierte el resultado nuevamente a pesos mexicanos.

Ejemplo de control:

```text
Municipio: Zapopan
Clasificación: R
Superficie interior: 85 m²
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

Resultado:

```text
$3,903,813.61 MXN
```

---

## 11. Aplicación visual con Streamlit

La aplicación se encuentra en:

```text
app.py
```

Su arquitectura es:

```text
Usuario
   ↓
Streamlit
   ↓
app.py
   ↓
Validaciones
   ↓
src/predict.py
   ↓
Ridge Regression
   ↓
Precio estimado
```

La interfaz no implementa un segundo modelo ni una lógica independiente de predicción.

Utiliza directamente la misma lógica de inferencia de `src/predict.py`.

---

## 12. Ejecutar la aplicación

Desde la raíz del proyecto, con el entorno virtual activado:

```bash
python -m streamlit run app.py
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```

Se recomienda utilizar `python -m streamlit` para garantizar que Streamlit se ejecute con el mismo intérprete del entorno virtual activo.

---

## 13. Funcionalidades de la aplicación

La versión estable permite capturar:

### Características del departamento

* Municipio.
* Clasificación SOFTEC.
* Superficie interior.
* Superficie de terraza.
* Número de recámaras.
* Cajones de estacionamiento.

### Características del desarrollo

* Número de niveles.
* Meses en venta.
* Unidades totales.
* Unidades del master plan.
* Inventario disponible.
* Meses para entrega.

Después de ejecutar la predicción, la aplicación muestra:

* Precio estimado.
* Precio estimado por metro cuadrado interior.
* Interpretación sencilla del resultado.
* Error absoluto promedio del modelo como referencia.
* Resumen de los datos utilizados.
* R², MAE y RMSE en una sección técnica desplegable.

---

## 14. Validaciones de entrada

La aplicación incorpora validaciones antes de enviar información al modelo.

Por ejemplo:

```text
Inventario disponible > Unidades totales
```

es considerado un dato inconsistente y bloquea la predicción.

También existe un dominio explícito para superficie definido en:

```text
params.yaml
```

con:

```yaml
sqm_max: 250.0
```

Una superficie superior a ese valor no es aceptada por la aplicación.

---

## 15. Control de extrapolación

Además de las validaciones lógicas, la aplicación compara las entradas numéricas contra los rangos observados en el conjunto utilizado para entrenar el modelo.

La lógica se encuentra en:

```text
src/validation.py
```

Los valores del conjunto de entrenamiento son llevados nuevamente a su escala original mediante el mismo `scaler_X` utilizado por el modelo.

Esto permite distinguir entre:

```text
Entrada dentro del dominio
        ↓
Predicción normal
```

```text
Entrada inconsistente
        ↓
Predicción bloqueada
```

```text
Entrada válida pero fuera del rango observado
        ↓
Advertencia de extrapolación
        ↓
Predicción permitida
```

Una extrapolación no implica automáticamente que la predicción sea incorrecta, pero indica que el resultado debe interpretarse con mayor precaución.

---

## 16. Validación de la aplicación

Se utilizó como caso de control el mismo ejemplo de inferencia ejecutado desde terminal:

```text
Municipio: Zapopan
Clasificación: R
Superficie interior: 85 m²
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
Precio estimado:
$3,903,814 MXN
```

y:

```text
Precio estimado por m² interior:
$45,927 MXN/m²
```

La diferencia respecto a la salida de terminal:

```text
$3,903,813.61 MXN
```

se debe únicamente al redondeo utilizado para presentar el resultado en la interfaz.

---

## 17. Fases del proyecto

### Fase 1 — Prototipo analítico

Se desarrolló un prototipo monolítico para:

* Explorar los datos.
* Preparar variables.
* Entrenar diferentes algoritmos.
* Comparar su desempeño.
* Seleccionar el modelo final.

El algoritmo seleccionado fue Ridge Regression.

---

### Fase 2 — Industrialización

El prototipo fue transformado en un proyecto reproducible utilizando:

* DVC.
* Pipeline por etapas.
* Configuración centralizada.
* Scripts reutilizables.
* Separación entre entrenamiento e inferencia.
* Tracking de métricas.
* Control de artefactos.

El pipeline final es:

```text
prepare_data
→ split_data
→ train
→ evaluate
```

---

### Fase 3 — Aplicación visual

Se desarrolló una capa visual con Streamlit para facilitar el consumo del modelo por usuarios no técnicos.

La fase incorporó:

* Formulario de captura.
* Integración directa con la inferencia.
* Precio estimado.
* Precio estimado por metro cuadrado.
* Presentación de métricas.
* Explicación sencilla del resultado.
* Ayudas de captura.
* Validaciones lógicas.
* Límites de dominio.
* Advertencias de extrapolación.
* Visualización de rangos observados durante entrenamiento.

La aplicación fue validada utilizando el mismo caso de control empleado por `src/predict.py`.

---

## 18. IA generativa como extensión opcional

Durante la Fase 3 se exploró una arquitectura opcional para incorporar IA generativa.

El objetivo conceptual es permitir una interacción como:

```text
Usuario escribe una descripción
        ↓
LLM extrae variables estructuradas
        ↓
Usuario revisa o completa los datos
        ↓
Ridge Regression calcula el precio
        ↓
LLM puede explicar el resultado
```

Una regla fundamental de esta arquitectura es:

> El modelo generativo no sustituye al modelo estadístico encargado de estimar el precio.

La predicción continuaría siendo responsabilidad de Ridge Regression.

La integración experimental de IA se mantiene separada de la versión estable debido a que requiere credenciales externas y consumo de un servicio de API.

El prototipo se encuentra en la rama:

```text
feature/optional-generative-ai
```

y no forma parte de los requisitos necesarios para ejecutar la aplicación estable disponible en:

```text
main
```

---

## 19. Versión estable

La rama recomendada para evaluación y reproducción del proyecto es:

```text
main
```

Esta versión:

* No requiere una API de IA generativa.
* No requiere credenciales externas.
* Ejecuta directamente el modelo Ridge.
* Es reproducible mediante `requirements.txt`.
* Utiliza DVC para reproducir el pipeline.
* Incluye la aplicación Streamlit.
* Incluye validaciones de entrada.
* Incluye control de extrapolación.

---

## 20. Limitaciones

La herramienta debe interpretarse como un sistema de estimación y no como una valuación inmobiliaria formal.

Entre sus principales limitaciones se encuentran:

* Dependencia de la representatividad del dataset disponible.
* Posible pérdida de precisión en casos alejados del dominio de entrenamiento.
* Dependencia de las variables incluidas en el dataset original.
* Ausencia de variables externas que podrían influir en el precio.
* El MAE representa un error promedio de evaluación y no constituye un intervalo de confianza para una predicción individual.

---

## 21. Trabajo futuro

Entre las posibles extensiones se encuentran:

1. Incorporar nuevas fuentes de datos.
2. Actualizar periódicamente el dataset.
3. Evaluar modelos adicionales.
4. Incorporar nuevas variables inmobiliarias y geográficas.
5. Agregar mecanismos más avanzados de explicabilidad.
6. Publicar la aplicación en un entorno accesible vía web.
7. Retomar la integración opcional de IA generativa.
8. Interpretar consultas inmobiliarias en lenguaje natural.
9. Generar explicaciones personalizadas manteniendo al modelo estadístico como responsable de la predicción.
10. Evaluar estrategias de monitoreo y reentrenamiento del modelo.

---

## 22. Resumen de ejecución

Para un evaluador que ya cuenta con el archivo privado:

```bash
python -m venv entorno_rspp
```

```bash
entorno_rspp\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

Colocar:

```text
data/raw/Guadalajara 4Q22.xlsx
```

Ejecutar:

```bash
dvc repro
```

Verificar:

```bash
dvc status
```

Consultar métricas:

```bash
dvc metrics show
```

Probar inferencia:

```bash
python src\predict.py
```

Ejecutar aplicación:

```bash
python -m streamlit run app.py
```

Con estos pasos es posible reproducir el pipeline y utilizar la versión estable de TOG Price Predictor.
