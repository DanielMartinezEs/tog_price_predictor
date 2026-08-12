# TOG Price Predictor

Proyecto de Machine Learning para la estimación de precios de departamentos en la Zona Metropolitana de Guadalajara.

El proyecto evolucionó desde un prototipo analítico hasta una solución reproducible y visual que permite a usuarios no técnicos obtener una estimación de precio mediante un modelo de Machine Learning.

La versión estable disponible en la rama `main` incluye:

* Comparación y selección de modelos.
* Modelo final Ridge Regression.
* Pipeline reproducible con DVC.
* Configuración centralizada mediante `params.yaml`.
* Scripts reutilizables para preparación, entrenamiento, evaluación e inferencia.
* Tracking reproducible de métricas.
* Aplicación visual desarrollada con Streamlit.
* Validaciones lógicas de entrada.
* Detección de valores fuera del dominio observado durante el entrenamiento.
* Presentación del precio estimado y precio por metro cuadrado.
* Captura manual de variables.
* Interpretación opcional de lenguaje natural mediante IA generativa.
* Extracción estructurada de variables.
* Revisión humana cuando la descripción proporcionada a la IA está incompleta.

La IA generativa funciona únicamente como una capa de interacción.

**La estimación del precio continúa siendo responsabilidad exclusiva del modelo Ridge Regression.**

---

## 1. Arquitectura general

El flujo principal de Machine Learning es:

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

La aplicación incorpora además una capa opcional de IA generativa:

```text
Descripción en lenguaje natural
          ↓
src/ai_layer.py
          ↓
IA generativa
          ↓
Extracción estructurada
          ↓
Formulario Streamlit
          ↓
Revisión del usuario
          ↓
Validaciones
          ↓
src/predict.py
          ↓
Ridge Regression
          ↓
Precio estimado
```

La IA generativa no sustituye al modelo estadístico y no calcula el precio del inmueble.

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
├── fase1_prototipo_monolito_abr2026.py
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│
├── models/
│   ├── modelo_final.pkl
│   ├── scaler_X.pkl
│   ├── scaler_Y.pkl
│   └── feature_names.json
│
├── reports/
│   └── metrics.json
│
└── src/
    ├── ai_layer.py
    ├── predict.py
    ├── validation.py
    │
    └── stages/
        ├── prepare_data.py
        ├── split_data.py
        ├── train.py
        └── evaluate.py
```

El repositorio incluye además los notebooks utilizados durante las distintas etapas de desarrollo y experimentación.

El entorno virtual, cachés y archivos temporales locales no forman parte de los componentes que deben versionarse.

---

## 3. Dataset privado

El dataset original no se versiona en GitHub por razones de confidencialidad.

Para fines de evaluación académica, el archivo se proporciona de manera privada junto con la entrega del proyecto y del documento de tesis.

El pipeline espera encontrarlo exactamente en:

```text
data/raw/Guadalajara 4Q22.xlsx
```

El evaluador únicamente debe colocar el archivo recibido en esa ubicación antes de ejecutar el pipeline.

Información adicional se encuentra en:

```text
data/raw/README.md
```

---

## 4. Preparar el entorno

Se recomienda utilizar un entorno virtual exclusivo para el proyecto.

Desde la raíz:

```bash
python -m venv entorno_rspp
```

Activar en Windows CMD:

```bash
entorno_rspp\Scripts\activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

La versión de `scikit-learn` está fijada en:

```text
scikit-learn==1.4.2
```

Esto mantiene compatibilidad con los artefactos serializados utilizados por el modelo.

El entorno virtual debe crearse localmente y no debe versionarse en Git.

---

## 5. Verificar el entorno

Con el entorno virtual activado:

```bash
python -c "import sys; print(sys.executable)"
```

La ruta debe apuntar al intérprete ubicado dentro de:

```text
tog_dme\entorno_rspp\Scripts\python.exe
```

Esto evita que otro intérprete instalado en el equipo, por ejemplo Anaconda, ejecute accidentalmente el proyecto.

También puede comprobarse la integridad de las dependencias mediante:

```bash
python -m pip check
```

Una instalación correcta debe indicar:

```text
No broken requirements found.
```

---

## 6. Pipeline reproducible con DVC

El pipeline se encuentra definido en:

```text
dvc.yaml
```

y contiene las etapas:

```text
prepare_data
     ↓
split_data
     ↓
train
     ↓
evaluate
```

Con el dataset privado disponible:

```bash
dvc repro
```

Para forzar la reproducción completa:

```bash
dvc repro -f
```

---

## 7. Verificar el pipeline

Para comprobar el estado:

```bash
dvc status
```

Cuando no existen cambios pendientes, el resultado esperado es:

```text
Data and pipelines are up to date.
```

Para visualizar el flujo:

```bash
dvc dag
```

---

## 8. Modelo seleccionado

Durante la Fase 1 se compararon diferentes algoritmos de Machine Learning.

El modelo seleccionado fue:

```text
Ridge Regression
```

Las principales métricas reproducidas en el conjunto de prueba son:

```text
R2_test = 0.858668
RMSE_test_pesos = 724,228.73
MAE_test_pesos = 549,476.61
```

El modelo explica aproximadamente el **85.9 % de la variabilidad del precio observada en el conjunto de prueba**.

---

## 9. Consultar métricas

DVC permite recuperar las métricas mediante:

```bash
dvc metrics show
```

Los resultados también se almacenan en:

```text
reports/metrics.json
```

Las principales métricas esperadas son aproximadamente:

```text
R2_test            0.85867
MAE_test_pesos     549476.61
RMSE_test_pesos    724228.73
```

---

## 10. Inferencia desde terminal

La lógica reutilizable de inferencia se encuentra en:

```text
src/predict.py
```

Puede probarse mediante:

```bash
python src\predict.py
```

Caso de control:

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

Resultado esperado:

```text
$3,903,813.61 MXN
```

---

## 11. Aplicación visual con Streamlit

La aplicación principal se encuentra en:

```text
app.py
```

Puede ejecutarse desde la raíz del proyecto mediante:

```bash
python -m streamlit run app.py
```

Normalmente se abrirá en:

```text
http://localhost:8501
```

Se recomienda utilizar `python -m streamlit` para garantizar que Streamlit se ejecute con el mismo intérprete del entorno virtual activo.

---

## 12. Captura manual

La aplicación permite utilizar el modelo completamente sin IA generativa.

El usuario puede capturar manualmente:

### Departamento

* Municipio.
* Clasificación SOFTEC.
* Superficie interior.
* Superficie de terraza.
* Número de recámaras.
* Cajones de estacionamiento.

### Desarrollo

* Número de niveles.
* Meses en venta.
* Unidades totales.
* Unidades del master plan.
* Inventario disponible.
* Meses para entrega.

Después de ejecutar la predicción se muestran:

* Precio estimado.
* Precio estimado por metro cuadrado interior.
* Interpretación sencilla del resultado.
* MAE como referencia.
* Resumen de los datos utilizados.
* R², MAE y RMSE en una sección técnica.

---

## 13. IA generativa

La lógica de interpretación de lenguaje natural se encuentra en:

```text
src/ai_layer.py
```

La función de esta capa es transformar una descripción inmobiliaria en las variables estructuradas que posteriormente utiliza la aplicación.

El flujo es:

```text
Texto del usuario
      ↓
IA generativa
      ↓
Variables estructuradas
      ↓
Formulario editable
      ↓
Usuario revisa
      ↓
Validaciones
      ↓
Ridge Regression
      ↓
Precio
```

La IA generativa:

* Interpreta lenguaje natural.
* Extrae variables estructuradas.
* Autocompleta los campos identificados.
* No entrena el modelo.
* No modifica los artefactos de Machine Learning.
* No calcula el precio.
* No sustituye a Ridge Regression.
* No debe inventar valores faltantes.
* Mantiene disponible la captura manual.

---

## 14. Descripciones incompletas

Ridge Regression requiere todas las variables utilizadas durante su entrenamiento.

Por ello, si el usuario escribe solamente:

```text
Tengo un departamento de 90 metros cuadrados en Zapopan,
con 3 recámaras y 2 cajones de estacionamiento.
```

la IA puede identificar:

```text
Municipio: Zapopan
Superficie interior: 90 m²
Recámaras: 3
Estacionamientos: 2
```

pero no debe inventar las variables restantes.

La aplicación identifica explícitamente cuáles no fueron encontradas.

Los campos no identificados conservan los valores existentes en el formulario, pero se informa al usuario que esos valores **no fueron inferidos por la IA**.

Antes de permitir la predicción después de una descripción incompleta, el usuario debe confirmar que revisó y completó manualmente esos campos.

```text
Prompt parcial
      ↓
IA extrae variables disponibles
      ↓
Identificación de campos faltantes
      ↓
Usuario revisa/completa
      ↓
Confirmación explícita
      ↓
Validaciones
      ↓
Ridge Regression
      ↓
Predicción
```

---

## 15. Validaciones de entrada

La aplicación incorpora validaciones deterministas antes de ejecutar el modelo.

Por ejemplo:

```text
Inventario disponible > Unidades totales
```

es considerado inconsistente y bloquea la predicción.

También existe un límite explícito de superficie configurado en:

```text
params.yaml
```

con:

```yaml
sqm_max: 250.0
```

Una superficie superior a ese límite no es aceptada.

---

## 16. Control de extrapolación

La lógica se encuentra en:

```text
src/validation.py
```

La aplicación compara las entradas numéricas con los rangos observados durante el entrenamiento.

Los valores del conjunto de entrenamiento son llevados nuevamente a su escala original mediante el mismo `scaler_X` utilizado por el modelo.

Esto permite distinguir:

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

Una extrapolación no implica automáticamente que el resultado sea incorrecto, pero requiere una interpretación con mayor precaución.

---

## 17. Configuración opcional de la API

**La API generativa no es necesaria para utilizar la captura manual ni para ejecutar Ridge Regression.**

Para utilizar la opción de lenguaje natural es necesario configurar una API key válida.

En Windows puede almacenarse como variable de entorno:

```bash
setx OPENAI_API_KEY "TU_API_KEY"
```

Después debe cerrarse la terminal y abrirse nuevamente.

Puede verificarse la existencia de la variable sin mostrar el secreto mediante:

```bash
python -c "import os; print('OPENAI_API_KEY configurada' if os.getenv('OPENAI_API_KEY') else 'OPENAI_API_KEY no encontrada')"
```

La API key:

* No debe almacenarse en el código.
* No debe incluirse en `params.yaml`.
* No debe subirse a GitHub.
* No debe compartirse junto con la entrega académica.

El uso de la API depende de un servicio externo y puede implicar costos de consumo.

Si la API no está disponible, el formulario manual continúa siendo utilizable.

---

## 18. Validación integral

La integración completa se validó con la siguiente descripción:

```text
Departamento en Zapopan, clasificación SOFTEC R,
con 85 metros cuadrados interiores, sin terraza,
2 recámaras y 1 cajón de estacionamiento.

El desarrollo tiene 8 niveles, lleva 12 meses en venta,
tiene 100 unidades totales, 100 unidades en el master plan,
inventario disponible de 30 unidades y faltan 12 meses
para la entrega.
```

La IA identificó correctamente las variables requeridas y autocompletó el formulario.

La aplicación generó:

```text
Precio estimado:
$3,903,814 MXN
```

y:

```text
Precio estimado por m² interior:
$45,927 MXN/m²
```

La salida equivalente desde terminal es:

```text
$3,903,813.61 MXN
```

La diferencia se debe únicamente al redondeo visual.

Esto comprueba que la incorporación de la IA generativa no modifica la inferencia realizada por Ridge Regression.

---

## 19. Fases del proyecto

### Fase 1 — Prototipo analítico

Se desarrolló un prototipo monolítico para explorar y preparar los datos, entrenar distintos algoritmos, comparar su desempeño y seleccionar el modelo final.

Modelo seleccionado:

```text
Ridge Regression
```

### Fase 2 — Industrialización

El prototipo se transformó en un proyecto reproducible mediante:

* DVC.
* Pipeline por etapas.
* Configuración centralizada.
* Scripts reutilizables.
* Separación entre entrenamiento e inferencia.
* Tracking de métricas.
* Control de artefactos.

### Fase 3 — Aplicación visual e IA generativa

Se desarrolló una aplicación Streamlit que incorpora:

* Captura manual.
* Inferencia mediante Ridge Regression.
* Precio estimado y precio por metro cuadrado.
* Métricas.
* Explicaciones sencillas.
* Validaciones lógicas.
* Límites de dominio.
* Advertencias de extrapolación.
* Rangos observados durante entrenamiento.
* Interpretación opcional de lenguaje natural.
* Extracción estructurada mediante IA generativa.
* Autocompletado del formulario.
* Identificación explícita de variables faltantes.
* Revisión humana antes de predecir cuando la descripción es incompleta.
* Manejo de errores de API sin inutilizar la captura manual.

---

## 20. Responsabilidad de cada componente

```text
IA generativa
→ interpreta lenguaje natural
→ extrae variables

Streamlit
→ captura y presenta información
→ permite revisión humana

src/validation.py
→ valida consistencia
→ detecta extrapolaciones

Ridge Regression
→ calcula el precio
```

Esta separación evita utilizar un modelo generativo como sustituto del modelo estadístico entrenado específicamente para la estimación de precios.

---

## 21. Versión estable

La rama recomendada para evaluación es:

```text
main
```

La versión estable incluye:

* Pipeline reproducible con DVC.
* Ridge Regression.
* Inferencia desde terminal.
* Aplicación Streamlit.
* Captura manual.
* Validaciones de entrada.
* Control de extrapolación.
* IA generativa opcional para lenguaje natural.

La predicción mediante captura manual no requiere acceso a la API generativa.

---

## 22. Limitaciones

La herramienta debe interpretarse como un sistema de estimación y no como una valuación inmobiliaria formal.

Entre sus principales limitaciones se encuentran:

* Dependencia de la representatividad del dataset disponible.
* Posible pérdida de precisión en casos alejados del dominio de entrenamiento.
* Dependencia de las variables disponibles en la fuente original.
* Ausencia de otras variables externas que podrían influir en el precio.
* El MAE representa un error promedio y no un intervalo de confianza individual.
* Ridge Regression requiere todas las variables utilizadas durante su entrenamiento.
* Una descripción incompleta mediante IA requiere intervención del usuario.
* La interpretación mediante IA depende de un servicio externo.
* El uso de una API generativa puede implicar costo, conectividad y disponibilidad.
* La salida generativa debe revisarse antes de utilizarse como entrada estadística.

---

## 23. Entrega académica

El repositorio de GitHub contiene el código, configuración y componentes necesarios para reproducir el proyecto.

El dataset original no se publica en el repositorio.

Para la evaluación académica, el archivo:

```text
Guadalajara 4Q22.xlsx
```

se proporciona de manera privada junto con la entrega de la tesis.

El evaluador debe colocarlo en:

```text
data/raw/Guadalajara 4Q22.xlsx
```

antes de ejecutar `dvc repro`.

La API key utilizada durante el desarrollo **no forma parte de la entrega**.

---

## 24. Trabajo futuro

Entre las posibles extensiones se encuentran:

1. Incorporar nuevas fuentes de datos.
2. Actualizar periódicamente el dataset.
3. Evaluar modelos adicionales.
4. Incorporar nuevas variables inmobiliarias y geográficas.
5. Agregar mecanismos más avanzados de explicabilidad.
6. Publicar la aplicación en un entorno web.
7. Incorporar administración segura de secretos para despliegue.
8. Evaluar diferentes modelos generativos.
9. Incorporar explicaciones generativas posteriores a la predicción.
10. Implementar monitoreo de desempeño y estrategias de reentrenamiento.
11. Evaluar formalmente precisión, costo y latencia de la capa generativa.

---

## 25. Ejecución rápida para evaluación

Con el dataset privado disponible:

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

Ejecutar la aplicación:

```bash
python -m streamlit run app.py
```

La interpretación mediante IA requiere adicionalmente una `OPENAI_API_KEY` válida.

---

## 26. Resultado final

TOG Price Predictor integra:

```text
Pipeline reproducible
        +
Ridge Regression
        +
Aplicación Streamlit
        +
IA generativa opcional
```

El modelo estadístico conserva la responsabilidad exclusiva de estimar el precio.

La IA generativa facilita la interacción en lenguaje natural, manteniendo revisión humana y validaciones deterministas antes de ejecutar la inferencia.
