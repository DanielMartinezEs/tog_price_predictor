# Raw Data

Este directorio debe contener el archivo original privado utilizado por el pipeline:

Guadalajara 4Q22.xlsx

Por confidencialidad, este archivo no se versiona en GitHub.

## Instrucciones para tutores/evaluadores

1. Solicitar acceso al archivo privado al autor del proyecto.
2. Descargar el archivo `Guadalajara 4Q22.xlsx`.
3. Colocarlo exactamente en esta ruta:

data/raw/Guadalajara 4Q22.xlsx

4. Ejecutar el pipeline desde la raíz del proyecto:

dvc repro

## Nota

El pipeline espera que el nombre del archivo sea exactamente:

Guadalajara 4Q22.xlsx

Si se desea usar otro archivo, modificar el parámetro correspondiente en:

params.yaml