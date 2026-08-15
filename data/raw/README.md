# Datos originales

Este directorio está reservado para el archivo de entrada utilizado por el pipeline:

```text
Guadalajara 4Q22.xlsx
```

El archivo corresponde a información DIME de Softec utilizada para el desarrollo académico del proyecto.

## Acceso

El dataset original es de acceso restringido y **no forma parte del repositorio público**.

La reproducción completa del pipeline desde los datos originales requiere que el usuario cuente previamente con acceso autorizado a dicha fuente.

Cuando se disponga del archivo, debe colocarse exactamente en:

```text
data/raw/Guadalajara 4Q22.xlsx
```

Posteriormente, desde la raíz del proyecto, puede ejecutarse:

```bash
dvc repro
```

El nombre y ubicación esperados pueden modificarse mediante la configuración definida en `params.yaml`.

> No deben versionarse ni publicarse en este repositorio copias del dataset original o de los datos derivados.