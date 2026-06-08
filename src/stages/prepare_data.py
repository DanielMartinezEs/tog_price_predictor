# PREPARE DATA STAGE
# Lee datos crudos, aplica limpieza/feature engineering y guarda data preparada.

from pathlib import Path
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import yaml


# ============================================================
# CONFIG
# ============================================================

PARAMS_PATH = "params.yaml"

with open(PARAMS_PATH, "r", encoding="utf-8") as conf_file:
    config = yaml.safe_load(conf_file)

np.random.seed(config["base"]["numpy_seed"])


# ============================================================
# LOAD RAW DATA
# ============================================================

raw_data_path = config["data_load"]["raw_data_path"]

data_ori = pd.read_excel(raw_data_path, header=0)


# ============================================================
# REESTRUCTURA BASE ORIGINAL A/B/C
# ============================================================

df = data_ori.copy()

df_1 = df.iloc[:, 0:8]
df_2 = df.iloc[:, 44:57]
dfA_p = df.iloc[:, 8:20]
dfB_p = df.iloc[:, 20:32]
dfC_p = df.iloc[:, 32:44]

dfA = pd.concat([df_1, dfA_p, df_2], axis=1).rename(columns={
    'Área A': 'sqm',
    'Terraza A': 'terrace',
    'Terreno A': 'terrain',
    'Recámara A': 'bhk',
    'Baños A': 'baths',
    'Alcoba A': 'alcoba',
    'Cuarto de Servicio A': 'room_serv',
    'Cajón A': 'park_u',
    'Niveles A': 'levels',
    'Precio Actual A': 'price',
    'Precio Inicial A': 'first_price',
    'ValorM2 A': 'price_per_sqm'
}).dropna()

dfB = pd.concat([df_1, dfB_p, df_2], axis=1).rename(columns={
    'Área B': 'sqm',
    'Terraza B': 'terrace',
    'Terreno B': 'terrain',
    'Recámara B': 'bhk',
    'Baños B': 'baths',
    'Alcoba B': 'alcoba',
    'Cuarto de Servicio B': 'room_serv',
    'Cajón B': 'park_u',
    'Niveles B': 'levels',
    'Precio Actual B': 'price',
    'Precio Inicial B': 'first_price',
    'ValorM2 B': 'price_per_sqm'
}).dropna()

dfC = pd.concat([df_1, dfC_p, df_2], axis=1).rename(columns={
    'Área C': 'sqm',
    'Terraza C': 'terrace',
    'Terreno C': 'terrain',
    'Recámara C': 'bhk',
    'Baños C': 'baths',
    'Alcoba C': 'alcoba',
    'Cuarto de Servicio C': 'room_serv',
    'Cajón C': 'park_u',
    'Niveles C': 'levels',
    'Precio Actual C': 'price',
    'Precio Inicial C': 'first_price',
    'ValorM2 C': 'price_per_sqm'
}).dropna()

data = pd.concat([dfA, dfB, dfC]).sort_values('ID', ascending=True)


# ============================================================
# RENOMBRADO GENERAL
# ============================================================

data = data.rename(columns={
    'ID': 'id',
    'Clasificación': 'classification',
    'Tipo': 'type',
    'Nombre': 'project_name',
    'Dirección': 'address',
    'Colonia': 'colony',
    'Municipio': 'town',
    'Promotor': 'promoter',
    'Absorción': 'absortion',
    'Meses en venta': 'months_in_sale',
    'Meses de inventario': 'inventory_months',
    'Éxito Comercial': 'comm_succ',
    'Unidades Totales': 'total_units',
    'Unidades proyectadas': 'master_plan_units',
    'Inventario': 'inventory',
    'Vendidas': 'sold_units',
    'Fecha de Alta': 'entry_date',
    'Fecha de Inicio': 'initial_date',
    'Fecha de Actualización': 'update_date',
    'Fecha de Entrega': 'delivery_date',
    'Financiamiento 1a opción': 'fin_op'
}).dropna()


# ============================================================
# CLEANING: COLONY
# ============================================================

data["colony"] = data["colony"].astype(str).str.lower()

colony_patterns = [
    (data["colony"].str.contains("royal co", case=False, regex=False), "royal country"),
    (data["colony"].str.contains("videncia", case=False, regex=False), "providencia"),
    (data["colony"].str.contains("ladrón de guevar", case=False, regex=False), "ladrón de guevara"),
    (data["colony"].str.contains("Villa Bosque (Villa Panamericana)", case=False, regex=False), "villa bosque"),
    (data["colony"].str.contains("americana", case=False, regex=False), "americana"),
    (data["colony"].str.contains("lafayette", case=False, regex=False), "americana"),
    (data["colony"].str.contains("juan ocot", case=False, regex=False), "san juan de ocotán"),
    (data["colony"].str.contains("santa ana tepet", case=False, regex=False), "santa ana tepetitlán"),
]

criteria, values = zip(*colony_patterns)
data["colony"] = np.select(criteria, values, default=data["colony"])


# ============================================================
# CLEANING: TYPE
# ============================================================

data["type"] = data["type"].astype(str).str.lower()

type_patterns = [
    (data["type"].str.contains("depto", case=False, regex=False), "depto"),
    (data["type"].str.contains("loft", case=False, regex=False), "depto"),
    (data["type"].str.contains("cd", case=False, regex=False), "casa"),
    (data["type"].str.contains("cs", case=False, regex=False), "casa"),
    (data["type"].str.contains("ch", case=False, regex=False), "casa"),
    (data["type"].str.contains("town house", case=False, regex=False), "casa"),
]

criteria, values = zip(*type_patterns)
data["type"] = np.select(criteria, values, default=data["type"])


# ============================================================
# DROP VARIABLES MUERTAS
# ============================================================

data = data.drop(
    config["prepare"]["dead_columns"],
    axis="columns",
    errors="ignore"
)


# ============================================================
# FEATURE: MONTHS TO DELIVERY
# ============================================================

base_date = pd.to_datetime(config["prepare"]["delivery_base_date"])

data["delivery_date"] = pd.to_datetime(data["delivery_date"])

def months_diff(fecha):
    rd = relativedelta(fecha, base_date)
    return rd.years * 12 + rd.months + rd.days / 30.44

data["months_to_delivery"] = data["delivery_date"].apply(months_diff)
data.loc[data["months_to_delivery"] < 0, "months_to_delivery"] = 0
data = data.drop(["delivery_date"], axis="columns")


# ============================================================
# FEATURE: CLASSIFICATION MAPPING
# ============================================================

data["classification"] = data["classification"].map(
    config["prepare"]["classification_mapping"]
)


# ============================================================
# FILTROS FINALES
# ============================================================

data = data[data["type"] == "depto"].copy()

data = data.drop(["terrain", "type"], axis="columns", errors="ignore")

data = data[~data["town"].isin(config["prepare"]["towns_to_drop"])]

data = data[
    (data["price_per_sqm"] < config["prepare"]["price_per_sqm_max"]) &
    (data["sqm"] < config["prepare"]["sqm_max"])
]


# ============================================================
# DUMMIES Y DATASET FINAL
# ============================================================

dummies = pd.get_dummies(data["town"])
data = pd.concat([data, dummies], axis="columns")

data = data.drop(
    config["prepare"]["final_drop_columns"],
    axis="columns",
    errors="ignore"
)


# ============================================================
# SAVE PREPARED DATA
# ============================================================

processed_data_path = config["prepare"]["prepared_data_path"]

Path(processed_data_path).parent.mkdir(parents=True, exist_ok=True)

data.to_csv(processed_data_path, index=False)

print(f"Prepared data saved to: {processed_data_path}")
print(f"Prepared data shape: {data.shape}")