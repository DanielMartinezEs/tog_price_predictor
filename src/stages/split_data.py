# SPLIT DATA STAGE
# Lee data preparada, divide train/test, escala variables y guarda datasets procesados.

from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

PARAMS_PATH = "params.yaml"

with open(PARAMS_PATH, "r", encoding="utf-8") as conf_file:
    config = yaml.safe_load(conf_file)

np.random.seed(config["base"]["numpy_seed"])


# ============================================================
# LOAD PREPARED DATA
# ============================================================

prepared_data_path = config["prepare"]["prepared_data_path"]

data = pd.read_csv(prepared_data_path)

print(f"Prepared data loaded from: {prepared_data_path}")
print(f"Prepared data shape: {data.shape}")


# ============================================================
# X / Y
# ============================================================

X_df = data.drop(["price"], axis="columns")
y_df = data["price"]

feature_names = X_df.columns.tolist()

X = X_df.values
Y = y_df.values.reshape(-1, 1)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    Y,
    test_size=config["split"]["test_size"],
    random_state=config["split"]["random_state"]
)


# ============================================================
# SCALING WITHOUT DATA LEAKAGE
# ============================================================

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

xtrain = scaler_X.fit_transform(xtrain)
xtest = scaler_X.transform(xtest)

ytrain = scaler_Y.fit_transform(ytrain).ravel()
ytest = scaler_Y.transform(ytest).ravel()


# ============================================================
# SAVE TRAIN / TEST DATASETS
# ============================================================

train_data_path = config["split"]["train_data_path"]
test_data_path = config["split"]["test_data_path"]

Path(train_data_path).parent.mkdir(parents=True, exist_ok=True)
Path(test_data_path).parent.mkdir(parents=True, exist_ok=True)

train_df = pd.DataFrame(xtrain, columns=feature_names)
train_df["price"] = ytrain

test_df = pd.DataFrame(xtest, columns=feature_names)
test_df["price"] = ytest

train_df.to_csv(train_data_path, index=False)
test_df.to_csv(test_data_path, index=False)

print(f"Train data saved to: {train_data_path}")
print(f"Test data saved to: {test_data_path}")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# SAVE SCALERS AND FEATURE NAMES
# ============================================================

scaler_x_path = config["artifacts"]["scaler_x_path"]
scaler_y_path = config["artifacts"]["scaler_y_path"]
feature_names_path = config["artifacts"]["feature_names_path"]

Path(scaler_x_path).parent.mkdir(parents=True, exist_ok=True)
Path(scaler_y_path).parent.mkdir(parents=True, exist_ok=True)
Path(feature_names_path).parent.mkdir(parents=True, exist_ok=True)

joblib.dump(scaler_X, scaler_x_path)
joblib.dump(scaler_Y, scaler_y_path)

with open(feature_names_path, "w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=4)

print(f"Scaler X saved to: {scaler_x_path}")
print(f"Scaler Y saved to: {scaler_y_path}")
print(f"Feature names saved to: {feature_names_path}")