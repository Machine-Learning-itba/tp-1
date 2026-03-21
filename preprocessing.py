"""
Modulo compartido de carga y preprocesamiento del dataset Wine Quality.
Replica exactamente el pipeline del notebook (sin escalado, que va dentro del CV).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
RANDOM_STATE = 42
N_SPLITS = 5


def load_data():
    """
    Retorna X_train, X_test, y_train, y_test (sin escalar).
    El escalado se hace dentro del Pipeline de cada script para evitar data leakage en CV.
    """
    red = pd.read_csv(os.path.join(DATA_DIR, "winequality-red.csv"), sep=";")
    white = pd.read_csv(os.path.join(DATA_DIR, "winequality-white.csv"), sep=";")

    red["red_wine_type"] = 1
    white["red_wine_type"] = 0
    df = pd.concat([red, white], ignore_index=True)

    # Eliminar filas con valores erroneos (criterios fijos del dominio)
    erroneous = (
        (df["total sulfur dioxide"] > 300) |
        (df["free sulfur dioxide"] > 150) |
        (df["chlorides"] > 0.3) |
        (df["citric acid"] > 1.0) |
        (df["density"] > 1.01) |
        (df["pH"] < 2.80)
    )
    df = df[~erroneous].copy()

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Winsorizacion ajustada solo sobre train
    cols_to_winsorize = ["residual sugar", "volatile acidity", "sulphates", "chlorides"]
    for col in cols_to_winsorize:
        p1 = X_train[col].quantile(0.01)
        p99 = X_train[col].quantile(0.99)
        X_train[col] = X_train[col].clip(lower=p1, upper=p99)
        X_test[col] = X_test[col].clip(lower=p1, upper=p99)

    return X_train, X_test, y_train, y_test


def get_cv():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
