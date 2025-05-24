# src/pipeline.py
import json
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Clases: OutlierTransformer, CategoryGrouper
# Funciones: build_column_transformer,
# build_pipeline, make_pipeline_for_experiment,
# make_json_serializable

# --------- Wrappers Custom (sklearn friendly, tunables) ---------


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """
    Detecta y trata outliers en columnas numéricas.
    Estrategias soportadas: "clip" (winsorize), "none".
    Parámetros tunables:
        - strategy: {"clip", "none"}
        - pmin: percentil mínimo (ej: 0.01)
        - pmax: percentil máximo (ej: 0.99)
    """

    def __init__(self, strategy="clip", pmin=0.01, pmax=0.99):
        self.strategy = strategy
        self.pmin = pmin
        self.pmax = pmax

    def fit(self, X, y=None):
        if self.strategy == "clip":
            # Guarda límites por columna
            self.limits_ = {
                col: (X[col].quantile(self.pmin), X[col].quantile(self.pmax))
                for col in X.columns
            }
        return self

    def transform(self, X):
        X_out = X.copy()
        if self.strategy == "clip" and hasattr(self, "limits_"):
            for col, (minv, maxv) in self.limits_.items():
                X_out[col] = X_out[col].clip(minv, maxv)
        return X_out


class CategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa categorías poco frecuentes en un valor 'Other' según min_freq.
    Pensado para trabajar con una sola columna (por pipeline/ColumnTransformer).
    Parámetros tunables:
        - min_freq: mínima frecuencia absoluta para NO agrupar
        - other_label: etiqueta para categorías agrupadas
    """

    def __init__(self, col, min_freq=10, other_label="Other"):
        self.col = col
        self.min_freq = min_freq
        self.other_label = other_label

    def fit(self, X, y=None):
        # X puede ser DataFrame (N,1) o array (N,1)
        if hasattr(X, "iloc"):
            vals = X.iloc[:, 0].value_counts()
        else:
            import numpy as np

            vals = pd.Series(X[:, 0]).value_counts()
        self.major_cats_ = vals[vals >= self.min_freq].index.tolist()
        return self

    def transform(self, X):
        # Aplica el agrupamiento solo a la columna correspondiente
        if hasattr(X, "iloc"):
            X_ = X.copy()
            X_.iloc[:, 0] = X_.iloc[:, 0].apply(
                lambda x: x if x in self.major_cats_ else self.other_label
            )
            return X_
        else:
            import numpy as np

            X_ = X.copy()
            X_[:, 0] = np.where(
                np.isin(X_[:, 0], self.major_cats_), X_[:, 0], self.other_label
            )
            return X_


# --------- Fábrica de ColumnTransformer (totalmente parametrizable) ---------


def build_column_transformer(config: dict, features_dict: dict) -> ColumnTransformer:
    """
    Construye el ColumnTransformer de preprocesamiento según config y features_dict.
    Exponible a hiperparámetros para GridSearchCV.
    """
    transformers = []

    # --- Pipeline numérico ---
    num_steps = []
    if config.get("outliers", {}).get("strategy", "none") != "none":
        num_steps.append(
            (
                "outlier",
                OutlierTransformer(
                    strategy=config["outliers"]["strategy"],
                    pmin=config["outliers"]["params"]["pmin"],
                    pmax=config["outliers"]["params"]["pmax"],
                ),
            )
        )
    num_steps.append(
        ("imputer", SimpleImputer(strategy=config["imputation"]["num_method"]))
    )

    scaling = config.get("scaling", "standard")
    if scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    elif scaling == "robust":
        num_steps.append(("scaler", RobustScaler()))

    transformers.append(
        ("num", Pipeline(num_steps), features_dict["final_num_features"])
    )

    # --- Pipeline categórico (uno por columna) ---
    for cat in features_dict["final_cat_features"]:
        steps = []
        # Imputación
        cat_method = config["imputation"]["cat_method"]

        if cat_method == "constant":
            steps.append(
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=config["imputation"].get(
                            "cat_fill_value", "Missing"
                        ),
                    ),
                )
            )
        else:
            steps.append(
                ("imputer", SimpleImputer(strategy=config["imputation"]["cat_method"]))
            )

        # Agrupamiento, solo si está en config["grouping"]
        grouping_cfg = config.get("grouping", {}).get(cat, None)
        if grouping_cfg is not None:
            steps.append(
                (
                    "grouper",
                    CategoryGrouper(
                        col=cat,
                        min_freq=grouping_cfg.get("min_freq", 10),
                        other_label=grouping_cfg.get("other_label", "Other"),
                    ),
                )
            )
        # Encoding (solo OneHot, puedes agregar Ordinal fácilmente)
        steps.append(
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        )
        transformers.append((f"cat_{cat}", Pipeline(steps), [cat]))

    col_trans = ColumnTransformer(transformers=transformers, remainder="drop")
    print(
        f"[INFO] ColumnTransformer construido con {len(transformers)} transformadores."
    )
    return col_trans


# --------- Fábrica de Pipeline (modelo también parametrizable) ---------


def build_pipeline(config: dict, features_dict: dict):
    """
    Construye el pipeline completo: preprocesamiento + modelo, según config y features_dict.
    Tunable y compatible con GridSearchCV.
    """
    col_trans = build_column_transformer(config, features_dict)
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]

    if model_type == "LogisticRegression":
        model = LogisticRegression(**model_params)
    elif model_type == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(**model_params)
    elif model_type == "RandomForestClassifier":
        model = RandomForestClassifier(**model_params)
    # Agrega aquí otros modelos si lo deseas...
    else:
        raise NotImplementedError(f"Modelo no implementado: {model_type}")

    pipe = Pipeline([("preprocessing", col_trans), ("model", model)])
    print(f"[INFO] Pipeline completo construido: {model_type}")
    return pipe


# --------- Fábrica rápida para experimentos ---------


def make_pipeline_for_experiment(config: dict, features_dict: dict):
    """
    Alias rápido para construcción de pipeline, usado en experimentos.
    """
    return build_pipeline(config, features_dict)


# [9.loquesea] Serealiza objetos no serealizables a Json


def make_json_serializable(d):
    """
    Convierte todos los valores de un dict a algo serializable por JSON.
    Los objetos sklearn (como StandardScaler) se pasan a str.
    """
    out = {}
    for k, v in d.items():
        # Si es un objeto de sklearn o similar, pasalo a string
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out
