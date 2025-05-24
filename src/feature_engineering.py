# src/feature_engineering.py
import pandas as pd
import json
# Funciones: analyze_feature_redundancy_irrelevance,
# propose_category_groupings, prepare_final_features,
# collect_pipeline_issues, save_feature_config


# [7.1] Análisis y Decisión sobre Features Redundantes o Irrelevantes
def analyze_feature_redundancy_irrelevance(
    df: pd.DataFrame, features_info: dict
) -> dict:
    """
    Analiza y documenta features redundantes o irrelevantes.
    """
    # features_info puede contener correlaciones, cardinalidades, etc.
    redundantes = features_info.get("redundant", [])
    irrelevantes = features_info.get("irrelevant", [])
    doc = {}
    if redundantes:
        doc["redundant"] = redundantes
        print(f"[INFO] Features redundantes: {redundantes}")
    if irrelevantes:
        doc["irrelevant"] = irrelevantes
        print(f"[INFO] Features irrelevantes: {irrelevantes}")
    return doc


# [7.2] Limpieza y Agrupamiento de Categorías en Variables Categóricas
def propose_category_groupings(
    df: pd.DataFrame,
    cat_cols: list,
    min_freqs: dict = None,
    default_min_freq: int = 10,
    other_label: str = "Other",
) -> dict:
    """
    Permite definir min_freq distinto por columna (usando un dict), o usar uno por defecto.
    """
    groupings = {}
    for col in cat_cols:
        min_freq = (
            min_freqs[col] if min_freqs and col in min_freqs else default_min_freq
        )
        freqs = df[col].value_counts()
        rare = freqs[freqs < min_freq].index.tolist()
        if rare:
            groupings[col] = {
                "rare_categories": rare,
                "min_freq": min_freq,
                "other_label": other_label,
            }
            print(
                f"[INFO] En '{col}' se agruparán {len(rare)} categorías poco frecuentes como '{other_label}' (min_freq={min_freq})."
            )
    return groupings


# [7.3] Documentación y Preparación de Features Finales
def prepare_final_features(
    df: pd.DataFrame, drop_features: list, target_col: str
) -> dict:
    """
    Define y documenta las listas finales de features para modelado.
    """
    cols = [c for c in df.columns if c not in drop_features + [target_col]]
    num_features = df[cols].select_dtypes(include="number").columns.tolist()
    cat_features = [c for c in cols if c not in num_features]
    features_dict = {
        "final_features": cols,
        "final_num_features": num_features,
        "final_cat_features": cat_features,
    }
    print(f"[INFO] Features finales para modelado: {features_dict}")
    return features_dict


# [7.4] Identificación de Problemas Potenciales para el Pipeline
def collect_pipeline_issues(df: pd.DataFrame, features_dict: dict) -> dict:
    """
    Documenta problemas potenciales y necesidades de preprocesamiento para el pipeline.
    """
    issues = {}
    # NaNs
    for col in features_dict["final_num_features"]:
        n_nans = df[col].isnull().sum()
        if n_nans > 0:
            issues[col] = {"nans": n_nans, "type": "numeric"}
    for col in features_dict["final_cat_features"]:
        n_nans = df[col].isnull().sum()
        if n_nans > 0:
            issues[col] = {"nans": n_nans, "type": "categorical"}
    # Cardinalidad alta
    for col in features_dict["final_cat_features"]:
        n_unique = df[col].nunique()
        if n_unique > 20:  # threshold arbitrario, ajustable
            issues[col] = issues.get(col, {})
            issues[col].update({"high_cardinality": n_unique})
    print(f"[INFO] Problemas/preprocesamientos detectados para el pipeline: {issues}")
    return issues


# [7.5] Logging y Guardado de Configuración de Features
def save_feature_config(features_dict: dict, config_path: str) -> None:
    """
    Guarda y documenta la configuración final de features.
    """
    with open(config_path, "w") as f:
        json.dump(features_dict, f, indent=2)
    print(f"[INFO] Configuración de features guardada en {config_path}")
