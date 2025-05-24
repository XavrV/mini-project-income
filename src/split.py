# src/split.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Funciones: prepare_X_y, stratified_train_test_split,
# check_split_quality, log_and_save_split_state


# [8.1] Preparación de Features y Target para el Split
def prepare_X_y(
    df: pd.DataFrame, features_dict: dict, target_col: str
) -> (pd.DataFrame, pd.Series):
    """
    Prepara X (features) e y (target) a partir de las listas definidas.
    """
    # Puedes adaptar keys según tu features_dict final:
    features = features_dict.get("final_features", [])
    X = df[features].copy()
    y = df[target_col].copy()
    print(f"[INFO] Matriz X preparada (shape: {X.shape}), vector y (shape: {y.shape})")
    return X, y


# [8.2] División Reproducible y Estratificada
def stratified_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> dict:
    """
    Realiza el split train/test estratificado y reproducible.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"[INFO] Split estratificado hecho. X_train: {X_train.shape}, X_test: {X_test.shape}"
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# [8.3] Chequeos de Calidad Post-Split
def check_split_quality(y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Verifica la distribución del target en train y test y chequea posibles anomalías.
    """
    train_dist = y_train.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()
    n_train, n_test = len(y_train), len(y_test)
    n_train_null = y_train.isna().sum()
    n_test_null = y_test.isna().sum()
    obs = {
        "train_dist": train_dist,
        "test_dist": test_dist,
        "n_train": n_train,
        "n_test": n_test,
        "n_train_null": n_train_null,
        "n_test_null": n_test_null,
    }
    print(f"[INFO] Distribución de clases (train): {train_dist}")
    print(f"[INFO] Distribución de clases (test): {test_dist}")
    if n_train_null or n_test_null:
        print(f"[WARN] Nulos en splits - train: {n_train_null}, test: {n_test_null}")
    return obs


# [8.4] Guardado y Logging del Estado Post-Split
def log_and_save_split_state(split_dict: dict, out_dir: str = None) -> None:
    """
    Loguea y guarda el estado post-split.
    """
    for k, v in split_dict.items():
        print(f"[INFO] {k}: shape {v.shape}")
    if out_dir:
        import os

        os.makedirs(out_dir, exist_ok=True)
        for k, v in split_dict.items():
            save_path = f"{out_dir}/{k}.csv"
            v.to_csv(save_path, index=False)
            print(f"[INFO] {k} guardado en {save_path}")
