# src/load_validate.py
import pandas as pd
import numpy as np
# Funciones: load_dataset, validate_structure,
# quick_explore, check_nulls_duplicates, generate_data_dictionary,
# save_initial_state


# [3.1] Carga del Dataset
def load_dataset(path: str, encoding: str = "utf-8", sep: str = ",") -> pd.DataFrame:
    """
    Lee el dataset desde el path configurado.
    Entradas:
        - path: ruta del archivo CSV.
        - encoding: encoding del archivo (por defecto 'utf-8').
        - sep: separador de columnas (por defecto ',').
    Salida:
        - DataFrame con los datos cargados.
    Lógica:
        - Usa pandas.read_csv (try/except para error handling)
        - Devuelve el DataFrame leído o lanza excepción si falla
    """
    try:
        df = pd.read_csv(path, encoding=encoding, sep=sep)
        # Reemplazar '?' por np.nan en TODO el DataFrame
        df.replace("?", np.nan, inplace=True)

        print(f"[INFO] Dataset cargado correctamente desde {path} con shape {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Fallo al cargar dataset desde {path}: {e}")
        raise


# [3.2] Validación de Estructura y Tipos
def validate_structure(df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """
    Verifica que las columnas y tipos sean los esperados.
    Entradas:
        - df: DataFrame cargado.
        - expected_cols: lista de nombres de columnas esperadas.
    Salida:
        - DataFrame con columnas renombradas/tipos corregidos si es necesario.
    Lógica:
        - Chequear que columnas == expected_cols (ignora orden)
        - Loggear advertencia si faltan o sobran columnas
        - (Opcional) Renombrar columnas si hay variantes conocidas
        - (Opcional) Corregir tipos simples (ej: int, float, str)
        - No modifica datos si todo está OK
    """
    df_cols = set(df.columns)
    exp_cols = set(expected_cols)
    missing = exp_cols - df_cols
    extra = df_cols - exp_cols

    if missing:
        print(f"[WARN] Faltan columnas: {missing}")
    if extra:
        print(f"[WARN] Columnas extra no esperadas: {extra}")

    # Opcional: podrías renombrar columnas aquí si sabes que a veces tienen un typo o espacio.
    # Por ahora solo loggeamos.
    # Chequeo rápido de tipos (opcional):
    # Ejemplo: asegurarse que numéricas estén como numéricas, etc.

    print(f"[INFO] Columnas finales en el DataFrame: {list(df.columns)}")
    return df


# [3.3] Exploración Rápida del Dataset
def quick_explore(df: pd.DataFrame) -> dict:
    """
    Realiza una exploración rápida del dataset.
    Entradas:
        - df: DataFrame a explorar.
    Salida:
        - dict con resúmenes rápidos:
            - shape, info, describe, nulos por columna
            - valores únicos por columna
            - posibles outliers (min/max extremos)
    Lógica:
        - Compila los datos básicos en un dict para logging/diagnóstico
        - No modifica el df
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "nans_per_col": df.isnull().sum().to_dict(),
        "unique_values": {col: df[col].nunique() for col in df.columns},
        "describe": df.describe(include="all").T,  # .T para que sea más legible
        "min_per_col": df.select_dtypes(include="number").min().to_dict(),
        "max_per_col": df.select_dtypes(include="number").max().to_dict(),
    }
    print("[INFO] Resumen rápido del dataset:")
    print(f" - Shape: {summary['shape']}")
    print(f" - Nulos por columna: {summary['nans_per_col']}")
    print(f" - Únicos por columna: {summary['unique_values']}")
    return summary


# [3.4] Chequeo de Nulos y Duplicados
def check_nulls_duplicates(df: pd.DataFrame) -> dict:
    """
    Revisa nulos y duplicados.
    Entradas:
        - df: DataFrame a analizar.
    Salida:
        - dict con:
            - cantidad de nulos por columna
            - cantidad de filas duplicadas
    Lógica:
        - df.isnull().sum()
        - df.duplicated().sum()
        - Retorna dict resumen para logging
    """
    nans_per_col = df.isnull().sum().to_dict()
    total_dupes = df.duplicated().sum()
    summary = {"nans_per_col": nans_per_col, "total_duplicates": total_dupes}
    print(f"[INFO] Nulos por columna: {nans_per_col}")
    print(f"[INFO] Total filas duplicadas: {total_dupes}")
    return summary


# [3.5] Descripción de Variables (Data Dictionary)
def generate_data_dictionary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Genera un data dictionary resumido.
    Entradas:
        - df: DataFrame base.
        - target_col: nombre de la columna target.
    Salida:
        - DataFrame/tabla con:
            - nombre, tipo, rol (target/feature), notas/resumen si aplica
    Lógica:
        - Itera por columnas, define tipo (num/cat), rol (feature/target)
        - Opcional: añade significado conocido o notas manuales
    """
    data = []
    for col in df.columns:
        tipo = str(df[col].dtype)
        rol = "target" if col == target_col else "feature"
        notas = ""
        # Podrías agregar descripciones/manuales aquí si lo deseas
        data.append({"name": col, "type": tipo, "role": rol, "notes": notas})
    dict_df = pd.DataFrame(data)
    print("[INFO] Diccionario de datos generado.")
    return dict_df


# [3.6] Guardado de Estado Inicial
def save_initial_state(df: pd.DataFrame, path: str) -> None:
    """
    Guarda el estado inicial del dataset limpio.
    Entradas:
        - df: DataFrame a guardar.
        - path: ruta destino para el archivo de respaldo.
    Salida:
        - No retorna nada, solo guarda archivo y/o logs.
    Lógica:
        - Usa df.to_csv(path)
        - (Opcional) Loggea confirmación y shape guardado
    """
    df.to_csv(path, index=False)
    print(f"[INFO] Estado inicial del dataset guardado en {path} (shape: {df.shape})")


# --- Helpers opcionales ---


def log_shape(df: pd.DataFrame, msg: str = "") -> None:
    print(f"[INFO]{msg} Shape del DataFrame: {df.shape}")


def log_summary(info: dict, msg: str = "") -> None:
    print(f"[INFO]{msg}")
    for k, v in info.items():
        print(f"  {k}: {v}")
