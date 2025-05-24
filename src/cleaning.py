# src/cleaning.py
import pandas as pd
# Funciones: drop_duplicates, analyze_missing_values,
# log_cleaning_status, fix_column_types_and_formats,
# detect_gross_outliers


# [5.1] Eliminación de Duplicados
def drop_duplicates(df: pd.DataFrame) -> (pd.DataFrame, int):
    """
    Elimina filas duplicadas completas.
    Entradas:
        - df: DataFrame original.
    Salida:
        - Nuevo DataFrame sin duplicados
        - Número de filas eliminadas
    Lógica:
        - Usa df.drop_duplicates()
        - Calcula cuántas filas se eliminaron
    """
    initial_shape = df.shape
    df_clean = df.drop_duplicates()
    removed = initial_shape[0] - df_clean.shape[0]
    print(f"[INFO] Duplicados eliminados: {removed} (shape final: {df_clean.shape})")
    return df_clean, removed


# [5.2] Manejo de NaNs / Valores Faltantes
def analyze_missing_values(df: pd.DataFrame, target_col: str) -> dict:
    """
    Analiza presencia y ubicación de NaNs/valores faltantes.
    Entradas:
        - df: DataFrame.
        - target_col: columna target.
    Salida:
        - dict con:
            - resumen de nulos por columna
            - cantidad total y ubicación relevante
    Lógica:
        - df.isnull().sum()
        - Registrar columnas con nulos, especialmente target
        - No imputa ni elimina aquí, solo documenta
    """
    nans_per_col = df.isnull().sum().to_dict()
    total_nans = sum(nans_per_col.values())
    nans_in_target = nans_per_col.get(target_col, 0)
    summary = {
        "nans_per_col": nans_per_col,
        "total_nans": total_nans,
        "nans_in_target": nans_in_target,
    }
    print(f"[INFO] Nulos por columna: {nans_per_col}")
    print(f"[INFO] Total nulos en dataset: {total_nans}")
    print(f"[INFO] Nulos en target: {nans_in_target}")
    return summary


# [5.3] Registro y Documentación del Estado del Dataset
def log_cleaning_status(
    df: pd.DataFrame, removed_duplicates: int, missing_summary: dict
) -> None:
    """
    Registra el estado tras limpieza de duplicados y análisis de nulos.
    Entradas:
        - df: DataFrame tras limpieza.
        - removed_duplicates: filas eliminadas por duplicación.
        - missing_summary: dict con reporte de nulos.
    Salida:
        - No retorna nada, solo logging/documentación.
    """
    print("[INFO] Estado tras limpieza básica:")
    print(f"  - Shape actual: {df.shape}")
    print(f"  - Duplicados eliminados: {removed_duplicates}")
    print(f"  - Nulos por columna: {missing_summary['nans_per_col']}")
    print(f"  - Total nulos: {missing_summary['total_nans']}")
    print(f"  - Nulos en target: {missing_summary['nans_in_target']}")


# [5.4] Corrección de Formatos y Tipos Evidentes
def fix_column_types_and_formats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifica formatos y corrige tipos de datos evidentes.
    Entradas:
        - df: DataFrame original.
    Salida:
        - DataFrame con formatos/tipos corregidos.
    Lógica:
        - Elimina espacios extra en strings
        - Unifica mayúsculas/minúsculas en categóricas
        - Convierte columnas a numéricas si corresponde
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str).str.strip()
    # (Opcional) Convertir columnas a numéricas si tiene sentido
    # Ejemplo: "education.num", "capital.gain", "capital.loss", "hours.per.week"
    numeric_candidates = [
        "education.num",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
        "age",
    ]
    for col in numeric_candidates:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    print("[INFO] Tipos y formatos corregidos para columnas relevantes.")
    return df_clean


# [5.5] Revisión de Valores Atípicos Evidentes (Superficial)
def detect_gross_outliers(df: pd.DataFrame, num_cols: list) -> dict:
    """
    Detecta posibles outliers groseros para análisis posterior.
    Entradas:
        - df: DataFrame.
        - num_cols: lista de columnas numéricas.
    Salida:
        - dict con rangos y valores fuera de rango observados.
    Lógica:
        - Calcula min/max para cada num_col
        - Identifica valores fuera de rangos esperados (ej: negativos donde no debería haber)
        - No modifica datos, solo diagnóstico
    """
    outliers = {}
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        negative_count = (df[col] < 0).sum()
        outliers[col] = {
            "min": col_min,
            "max": col_max,
            "n_negative": int(negative_count),
        }
        print(
            f"[INFO] Columna '{col}': min={col_min}, max={col_max}, valores negativos={negative_count}"
        )
    return outliers
