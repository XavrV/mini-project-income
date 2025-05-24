# src/target_processing.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Funciones: inspect_target_values,
# clean_normalize_target, plot_target_distribution,
# encode_target, log_target_status


# [4.1] Identificación y Resumen del Target
def inspect_target_values(df: pd.DataFrame, target_col: str) -> dict:
    """
    Identifica y resume la columna target.
    Entradas:
        - df: DataFrame completo.
        - target_col: nombre de la columna target ('income').
    Salida:
        - dict con:
            - valores únicos y frecuencias absolutas/relativas
            - tipo de datos
            - posibles anomalías (nulos, valores inesperados)
    """
    vals = df[target_col].value_counts(dropna=False)
    freqs = df[target_col].value_counts(normalize=True, dropna=False)
    n_nulos = df[target_col].isnull().sum()
    tipos = df[target_col].dtype
    resumen = {
        "unique_values": vals.to_dict(),
        "relative_frequencies": freqs.to_dict(),
        "n_missing": n_nulos,
        "dtype": str(tipos),
    }
    print(
        f"[INFO] Valores únicos y frecuencias en '{target_col}': {resumen['unique_values']}"
    )
    print(f"[INFO] Frecuencias relativas: {resumen['relative_frequencies']}")
    print(f"[INFO] Nulos en target: {n_nulos}")
    print(f"[INFO] Tipo de datos en target: {tipos}")
    return resumen


# [4.2] Limpieza y Normalización del Target
def clean_normalize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Limpia y normaliza los valores del target.
    Entradas:
        - df: DataFrame original.
        - target_col: columna target.
    Salida:
        - Nuevo DataFrame con target limpio (espacios, mayúsculas/minúsculas, valores estándar).
    Lógica:
        - Elimina espacios en blanco, normaliza mayúsculas/minúsculas
        - Corrige typos frecuentes
        - Elimina/marca como nulos los valores vacíos/incorrectos
        - Deja solo dos categorías válidas
    """
    df = df.copy()
    # 1. Limpiar espacios y mayúsculas/minúsculas
    df[target_col] = (
        df[target_col].astype(str).str.strip().str.replace(".", "", regex=False)
    )
    df[target_col] = df[target_col].str.replace(" ", "")  # Elimina todos los espacios
    df[target_col] = df[target_col].str.replace("<=", "<=", regex=False)
    df[target_col] = df[target_col].str.replace(">", ">", regex=False)
    df[target_col] = df[target_col].str.replace("K", "K", regex=False)

    # 2. Solo dejar los valores válidos ('<=50K', '>50K'), otros a NaN
    valid_values = {"<=50K", ">50K"}
    df[target_col] = df[target_col].where(
        df[target_col].isin(valid_values), other=pd.NA
    )
    n_invalid = df[target_col].isna().sum()
    print(
        f"[INFO] Target '{target_col}' normalizado. Valores fuera de ['<=50K', '>50K']: {n_invalid}"
    )
    return df


# [4.3] Visualización del Balance de Clases
def plot_target_distribution(df: pd.DataFrame, target_col: str) -> None:
    """
    Visualiza la distribución de clases del target.
    Entradas:
        - df: DataFrame con target limpio.
        - target_col: columna target.
    Salida:
        - No retorna nada, solo visualización y/o logging.
    Lógica:
        - Gráfico de barras (countplot) con frecuencias absolutas y relativas
        - Muestra tabla resumen
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # Gráfico de cuentas absolutas
    sns.countplot(x=target_col, data=df, ax=ax[0])
    ax[0].set_title("Distribución absoluta del target")
    # Gráfico de cuentas relativas
    rel_freq = df[target_col].value_counts(normalize=True)
    rel_freq.plot(kind="bar", ax=ax[1])
    ax[1].set_title("Distribución relativa del target")
    plt.suptitle(f"Distribución del target: {target_col}")
    plt.tight_layout()
    plt.show()
    print("[INFO] Visualización de distribución de clases mostrada.")


# [4.4] Codificación del Target para Modelado
def encode_target(df: pd.DataFrame, target_col: str, mapping: dict) -> pd.DataFrame:
    """
    Codifica el target para modelado (binario: 0/1).
    Entradas:
        - df: DataFrame con target limpio.
        - target_col: nombre de la columna target.
        - mapping: dict de mapeo (ej: {'<=50K': 0, '>50K': 1}).
    Salida:
        - DataFrame con target codificado (tipo int/categorical).
    """
    df = df.copy()
    # Aplica el mapping (si hay valores fuera de mapping quedan como NaN)
    df[target_col] = df[target_col].map(mapping)
    n_nan = df[target_col].isna().sum()
    if n_nan > 0:
        print(
            f"[WARN] {n_nan} filas tienen valores no mapeados tras codificar el target."
        )
    else:
        print(f"[INFO] Target codificado correctamente ({mapping})")
    df[target_col] = df[target_col].astype("Int64")  # Soporta NaN
    return df


# [4.5] Logging y Registro del Estado del Target
def log_target_status(df: pd.DataFrame, target_col: str, removed_rows: int) -> None:
    """
    Registra el estado post-limpieza del target.
    Entradas:
        - df: DataFrame tras limpieza.
        - target_col: nombre de la columna target.
        - removed_rows: número de filas eliminadas/cambiadas.
    Salida:
        - No retorna nada, solo logging (shape, resumen del target, filas afectadas).
    """
    shape = df.shape
    vals = df[target_col].value_counts(dropna=False).to_dict()
    print(f"[INFO] Estado post-limpieza del target '{target_col}':")
    print(f"  - Shape actual: {shape}")
    print(f"  - Valores codificados: {vals}")
    print(f"  - Filas eliminadas/cambiadas: {removed_rows}")
