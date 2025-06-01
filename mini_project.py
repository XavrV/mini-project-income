# %% [markdown]
# 

# %% [markdown]
# ### BLOQUE 1: Introduccion, marco teorico y todo eso

# %% [markdown]
# ### BLOQUE 2: Importar Gente y Configuración Inicial

# %%
import os
import sys

import numpy as np
import pandas as pd

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# MLflow para tracking de experimentos
import mlflow
import mlflow.sklearn
import json

# scikit-learn: preprocesado, pipeline, modelos y métricas
import sklearn
from sklearn.model_selection    import train_test_split, GridSearchCV
from sklearn.impute             import SimpleImputer
from sklearn.preprocessing      import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base               import BaseEstimator, TransformerMixin
from sklearn.compose            import ColumnTransformer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.tree               import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble           import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score
)

import scipy
from scipy.stats.mstats import winsorize

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Configuración global reproducible
np.random.seed(42)
RANDOM_STATE = 42

# Mostrar todas las columnas y formateo de floats
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Estilos visuales por defecto
plt.style.use('default')
sns.set_palette('deep')

# Mostrar versiones para reproducibilidad
print("Python:", sys.version)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", plt.matplotlib.__version__)
print("Seaborn:", sns.__version__)
print("MLflow:", mlflow.__version__)
print("Scikit-learn:", sklearn.__version__)


# %%
config = { ## 
    "path": "adult.csv",
    "target": "income",
    "target_mapping": {"<=50K": 0, ">50K": 1},
    "expected_columns": [
        'age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
        'occupation', 'relationship', 'race', 'sex',
        'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income'
    ],

    "features": [
        'age', 'workclass', 'education.num', 'marital.status', 'occupation',
        'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
        'hours.per.week', 'native.country'
    ],
    "drop_features": ['education', 'fnlwgt'],

    "num_features": ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week'],
    "cat_features": [
        'workclass', 'marital.status', 'occupation',
        'relationship', 'race', 'sex', 'native.country'
    ],

    # Outliers (tuneable)
    "outliers": {
        "strategy": "clip",  # o "none"
        "params": {"pmin": 0.01, "pmax": 0.99}  # Para tunear desde param_grid
    },

    # Imputación
    "imputation": {
        "num_method": "median",
        "cat_method": "most_frequent",
        # "cat_fill_value": "Missing"
    },

    # Agrupamiento de categorías poco frecuentes
    "grouping": {
        "workclass": {"min_freq": 3000, "other_label": "Other"},
        "marital.status": {"min_freq": 2000, "other_label": "Other"},
        "occupation": {"min_freq": 2000, "other_label": "Other"},
        "relationship": {"min_freq": 4000, "other_label": "Other"},
        "race": {"min_freq": 500, "other_label": "Other"},
        "native.country": {"min_freq": 1000, "other_label": "Other"}
    },


    "scaling": "standard",   # O tunéar en param_grid
    "encoding": "onehot",    # 
    
    "split": {
        "test_size": 0.2,
        "random_state": 42
    },

    "model": {
        "type": "LogisticRegression",
        "params": {"max_iter": 1000, "solver": "lbfgs"}
    },

    "mlflow": {
        "tracking_uri": "file:./mlruns",
        "experiment_name": "Mini_Project_ Adult_Income"
    }
}


# %% [markdown]
# ### BLOQUE 3: Carga, Validación y Exploración Inicial

# %%
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
        df.replace('?', np.nan, inplace=True)

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
        "min_per_col": df.select_dtypes(include='number').min().to_dict(),
        "max_per_col": df.select_dtypes(include='number').max().to_dict()
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
    summary = {
        "nans_per_col": nans_per_col,
        "total_duplicates": total_dupes
    }
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
        data.append({
            "name": col,
            "type": tipo,
            "role": rol,
            "notes": notas
        })
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



# %%
df = load_dataset(config["path"])
df = validate_structure(df, config["expected_columns"])
resumen = quick_explore(df)
check_nulls_duplicates(df)
data_dict = generate_data_dictionary(df, config["target"])
save_initial_state(df, "adult_initial.csv")

# %% [markdown]
# ### BLOQUE 4: Procesamiento y Codificación del Target

# %%
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
        "dtype": str(tipos)
    }
    print(f"[INFO] Valores únicos y frecuencias en '{target_col}': {resumen['unique_values']}")
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
    df[target_col] = df[target_col].astype(str).str.strip().str.replace(".", "", regex=False)
    df[target_col] = df[target_col].str.replace(" ", "")  # Elimina todos los espacios
    df[target_col] = df[target_col].str.replace("<=", "<=", regex=False)
    df[target_col] = df[target_col].str.replace(">", ">", regex=False)
    df[target_col] = df[target_col].str.replace("K", "K", regex=False)
    
    # 2. Solo dejar los valores válidos ('<=50K', '>50K'), otros a NaN
    valid_values = {"<=50K", ">50K"}
    df[target_col] = df[target_col].where(df[target_col].isin(valid_values), other=pd.NA)
    n_invalid = df[target_col].isna().sum()
    print(f"[INFO] Target '{target_col}' normalizado. Valores fuera de ['<=50K', '>50K']: {n_invalid}")
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
    rel_freq.plot(kind='bar', ax=ax[1])
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
        print(f"[WARN] {n_nan} filas tienen valores no mapeados tras codificar el target.")
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



# %%
resumen_target = inspect_target_values(df, config["target"])
df = clean_normalize_target(df, config["target"])
plot_target_distribution(df, config["target"])

# %%
df = encode_target(df, config["target"], config["target_mapping"])
removed_rows = df[config["target"]].isna().sum()
log_target_status(df, config["target"], removed_rows)

# %% [markdown]
# ### BLOQUE 5: Limpieza General y Revisión de Consistencias

# %%
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
        "nans_in_target": nans_in_target
    }
    print(f"[INFO] Nulos por columna: {nans_per_col}")
    print(f"[INFO] Total nulos en dataset: {total_nans}")
    print(f"[INFO] Nulos en target: {nans_in_target}")
    return summary

# [5.3] Registro y Documentación del Estado del Dataset
def log_cleaning_status(df: pd.DataFrame, removed_duplicates: int, missing_summary: dict) -> None:
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
    numeric_candidates = ["education.num", "capital.gain", "capital.loss", "hours.per.week", "age"]
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
            "n_negative": int(negative_count)
        }
        print(f"[INFO] Columna '{col}': min={col_min}, max={col_max}, valores negativos={negative_count}")
    return outliers


# %%
df, removed_duplicates = drop_duplicates(df)
missing_summary = analyze_missing_values(df, config["target"])
log_cleaning_status(df, removed_duplicates, missing_summary)
df = fix_column_types_and_formats(df)
outliers_report = detect_gross_outliers(df, config["num_features"])

# %% [markdown]
# ### BLOQUE 6: Análisis Exploratorio de Datos (EDA)

# %%
# [6.1] Estadísticos Descriptivos Generales
def describe_numerical(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Calcula resumen estadístico para variables numéricas.
    """
    desc = df[num_cols].describe().T
    print("[INFO] Estadísticos descriptivos (numéricas):")
    print(desc)
    return desc

def describe_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Calcula resumen descriptivo para variables categóricas.
    """
    res = []
    for col in cat_cols:
        counts = df[col].value_counts()
        res.append({
            "col": col,
            "n_unique": df[col].nunique(),
            "most_freq": counts.idxmax() if not counts.empty else None,
            "freq_most_freq": counts.max() if not counts.empty else None,
            "top_5": counts.head(5).to_dict()
        })
    desc = pd.DataFrame(res)
    print("[INFO] Estadísticos descriptivos (categóricas):")
    print(desc)
    return desc

# [6.2] Visualización de Variables Numéricas
def plot_numerical_distributions(df: pd.DataFrame, num_cols: list) -> None:
    """
    Visualiza distribuciones de variables numéricas.
    """
    n = len(num_cols)
    fig, axs = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axs = [axs]
    for ax, col in zip(axs, num_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Histograma de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

    # Boxplots por si quieres verlo más claro
    fig, axs = plt.subplots(1, n, figsize=(5*n, 3))
    if n == 1:
        axs = [axs]
    for ax, col in zip(axs, num_cols):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(f"Boxplot de {col}")
    plt.tight_layout()
    plt.show()

# [6.3] Visualización de Variables Categóricas
def plot_categorical_distributions(df: pd.DataFrame, cat_cols: list, min_freq: int = 10) -> None:
    """
    Visualiza conteos de categorías para variables categóricas.
    """
    for col in cat_cols:
        counts = df[col].value_counts()
        # Agrupa categorías poco frecuentes
        small_cats = counts[counts < min_freq].index
        df_plot = df[col].replace(small_cats, "Other")
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df_plot, order=df_plot.value_counts().index)
        plt.title(f"Distribución de {col} (agrupando <{min_freq})")
        plt.show()

# [6.4] Relación Univariada de Features con el Target
def plot_feature_target_relationship(df: pd.DataFrame, feature_cols: list, target_col: str) -> None:
    """
    Explora relaciones univariadas entre features y el target.
    """
    for col in feature_cols:
        plt.figure(figsize=(7, 4))
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.boxplot(x=target_col, y=col, data=df)
            plt.title(f"{col} por {target_col}")
        else:
            cross = pd.crosstab(df[col], df[target_col], normalize='index')
            cross.plot(kind="bar", stacked=True)
            plt.title(f"{col} vs {target_col}")
            plt.xlabel(col)
            plt.ylabel("Proporción")
            plt.legend(title=target_col)
            plt.tight_layout()
            plt.show()

# [6.5] Correlaciones y Relaciones Notables
def correlation_analysis(df: pd.DataFrame, num_cols: list) -> (pd.DataFrame, object):
    """
    Calcula y visualiza matriz de correlaciones para numéricas.
    """
    corr = df[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Matriz de correlaciones numéricas")
    plt.show()
    return corr

def analyze_feature_redundancy(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Detecta relaciones o redundancias notables entre features.
    """
    redundancy = {}
    corr = df[feature_cols].corr()
    threshold = 0.95  # Sugerido, puedes parametrizarlo
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            c1 = corr.columns[i]
            c2 = corr.columns[j]
            val = corr.iloc[i, j]
            if abs(val) > threshold:
                redundancy[(c1, c2)] = val
    print(f"[INFO] Pares de features con |correlación| > {threshold}: {redundancy}")
    return redundancy

# [6.6] Registro de Observaciones y Hallazgos
def document_eda_findings(findings: dict, save_path: str = None) -> None:
    """
    Documenta observaciones clave del EDA.
    """
    print("[INFO] Hallazgos del EDA:")
    for k, v in findings.items():
        print(f"  - {k}: {v}")
    if save_path is not None:
        with open(save_path, "w") as f:
            for k, v in findings.items():
                f.write(f"{k}: {v}\n")
        print(f"[INFO] Hallazgos EDA guardados en {save_path}")

# [6.7] Logging y Exportación de Resultados del EDA (Opcional)
def save_eda_artifacts(plots: list, tables: list, out_dir: str) -> None:
    """
    Guarda los gráficos y tablas relevantes generados durante el EDA.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    for i, fig in enumerate(plots):
        fig_path = f"{out_dir}/eda_plot_{i}.png"
        fig.savefig(fig_path)
        print(f"[INFO] Gráfico guardado en {fig_path}")
    for i, table in enumerate(tables):
        table_path = f"{out_dir}/eda_table_{i}.csv"
        table.to_csv(table_path)
        print(f"[INFO] Tabla guardada en {table_path}")
    print(f"[INFO] Artefactos EDA guardados en {out_dir}")


# %%
desc_num = describe_numerical(df, config["num_features"])
desc_cat = describe_categorical(df, config["cat_features"])
plot_numerical_distributions(df, config["num_features"])

# 6.3 Visualización de Variables Categóricas (agrupando categorías poco frecuentes)
plot_categorical_distributions(df, config["cat_features"], min_freq=10)

# 6.4 Relación Univariada de Features con el Target
plot_feature_target_relationship(df, config["features"], config["target"])

# 6.5 Correlaciones y Relaciones Notables
corr_matrix = correlation_analysis(df, config["num_features"])
redundancies = analyze_feature_redundancy(df, config["num_features"])

# 6.6 (Opcional) Documentación de Hallazgos del EDA
findings = {
    "desc_num": desc_num.to_dict(),
    "desc_cat": desc_cat.to_dict(),
    "correlaciones": corr_matrix.to_dict(),
    "redundancias": redundancies,
    # Puedes agregar insights manualmente después de revisar visualizaciones
}
document_eda_findings(findings, save_path="eda_findings.txt")

# 6.7 (Opcional) Guardado de artefactos de EDA
# (Si quieres guardar gráficos/tablas, deberías recolectarlos en listas)
# save_eda_artifacts(plots, tables, out_dir="eda_artifacts")

# %% [markdown]
# ### BLOQUE 7: Ingeniería y Selección de Features YA COPIAIDO!!!!!

# %%
# [7.1] Análisis y Decisión sobre Features Redundantes o Irrelevantes
def analyze_feature_redundancy_irrelevance(df: pd.DataFrame, features_info: dict) -> dict:
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
    other_label: str = "Other"
) -> dict:
    """
    Permite definir min_freq distinto por columna (usando un dict), o usar uno por defecto.
    """
    groupings = {}
    for col in cat_cols:
        min_freq = min_freqs[col] if min_freqs and col in min_freqs else default_min_freq
        freqs = df[col].value_counts()
        rare = freqs[freqs < min_freq].index.tolist()
        if rare:
            groupings[col] = {
                "rare_categories": rare,
                "min_freq": min_freq,
                "other_label": other_label
            }
            print(f"[INFO] En '{col}' se agruparán {len(rare)} categorías poco frecuentes como '{other_label}' (min_freq={min_freq}).")
    return groupings

# [7.3] Documentación y Preparación de Features Finales
def prepare_final_features(df: pd.DataFrame, drop_features: list, target_col: str) -> dict:
    """
    Define y documenta las listas finales de features para modelado.
    """
    cols = [c for c in df.columns if c not in drop_features + [target_col]]
    num_features = df[cols].select_dtypes(include='number').columns.tolist()
    cat_features = [c for c in cols if c not in num_features]
    features_dict = {
        "final_features": cols,
        "final_num_features": num_features,
        "final_cat_features": cat_features
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


# %%
# 7.1 Análisis y documentación de features
features_info = {
    "redundant": [],  # Ejemplo: ['education'] si no la has dropeado ya
    "irrelevant": []  # Ejemplo: ['fnlwgt']
}
redundancy_doc = analyze_feature_redundancy_irrelevance(df, features_info)

# 7.2 Propuesta de agrupamiento de categorías poco frecuentes (SEGUN DATOS DE EDA)
min_freqs = {
    "native.country": 1000,
    "occupation": 2000,
    "race": 500,
    'workclass': 3000,
    'marital.status': 2000,
    'relationship': 4000, 
}
category_groupings = propose_category_groupings(
    df,
    config["cat_features"],
    min_freqs=min_freqs,
    default_min_freq=10  # Puedes ajustar según el EDA o dejarlo así si tu config ya agrupa con thresholds personalizados
)

# 7.3 Definición y documentación de features finales para modelado
features_dict = prepare_final_features(
    df,
    config["drop_features"],
    config["target"]
)
# (Esto te da listas: final_features, final_num_features, final_cat_features)

# 7.4 Identificación de problemas potenciales para el pipeline
pipeline_issues = collect_pipeline_issues(df, features_dict)

# 7.5 Guardado de la configuración de features
save_feature_config(features_dict, "features_config.json")

# %% [markdown]
# ### BLOQUE 8: Preparación de X/y y Split Estratificado

# %%
# [8.1] Preparación de Features y Target para el Split
def prepare_X_y(df: pd.DataFrame, features_dict: dict, target_col: str) -> (pd.DataFrame, pd.Series):
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
def stratified_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Realiza el split train/test estratificado y reproducible.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Split estratificado hecho. X_train: {X_train.shape}, X_test: {X_test.shape}")
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

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
        "n_test_null": n_test_null
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

# %%
# 8.1 Preparación de features y target para el split
X, y = prepare_X_y(df, features_dict, config["target"])

# 8.2 División reproducible y estratificada
splits = stratified_train_test_split(
    X,
    y,
    test_size=config["split"]["test_size"],
    random_state=config["split"]["random_state"]
)

# 8.3 Chequeo de calidad post-split (verifica balance de clases y posibles nulos)
split_quality = check_split_quality(splits["y_train"], splits["y_test"])

# 8.4 Logging y guardado del estado post-split (opcional)
log_and_save_split_state(splits, out_dir="split_artifacts")

X_train = splits["X_train"]
X_test = splits["X_test"]
y_train = splits["y_train"]
y_test = splits["y_test"]

# %%
desc_num = describe_numerical(X_train, config["num_features"])      # Resvisamos como se ve el train set
desc_cat = describe_categorical(X_train, config["cat_features"])
plot_numerical_distributions(X_train, config["num_features"])

# 6.3 Visualización de Variables Categóricas (agrupando categorías poco frecuentes)
plot_categorical_distributions(X_train, config["cat_features"], min_freq=10)

# 6.4 Relación Univariada de Features con el Target
plot_feature_target_relationship(X_train, config["features"], config["target"])

# 6.5 Correlaciones y Relaciones Notables
corr_matrix = correlation_analysis(X_train, config["num_features"])
redundancies = analyze_feature_redundancy(X_train, config["num_features"])

# 6.6 (Opcional) Documentación de Hallazgos del EDA
findings = {
    "desc_num": desc_num.to_dict(),
    "desc_cat": desc_cat.to_dict(),
    "correlaciones": corr_matrix.to_dict(),
    "redundancias": redundancies,
    # Puedes agregar insights manualmente después de revisar visualizaciones
}
document_eda_findings(findings, save_path="eda_findings.txt")

# 6.7 (Opcional) Guardado de artefactos de EDA
# (Si quieres guardar gráficos/tablas, deberías recolectarlos en listas)
# save_eda_artifacts(plots, tables, out_dir="eda_artifacts")

# %% [markdown]
# ### BLOQUE 9: Wrappers, ColumnTransformer y Pipeline Modular

# %%
# === BLOQUE 9: WRAPPERS, COLUMNTRANSFORMER Y PIPELINE MODULAR ===

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
                col: (
                    X[col].quantile(self.pmin),
                    X[col].quantile(self.pmax)
                ) for col in X.columns
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
            X_.iloc[:, 0] = X_.iloc[:, 0].apply(lambda x: x if x in self.major_cats_ else self.other_label)
            return X_
        else:
            import numpy as np
            X_ = X.copy()
            X_[:, 0] = np.where(np.isin(X_[:, 0], self.major_cats_), X_[:, 0], self.other_label)
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
            ("outlier", OutlierTransformer(
                strategy=config["outliers"]["strategy"],
                pmin=config["outliers"]["params"]["pmin"],
                pmax=config["outliers"]["params"]["pmax"]
            ))
        )
    num_steps.append(("imputer", SimpleImputer(strategy=config["imputation"]["num_method"])))

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
            steps.append((
                "imputer", 
                SimpleImputer(
                    strategy="constant",
                    fill_value=config["imputation"].get("cat_fill_value", "Missing")
                )
            ))
        else:
            steps.append((
            "imputer", 
            SimpleImputer(strategy=config["imputation"]["cat_method"])
        ))

        # Agrupamiento, solo si está en config["grouping"]
        grouping_cfg = config.get("grouping", {}).get(cat, None)
        if grouping_cfg is not None:
            steps.append((
                "grouper", 
                CategoryGrouper(
                    col=cat,
                    min_freq=grouping_cfg.get("min_freq", 10),
                    other_label=grouping_cfg.get("other_label", "Other")
                )
            ))
        # Encoding (solo OneHot, puedes agregar Ordinal fácilmente)
        steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transformers.append(
            (f"cat_{cat}", Pipeline(steps), [cat])
        )

    col_trans = ColumnTransformer(transformers=transformers, remainder="drop")
    print(f"[INFO] ColumnTransformer construido con {len(transformers)} transformadores.")
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

    pipe = Pipeline([
        ("preprocessing", col_trans),
        ("model", model)
    ])
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


# %% [markdown]
# ### BLOQUE 10: MLflow, Métricas, Experimentos y Comparación

# %%
# === BLOQUE 10: EXPERIMENTACIÓN AUTOMÁTICA Y MLflow AUTOLOG ===

def setup_mlflow_experiment(tracking_uri: str, experiment_name: str):
    """
    Configura MLflow: URI y nombre del experimento.
    Si no existe, lo crea.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[INFO] MLflow configurado: {tracking_uri}, experimento: {experiment_name}")

def run_gridsearch_experiment(
    pipeline, param_grid, X_train, y_train, X_test, y_test,
    scoring="accuracy", cv=5, out_dir="results_out", tags=None
):
    """
    Lanza GridSearchCV sobre el pipeline y trackea todo automáticamente en MLflow.
    Guarda el mejor modelo y el config usados como artefactos.
    """
    import os, json, joblib

    mlflow.sklearn.autolog()  # ¡clave! Trackea todo el GridSearch automáticamente

    with mlflow.start_run(tags=tags):
        # Lanzamos GridSearchCV
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )
        grid.fit(X_train, y_train)

        # Loguea los resultados y el mejor modelo
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_
        print("[INFO] Mejor score (cv):", best_score)
        print("[INFO] Mejores hiperparámetros:", best_params)

        # Loguea score en test set
        test_score = grid.score(X_test, y_test)
        mlflow.log_metric("test_score", test_score)

        # Guarda artefactos extra: modelo, params, resultados de GridSearch
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(best_model, f"{out_dir}/best_model.joblib")
        with open(f"{out_dir}/best_params.json", "w") as f:
            json.dump(make_json_serializable(best_params), f, indent=2)


        # GridSearchCV cv_results_
        pd.DataFrame(grid.cv_results_).to_csv(f"{out_dir}/cv_results.csv", index=False)
        mlflow.log_artifacts(out_dir)

    print(f"[INFO] Experimento terminado y trackeado en MLflow.")

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


# Ejemplo de uso (pseudo-código):
"""
setup_mlflow_experiment(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment_name"])

param_grid = {
    # Parametriza hiperparámetros del pipeline y modelo
    "preprocessing__num__outlier__pmin": [0.01, 0.05],
    "preprocessing__num__outlier__pmax": [0.95, 0.99],
    "preprocessing__num__scaler": [StandardScaler(), RobustScaler()],
    "model__C": [0.1, 1, 10],
    # ...agrega otros hiperparámetros aquí...
}

pipeline = make_pipeline_for_experiment(config, features_dict)

run_gridsearch_experiment(
    pipeline, param_grid,
    X_train, y_train, X_test, y_test,
    scoring="accuracy",
    cv=5,
    out_dir="results_out"
)
"""


# %%
# === PLANTILLA 00 EXPERIMENTO END-TO-END ===

# 1. Preparación del dict de features (puede venir de tu EDA o selección previa)
features_dict = {
    "final_num_features": config["num_features"],         # ajusta si haces selección de features
    "final_cat_features": config["cat_features"],         # ajusta si haces selección de features
}

# 2. Construcción del pipeline tunable
# pipeline = make_pipeline_for_experiment(config, features_dict)

pipeline = Pipeline([
    ("preprocessing", build_column_transformer(config, features_dict)),
    ("model", LogisticRegression()) # Placeholder, será sobreescrito por GridSearchCV
])

# 3. Definición del espacio de hiperparámetros (param_grid)

param_grid = [
    # Logistic Regression
    {
        "model": [LogisticRegression(max_iter=1000, solver="lbfgs")],
        "model__C": [0.1, 1, 10, 100],
        "model__penalty": ["l2"],
        "preprocessing__num__outlier__pmin": [0.01, 0.05],
        "preprocessing__num__outlier__pmax": [0.95, 0.99],
        "preprocessing__num__outlier__strategy": ["clip", "none"],
        "preprocessing__num__scaler": [StandardScaler(), RobustScaler()],
        
        # "preprocessing__cat_native.country__grouper__min_freq": [50, 100, 200],
        # "preprocessing__cat_workclass__grouper__min_freq": [1000, 2000, 3000],
        # "preprocessing__cat_occupation__grouper__min_freq": [1000, 2000],
        # "preprocessing__cat_relationship__grouper__min_freq": [1000, 2000],
        # "preprocessing__cat_marital.status__grouper__min_freq": [1000, 2000],
        # "preprocessing__cat_race__grouper__min_freq": [1000, 2000],
        

    },
    # Decision Tree
    {
        "model": [DecisionTreeClassifier()],
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 10, 20],
        "preprocessing__num__outlier__pmin": [0.01, 0.05],
        "preprocessing__num__outlier__pmax": [0.95, 0.99],
        "preprocessing__num__outlier__strategy": ["clip", "none"],
        "preprocessing__num__scaler": [StandardScaler(), RobustScaler()],
    },
    # Random Forest
    {
        "model": [RandomForestClassifier(n_estimators=100)],
        "model__max_depth": [5, 10, 20, None],
        "model__min_samples_split": [2, 10],
        "preprocessing__num__outlier__pmin": [0.01, 0.05],
        "preprocessing__num__outlier__pmax": [0.95, 0.99],
        "preprocessing__num__outlier__strategy": ["clip", "none"],
        "preprocessing__num__scaler": [StandardScaler(), RobustScaler()],
    }
]

# 5. Setup MLflow (URI y experimento vienen del config)
setup_mlflow_experiment(
    tracking_uri=config["mlflow"]["tracking_uri"],
    experiment_name=config["mlflow"]["experiment_name"]
)

# 6. Lanza el experimento completo y trackea todo
run_gridsearch_experiment(
    pipeline, param_grid,
    X_train, y_train, X_test, y_test,
    scoring="accuracy",      # Cambia a roc_auc, f1, etc. según tu métrica objetivo
    cv=5,                    # Folds de CV
    out_dir="results_out",
    tags={"proyecto": "test0", "autor": "Javiercito"}
)

# %%
# === PLANTILLA 01 EXPERIMENTO END-TO-END ===

# 1. Preparación del dict de features (puede venir de tu EDA o selección previa)
features_dict = {
    "final_num_features": config["num_features"],         # ajusta si haces selección de features
    "final_cat_features": config["cat_features"],         # ajusta si haces selección de features
}

# 2. Construcción del pipeline tunable
# pipeline = make_pipeline_for_experiment(config, features_dict)

pipeline = Pipeline([
    ("preprocessing", build_column_transformer(config, features_dict)),
    ("model", LogisticRegression()) # Placeholder, será sobreescrito por GridSearchCV
])

# 3. Definición del espacio de hiperparámetros (param_grid)

param_grid = [
    {
        "model": [RandomForestClassifier(n_estimators=100, random_state=42)],
        "model__max_depth": [10, 15, 20, None],
        "model__min_samples_split": [10, 20, 50],
        # (Opcional)
        # "model__max_features": ["auto", "sqrt", 0.5],

        # Preprocesamiento numérico fijo
        "preprocessing__num__outlier__strategy": ["none"],
        "preprocessing__num__scaler": [RobustScaler()],
    }
]

# 5. Setup MLflow (URI y experimento vienen del config)
setup_mlflow_experiment(
    tracking_uri=config["mlflow"]["tracking_uri"],
    experiment_name=config["mlflow"]["experiment_name"]
)

# 6. Lanza el experimento completo y trackea todo
run_gridsearch_experiment(
    pipeline, param_grid,
    X_train, y_train, X_test, y_test,
    scoring="accuracy",      # Cambia a roc_auc, f1, etc. según tu métrica objetivo
    cv=5,                    # Folds de CV
    out_dir="results_out",
    tags={"proyecto": "test1", "autor": "Javiercito"}
)

# %%
# === PLANTILLA 02 EXPERIMENTO END-TO-END ===

# Configuración robusta identificada
pipeline = Pipeline([
    ("preprocessing", build_column_transformer(config, features_dict)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        random_state=42
    ))
])

# Param_grid para grouping de dos categóricas
param_grid = [
    {
        # Grouping tunable
        "preprocessing__cat_native.country__grouper__min_freq": [5000],
        "preprocessing__cat_workclass__grouper__min_freq": [3000],
        "preprocessing__cat_occupation__grouper__min_freq": [2000, 5000, 7000],
        "preprocessing__cat_relationship__grouper__min_freq": [4000, 6000],
        "preprocessing__cat_marital.status__grouper__min_freq": [2000, 3000, 4000],
        "preprocessing__cat_race__grouper__min_freq": [2000],
        # Preprocesamiento numérico fijo
        "preprocessing__num__outlier__strategy": ["none"],
        "preprocessing__num__scaler": [RobustScaler()],
    }
]

# ... El resto igual
run_gridsearch_experiment(
    pipeline, param_grid,
    X_train, y_train, X_test, y_test,
    scoring="accuracy",
    cv=5,
    out_dir="results_out_grouping",
    tags={"proyecto": "rf_grouping", "autor": "Javiercito"}
)


# %%



