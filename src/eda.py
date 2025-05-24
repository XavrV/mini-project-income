# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Funciones: describe_numerical, describe_categorical,
# plot_numerical_distributions, plot_categorical_distributions,
# plot_feature_target_relationship, correlation_analysis,
# analyze_feature_redundancy, document_eda_findings,
# save_eda_artifacts
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
        res.append(
            {
                "col": col,
                "n_unique": df[col].nunique(),
                "most_freq": counts.idxmax() if not counts.empty else None,
                "freq_most_freq": counts.max() if not counts.empty else None,
                "top_5": counts.head(5).to_dict(),
            }
        )
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
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 4))
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
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 3))
    if n == 1:
        axs = [axs]
    for ax, col in zip(axs, num_cols):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(f"Boxplot de {col}")
    plt.tight_layout()
    plt.show()


# [6.3] Visualización de Variables Categóricas
def plot_categorical_distributions(
    df: pd.DataFrame, cat_cols: list, min_freq: int = 10
) -> None:
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
def plot_feature_target_relationship(
    df: pd.DataFrame, feature_cols: list, target_col: str
) -> None:
    """
    Explora relaciones univariadas entre features y el target.
    """
    for col in feature_cols:
        plt.figure(figsize=(7, 4))
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.boxplot(x=target_col, y=col, data=df)
            plt.title(f"{col} por {target_col}")
        else:
            cross = pd.crosstab(df[col], df[target_col], normalize="index")
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
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
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
