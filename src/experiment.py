import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
import os
import json
import joblib
import pandas as pd


# Funciones: setup_mlflow_experiment,
# run_gridsearch_experiment,
# make_json_serializable
def setup_mlflow_experiment(tracking_uri: str, experiment_name: str):
    """
    Configura MLflow: URI y nombre del experimento.
    Si no existe, lo crea.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[INFO] MLflow configurado: {tracking_uri}, experimento: {experiment_name}")


def run_gridsearch_experiment(
    pipeline,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    scoring="accuracy",
    cv=5,
    out_dir="results_out",
    tags=None,
):
    """
    Lanza GridSearchCV sobre el pipeline y trackea todo automáticamente en MLflow.
    Guarda el mejor modelo y el config usados como artefactos.
    """
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
            return_train_score=True,
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
