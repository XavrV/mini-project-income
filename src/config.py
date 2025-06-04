config = {
    "path": "data/adult.csv",
    "target": "income",
    "target_mapping": {"<=50K": 0, ">50K": 1},
    "expected_columns": [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education.num",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
        "native.country",
        "income",
    ],
    "features": [
        "age",
        "workclass",
        "education.num",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
        "native.country",
    ],
    "drop_features": ["education", "fnlwgt"],
    "num_features": [
        "age",
        "education.num",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ],
    "cat_features": [
        "workclass",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    ],
    # Outliers (tuneable)
    "outliers": {
        "strategy": "clip",  # o "none"
        "params": {"pmin": 0.01, "pmax": 0.99},  # Para tunear, pásalos en param_grid
    },
    # Imputación de NaNs
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
        "native.country": {"min_freq": 1000, "other_label": "Other"},
    },
    "scaling": "standard",  # O tunéar en param_grid
    "encoding": "onehot",
    "split": {"test_size": 0.2, "random_state": 42},
    "model": {
        "type": "RandomForestClassifier",
        "params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        # "type": "LogisticRegression",
        # "params": {"max_iter": 1000, "solver": "lbfgs"},
    },
    "mlflow": {
        "tracking_uri": "file:./mlruns",
        "experiment_name": "Mini_Proyecto_Adult_Income_LR",
    },
}
