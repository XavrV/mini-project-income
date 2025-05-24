# === Mini Project Income: Flujo Maestro Modular ===
import pandas as pd
from src.config import config
from src import (
    load_validate,
    target_processing,
    cleaning,
    eda,
    feature_engineering,
    split,
    pipeline,
    experiment,
)


def run_loading():
    df = load_validate.load_dataset(config["path"])
    df = load_validate.validate_structure(df, config["expected_columns"])
    load_validate.quick_explore(df)
    load_validate.check_nulls_duplicates(df)
    return df


def run_target_processing(df):
    df = target_processing.clean_normalize_target(df, config["target"])
    df = target_processing.encode_target(df, config["target"], config["target_mapping"])
    target_processing.plot_target_distribution(df, config["target"])
    removed_rows = df[config["target"]].isna().sum()
    target_processing.log_target_status(df, config["target"], removed_rows)
    return df


def run_cleaning(df):
    df, _ = cleaning.drop_duplicates(df)
    cleaning.analyze_missing_values(df, config["target"])
    df = cleaning.fix_column_types_and_formats(df)
    cleaning.detect_gross_outliers(df, config["num_features"])
    return df


def run_eda(df):
    eda.describe_numerical(df, config["num_features"])
    eda.describe_categorical(df, config["cat_features"])
    eda.plot_numerical_distributions(df, config["num_features"])
    eda.plot_categorical_distributions(df, config["cat_features"], min_freq=10)
    eda.plot_feature_target_relationship(df, config["features"], config["target"])
    corr = eda.correlation_analysis(df, config["num_features"])
    eda.analyze_feature_redundancy(df, config["num_features"])


def run_feature_engineering(df):
    features_dict = feature_engineering.prepare_final_features(
        df, config["drop_features"], config["target"]
    )
    feature_engineering.collect_pipeline_issues(df, features_dict)
    feature_engineering.save_feature_config(features_dict, "features_config.json")
    return features_dict


def run_splitting(df, features_dict):
    X, y = split.prepare_X_y(df, features_dict, config["target"])
    splits = split.stratified_train_test_split(X, y)
    split.check_split_quality(splits["y_train"], splits["y_test"])
    return splits


def run_experimentation(splits, features_dict):
    pipe = pipeline.make_pipeline_for_experiment(config, features_dict)
    param_grid = {
        "model__C": [0.1, 1, 10],
        "preprocessing__num__outlier__strategy": ["clip", "none"],
    }
    experiment.setup_mlflow_experiment(
        tracking_uri=config["mlflow"]["tracking_uri"],
        experiment_name=config["mlflow"]["experiment_name"],
    )
    experiment.run_gridsearch_experiment(
        pipe,
        param_grid,
        splits["X_train"],
        splits["y_train"],
        splits["X_test"],
        splits["y_test"],
        scoring="accuracy",
        out_dir="results_out",
        tags={"proyecto": "mini_project_income", "autor": "Javiercito"},
    )


def run_pipeline():
    df = run_loading()
    df = run_target_processing(df)
    df = run_cleaning(df)
    # run_eda(df)
    features_dict = run_feature_engineering(df)
    splits = run_splitting(df, features_dict)
    run_experimentation(splits, features_dict)


if __name__ == "__main__":
    run_pipeline()
