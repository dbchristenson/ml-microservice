"""
This module contains functions to train, evaluate, and save a model.

It uses the RandomForestRegressor from sklearn and the Loguru library
for logging. The model is trained using grid search for hyperparameter tuning,
and the best model is saved to disk. The module also includes functions to
split the data into features and target, evaluate the model, and save the model
to disk.
"""

import pickle as pk

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from src.config.config import settings
from src.model.pipeline.preparation import prepare_data


def _get_X_y(df: pd.DataFrame, x_cols: list | None = None) -> tuple:
    """
    Splits the DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_cols (list | None): List of feature columns. If None, all columns " \
                              "except 'rent' are used."

    Returns:
        tuple: Tuple with the features DataFrame (X) and target Series (y).
    """
    if not x_cols:
        X_df = df.drop(columns=["rent"])
    if x_cols:
        X_df = df[x_cols]

    y_df = df.rent

    return X_df, y_df


def _train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> RandomForestRegressor:
    """
    Trains a RFRegressor model using grid search for parameter tuning.

    This function performs a grid search over the specified hyperparameters
    to find the best model. The grid search is performed using cross-validation
    to ensure that the model is robust and generalizes well to unseen data.
    The best model is then returned.

    Args:
        X_train (pd.DataFrame): Training features DataFrame.
        y_train (pd.DataFrame): Training target Series.

    Returns:
        RandomForestRegressor: The trained model.
    """
    grid_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9, 12, 15],
    }
    logger.info("Training model with grid search...")
    logger.debug(f"Grid search space: {grid_space}")
    grid_search = GridSearchCV(
        RandomForestRegressor(), grid_space, cv=5, scoring="r2"
    )
    model_grid = grid_search.fit(X_train, y_train)

    best_model = model_grid.best_estimator_
    logger.debug(f"Best model params: {model_grid.best_params_}")

    return best_model


def _evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame
):
    score = model.score(X_test, y_test)
    return score


def _save_model(model: RandomForestRegressor, model_name: str) -> None:
    # could add os create dir logic here
    with open(f"{settings.model_path}/{model_name}", "wb") as writer:
        pk.dump(model, writer)


def build_model(
    x_cols: list | None = None,
    model_name: str = settings.model_name,
    train_size: float = 0.8,
    test_size: float = 0.2,
):
    # load data
    df = prepare_data()

    # split vars and target
    X_df, y_df = _get_X_y(df, x_cols)

    # get tts
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, train_size=train_size, test_size=test_size
    )

    # train
    logger.info("Training model...")
    rf = _train_model(X_train, y_train)

    # evaluate
    score = _evaluate_model(rf, X_test, y_test)
    logger.info(f"Model score: {score}")

    # save
    _save_model(rf, model_name=model_name)
    logger.info(f"Model saved to {settings.model_path}/{model_name}")

    return score


if __name__ == "__main__":
    score = build_model()
