import pickle as pk

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from config import settings
from model.preparation import prepare_data


def get_X_y(df: pd.DataFrame, x_cols: list | None = None) -> tuple:
    if not x_cols:
        X_df = df.drop(columns=["rent"])
    if x_cols:
        X_df = df[x_cols]

    y_df = df.rent

    return X_df, y_df


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> RandomForestRegressor:
    grid_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9, 12, 15],
    }
    logger.info("Training model with grid search...")
    logger.debug(f"Grid search space: {grid_space}")
    g = GridSearchCV(RandomForestRegressor(), grid_space, cv=5, scoring="r2")
    model_grid = g.fit(X_train, y_train)

    best_model = model_grid.best_estimator_
    logger.debug(f"Best model params: {model_grid.best_params_}")

    return best_model


def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame
):
    score = model.score(X_test, y_test)
    return score


def save_model(model: RandomForestRegressor, model_name: str) -> None:
    # could add os create dir logic here
    pk.dump(model, open(f"{settings.model_path}/{model_name}", "wb"))


def build_model(
    x_cols: list | None = None, model_name: str = settings.model_name
):
    # load data
    df = prepare_data()

    # split vars and target
    X_df, y_df = get_X_y(df, x_cols)

    # get tts
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, train_size=0.8, test_size=0.2
    )

    # train
    logger.info("Training model...")
    rf = train_model(X_train, y_train)

    # evaluate
    score = evaluate_model(rf, X_test, y_test)
    logger.info(f"Model score: {score}")

    # save
    save_model(rf, model_name=model_name)
    logger.info(f"Model saved to {settings.model_path}/{model_name}")

    return score


if __name__ == "__main__":
    score = build_model()
