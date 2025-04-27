import pickle as pk

import pandas as pd
from preparation import prepare_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split


def get_X_y(df: pd.DataFrame, x_cols: list | None = None) -> tuple:
    X_df, y_df = get_X_y(x_cols)

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
    g = GridSearchCV(RandomForestRegressor(), grid_space, cv=5, scoring="r2")
    model_grid = g.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    return best_model


def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame
):
    score = model.score(X_test, y_test)
    return score


def save_model(model: RandomForestRegressor) -> None:
    # could add os create dir logic here
    pk.dump(model, open("models/rf_v1", "wb"))


def build_model(x_cols: list | None = None):
    # load data
    print("loading data")
    df = prepare_data()

    # split vars and target
    print("splitting data")
    X_df, y_df = get_X_y(df, x_cols)

    # get tts
    print("getting tts")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, train_size=0.8, test_size=0.2
    )

    # train
    print("training")
    rf = train_model(X_train, y_train)

    # evaluate
    score = evaluate_model(rf, X_test, y_test)
    print("Model Score: ", score)

    # save
    save_model(rf)

    return score


if __name__ == "__main__":
    print("building model")
    score = build_model()
    print(f"model score: {score}")
