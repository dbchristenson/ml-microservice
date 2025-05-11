import re

import numpy as np  # noqa E401
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer as MLB  # noqa E401

from src.model.pipeline.collection import load_data_from_db


def handle_garden_column(x: pd.Series) -> int:
    """Handles the 'garden' column in the dataset"""
    if x == "Not present":
        return 0

    return int(re.findall(r"\d+", x)[0])


def handle_neighborhood_column(x: pd.Series) -> str:
    """Handles the 'neighborhood' column in the dataset"""
    return x.split()[0]


def clean_fac(tokens: str) -> list:
    """Gets a clean, normalized list of facilities on the property"""
    if isinstance(tokens, float):
        return []

    res = []

    for token in tokens:
        token = token.strip().lower()
        # turn spaces or hyphens into underscores
        token = re.sub(r"[\s\-]+", "_", token)
        res.append(token)

    return res


def get_cleaned_facility_list(df: pd.DataFrame):
    """Gets a cleaned set of all unique facilities in the data"""
    facilities_set = set()

    for s in df.facilities.values:
        try:
            s = s.lower()
        except AttributeError:
            continue

        facility_list = s.split(",")
        for facility in facility_list:
            facilities_set.add(facility.strip())

    cleaned_facilities = []
    for facility in facilities_set:
        cleaned_s = facility.replace(" ", "_")
        cleaned_s = cleaned_s.replace("-", "_")

        cleaned_facilities.append(cleaned_s)

    return cleaned_facilities


def get_energy_grades():
    """Gets an ordered list of energy grades"""
    ordered_grades = [
        "G",
        "F",
        "E",
        "D",
        "C",
        "B",
        "A",
        "A+",
        "A++",
        "A+++",
        "A++++",
    ]

    return ordered_grades


def prepare_data() -> pd.DataFrame:
    """Prepares dataset for machine learning training"""
    logger.info("Preparing data for model training...")
    df = load_data_from_db()

    # boolean columns → sparse one-hots
    logger.info("Converting boolean columns to one-hot encoding...")
    bools = ["balcony", "storage", "parking", "furnished", "garage"]
    df = pd.get_dummies(df, columns=bools, drop_first=True, dtype="int8")

    # garden
    logger.info("Converting garden column to int...")
    df["garden"] = df.garden.map(handle_garden_column)

    """
    This code is commented out because it highly complicates the inputs
    required from users to give the model to make a prediction. This would
    mean a more robust way of passing data to the model is required. For
    example passing a url to the apartment listing.
    # location
    df["neighborhood"] = df.neighborhood.map(handle_neighborhood_column)
    df = pd.get_dummies(df, columns=["neighborhood"], drop_first=True)
    df = df.drop(columns=["address", "zip"])  # non-interpretable locations
    """

    """
    This functionality is commented out for the exact same reason as above.
    # facilities → sparse MLb
    df["facility_list"] = df.facilities.str.split(r"REEXPHERE").apply(clean_fac) # noqa W605
    mlb = MLB(sparse_output=True)  # scikit-learn ≥1.4
    fac_sparse = mlb.fit_transform(df.facility_list)
    fac_df = pd.DataFrame.sparse.from_spmatrix(
        fac_sparse, index=df.index, columns=mlb.classes_
    )
    df = pd.concat([df, fac_df], axis=1)
    df["n_facilities"] = df.facility_list.str.len()
    df.drop(columns=["facility_list", "facilities"], inplace=True)
    """

    """
    This functionality is commented out for the exact same reason as above.
    # energy
    ordered_grades = get_energy_grades()
    cat_type = pd.api.types.CategoricalDtype(
        categories=ordered_grades, ordered=True
    )
    df["energy_cat"] = df.energy.astype(cat_type)  # convert and pull out codes
    df["energy_code"] = df.energy_cat.cat.codes.replace(-1, np.nan) + 1
    df["energy_missing"] = df.energy_code.isna().astype(int)
    median_code = df.energy_code.median()
    df["energy_code"] = df.energy_code.fillna(median_code)
    df = df.drop(columns=["energy", "energy_cat"])
    """

    # drop unecessary columns
    logger.info("Dropping unnecessary columns...")
    df = df.drop(
        columns=["energy", "facilities", "neighborhood", "address", "zip"]
    )

    return df
