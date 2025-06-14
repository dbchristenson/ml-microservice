"""
This module contains functions to prepare the dataset for machine learning.
It includes functions to handle specific columns, clean facility names,
and prepare the data for training. The functions are designed to be used
in a pipeline to ensure that the data is in the correct format and
ready for modeling.
"""

import re

import pandas as pd
from loguru import logger

from src.model.pipeline.collection import load_data_from_db


def _handle_garden_column(x_series: pd.Series) -> int:
    """
    Handles the 'garden' column in the dataset.

    Args:
        x_series (pd.Series): The value of the 'garden' column.

    Returns:
        int: 1 if the garden is present, 0 otherwise.
    """
    if x_series == "Not present":
        return 0

    return int(re.findall(r"\d+", x_series)[0])


def _handle_neighborhood_column(x_series: pd.Series) -> str:
    """
    Handles the 'neighborhood' column in the dataset.

    Args:
        x_series (pd.Series): The value of the 'neighborhood' column.

    Returns:
        str: The neighborhood name.
    """
    return x_series.split()[0]


def _clean_fac(tokens: str) -> list:
    """
    Gets a clean, normalized list of facilities on the property.

    Args:
        tokens (str): The string of facilities.

    Returns:
        list: A list of cleaned facility names.
    """
    if isinstance(tokens, float):
        return []

    res = []

    for token in tokens:
        token = token.strip().lower()
        # turn spaces or hyphens into underscores
        token = re.sub(r"[\s\-]+", "_", token)
        res.append(token)

    return res


def _get_cleaned_facility_list(df: pd.DataFrame) -> list:
    """
    Gets a cleaned set of all unique facilities in the data

    Args:
        df (pd.DataFrame): The DataFrame containing the facilities data.

    Returns:
        list: A list of cleaned facility names.
    """
    facilities_set = set()

    for str in df.facilities.values:
        try:
            str = str.lower()
        except AttributeError:
            continue

        facility_list = str.split(",")
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
    """
    Prepares dataset for machine learning training

    This function loads data from a SQLite database, processes the data,
    and returns a cleaned DataFrame ready for model training. It handles
    boolean columns, garden information, and drops unnecessary columns.
    It also includes commented-out sections for additional processing
    that may be added in the future.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for model training.
    """
    logger.info("Preparing data for model training...")
    df = load_data_from_db()

    # boolean columns → one-hot encoding
    logger.info("Converting boolean columns to one-hot encoding...")
    bools = ["balcony", "storage", "parking", "furnished", "garage"]
    df = pd.get_dummies(df, columns=bools, drop_first=True, dtype="int8")

    # garden
    logger.info("Converting garden column to int...")
    df["garden"] = df.garden.map(_handle_garden_column)

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
