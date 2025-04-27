import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer as MLB


def handle_garden_column(x: pd.Series) -> int:
    """Handles the 'garden' column in the dataset"""
    if x == "Not present":
        return 0

    return int(re.findall(r"\d+", x)[0])


def handle_neighborhood_column(x: pd.Series) -> str:
    """Handles the 'neighborhood' column in the dataset"""
    return x.split()[0]


def clean_facility(tokens: str) -> list:
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


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares dataset for machine learning training"""

    # encode simple bool columns
    cols_to_encode = ["balcony", "storage", "parking", "furnished", "garage"]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # garden
    df["garden"] = df.garden.map(handle_garden_column)

    # location
    df["neighborhood"] = df.neighborhood.map(handle_neighborhood_column)
    df = pd.get_dummies(df, columns=["neighborhood"], drop_first=True)
    df = df.drop(columns=["address", "zip"])  # non-interpretable locations

    # facilities
    facility_list = df.facilities.str.split(r",\s*")
    df["facility_list"] = facility_list.apply(lambda x: clean_facility(x))
    mlb = MLB(classes=get_cleaned_facility_list(df))
    fac_dummies = pd.DataFrame(
        mlb.fit_transform(df.facility_list),
        columns=mlb.classes_,
        index=df.index,
    )
    df = pd.concat([df, fac_dummies], axis=1)
    df["n_facilities"] = df.facility_list.str.len()
    df = df.drop(columns=["facility_list", "facilities"])

    # energy
    ordered_grades = get_energy_grades()
    cat_type = pd.api.types.CategoricalDtype(
        categories=ordered_grades, ordered=True
    )
    df["energy_cat"] = df.energy.astype(cat_type)  # convert and pull out codes
    df["energy_code"] = df.energy_cat.cat.codes.replace(-1, np.nan) + 1
    df["energy_missing"] = df.energy_code.isna().astype(int)
    median_code = df.energy_code.median()
    df.energy_code.fillna(median_code, inplace=True)
    df = df.drop(columns=["energy", "energy_cat"])

    return df
