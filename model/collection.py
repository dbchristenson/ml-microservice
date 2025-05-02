import pandas as pd
from loguru import logger
from sqlalchemy import select

from config import engine, settings
from dbmodel import RentApartments


def load_data(path=settings.data_file_name) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    logger.info("Loading data from path: {}", path)
    apartment_df = pd.read_csv(path)
    return apartment_df


def load_data_from_db() -> pd.DataFrame:
    """
    Load data from a SQLite database.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    logger.info("Loading data from SQLite database")
    query = select(RentApartments)
    return pd.read_sql(query, engine)


# test
if __name__ == "__main__":
    df = load_data()
    print(df)
