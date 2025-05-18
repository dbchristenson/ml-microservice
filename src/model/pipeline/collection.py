import pandas as pd
from loguru import logger
from sqlalchemy import select

from src.config.config import engine
from src.db.db_model import RentApartments


def load_data_from_db() -> pd.DataFrame:
    """
    Load data from a SQLite database.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    logger.info("Loading data from SQLite database")
    query = select(RentApartments)
    return pd.read_sql(query, engine)
