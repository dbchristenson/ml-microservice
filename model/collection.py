import pandas as pd
from loguru import logger

from config import settings


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


# test
if __name__ == "__main__":
    df = load_data()
    print(df)
