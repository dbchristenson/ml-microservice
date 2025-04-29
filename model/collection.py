import pandas as pd

from config import settings


def load_data(path=settings.data_file_name) -> pd.DataFrame:
    apartment_df = pd.read_csv(path)
    return apartment_df


# test
if __name__ == "__main__":
    df = load_data()
    print(df)
