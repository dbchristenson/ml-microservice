import pandas as pd


def load_data(path="data/rent_apartments.csv") -> pd.DataFrame:
    apartment_df = pd.read_csv(path)
    return apartment_df


# test
if __name__ == "__main__":
    df = load_data()
    print(df)
