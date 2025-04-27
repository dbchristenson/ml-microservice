import pandas as pd


def load_data(path="data/rent_apartments.csv"):
    apartment_df = pd.read_csv(path)
    return apartment_df


# test
df = load_data()
print(df)
