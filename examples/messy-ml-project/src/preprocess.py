# preprocessing utils
# copied from stackoverflow honestly

import pandas as pd


def clean_data(df):
    """remove nulls and stuff"""
    df = df.dropna()
    # remove obvious outliers (just eyeballing it)
    df = df[df["price"] > 50000]
    df = df[df["price"] < 1000000]
    return df


def add_features(df):
    """some extra features that might help idk"""
    df["price_per_sqft"] = df["price"] / df["sqft"]
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/houses.csv")
    df = clean_data(df)
    df = add_features(df)
    print(df.head())
    print(f"\n{len(df)} rows after cleaning")
