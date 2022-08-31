import pandas as pd
import numpy as np


def clean_data(df):
    df.date = pd.to_datetime(df.date)
    df = df.set_index("date")
    df = df.loc[
        (df.country_region_code == "PT") & (df.sub_region_1.isnull()),
        "retail_and_recreation_percent_change_from_baseline",
    ].to_frame()
    df.columns = ["y"]

    return df


def feature_extraction(df):
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.day_of_week
    df["month"] = df.index.month
    nr_of_lags = 15
    for i in np.arange(7, nr_of_lags):
        df[f"lag_{i}"] = df.y.shift(i)

    return df
