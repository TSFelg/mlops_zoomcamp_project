import urllib.request
import zipfile
import pandas as pd
import numpy as np
from utils import *
import mlflow
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context


@task
def get_data():
    url = "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip"
    extract_dir = "data"

    zip_path, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(extract_dir)

    df = pd.read_csv("data/2022_PT_Region_Mobility_Report.csv")

    return df


@flow
def main(run_date=None):
    df = get_data()
    df = clean_data(df)

    for i in np.arange(1, 7):
        last_day = df.iloc[-1:].index + pd.DateOffset(1)
        df = df.append(pd.DataFrame(index=last_day))

    df = feature_extraction(df)
    df = df.drop(columns=["y"])

    df = df.iloc[-2:]

    RUN_ID = "52bf59cb151d46ac8827defb5ffa8363"
    logged_model = f"gs://mlflow-mlops/4/{RUN_ID}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)

    pred = model.predict(df)

    print(pred)


if __name__ == "__main__":
    main()
