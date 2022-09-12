import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

    df_1 = pd.read_csv("data/2021_PT_Region_Mobility_Report.csv")
    df_2 = pd.read_csv("data/2022_PT_Region_Mobility_Report.csv")

    df = pd.concat([df_1, df_2])

    return df


@task
def prepare_data(df):
    df = clean_data(df)
    df = feature_extraction(df)
    df = df.dropna()

    return df


@task
def model_train(df):
    X = df.drop(columns=["y"])
    y = df.y

    mlflow.set_experiment("retail_forecasting_dev")

    with mlflow.start_run():

        model = LinearRegression()

        model.fit(X, y)

        mlflow.sklearn.log_model(model, artifact_path="model")

    mlflow.end_run()


@flow
def main():
    df = get_data()
    df = prepare_data(df)

    model_train(df)


if __name__ == "__main__":
    main()
