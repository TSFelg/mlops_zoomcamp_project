import urllib.request
import zipfile
import pandas as pd
import numpy as np
from utils import *
import mlflow
from mlflow.tracking import MlflowClient
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context
from datetime import datetime
from pathlib import Path
from google.cloud import storage


@task
def get_data():
    url = "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip"
    extract_dir = "data/raw"

    zip_path, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(extract_dir)

    df = pd.read_csv("data/raw/2022_PT_Region_Mobility_Report.csv")

    return df


@task
def write_predictions(df):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("project-predictions")  # your bucket name

    data_path = Path("data")
    output_path = Path(
        data_path / "predictions" / datetime.today().strftime("%Y-%m-%d")
    )
    df.to_csv(output_path)

    blob = bucket.blob(str(output_path))
    blob.upload_from_filename(str(output_path))

    return df


@flow
def main(run_date=None):
    df = get_data()
    df = clean_data(df)

    for i in np.arange(1, 7):
        last_day = df.iloc[-1:].index + pd.DateOffset(1)
        df = pd.concat([df, pd.DataFrame(index=last_day)])

    df = feature_extraction(df)
    df = df.drop(columns=["y"])

    df = df.iloc[-2:]

    client = MlflowClient()
    runs = client.search_runs(experiment_ids="5")
    run = runs[0]
    RUN_ID = run.info.run_id
    logged_model = f"gs://mlflow-mlops/5/{RUN_ID}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)

    df["y"] = model.predict(df)
    df["run_id"] = RUN_ID

    df = write_predictions(df)


if __name__ == "__main__":
    main()
