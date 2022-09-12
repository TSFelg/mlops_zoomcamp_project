# MLOps Zoomcamp Project

## Overview

This project consists of an application that forecasts the mobility trends for retail & recreation in Portugal using  [Google's Community Mobility Reports](https://www.google.com/covid19/mobility). According to Google, this includes places like restaurants, cafes, shopping centers, theme parks, museums, libraries, and movie theaters.

This project aims to be a proof of concept for an application that allows restaurant owners to have better predictions on consumer demand.

Currently, the model only leverages past mobility data but in the future the objective is to integrate other data sources such as holidays and weather forecasts.

The main focus of the project is on the operations side which is what the course mainly covers. As can be seen on the diagram below, the project leverages Prefect as an orchestration server to run the predictions everyday in the morning for the following days which are stored on a bucket. The model is automatically re-trained on a monthly basis to adjust for distribution shifts. Model training is integrated with MLFlow for tracking and registry. All of these operations run on GCP.

![diagram](resources/diagram.png)

