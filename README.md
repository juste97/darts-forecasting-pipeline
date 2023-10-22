[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![image](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)


# ❗ WORK IN PROGRESS ❗- Pipeline for Forecasting with Darts 📈

## Overview 👀

This repository contains code for (probabilistic) forecasting experiments using the Rossmann sales dataset with the Python library Darts in combination with Hydra, Optuna optimizer and MLFlow.

### [Darts](https://unit8co.github.io/darts/)

"Darts is a Python library for user-friendly forecasting and anomaly detection on time series. It contains a variety of models, from classics such as ARIMA to deep neural networks. The forecasting models can all be used in the same way, using fit() and predict() functions, similar to scikit-learn. The library also makes it easy to backtest models, combine the predictions of several models, and take external data into account. Darts supports both univariate and multivariate time series and models. The ML-based models can be trained on potentially large datasets containing multiple time series, and some of the models offer a rich support for probabilistic forecasting."

#### [Hydra](https://hydra.cc/)

"Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads."

#### [Optuna](https://optuna.org/)

"Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters."

#### [MLflow](https://mlflow.org/)

MLflow is a platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models. MLflow offers a set of lightweight APIs that can be used with any existing machine learning application or library (TensorFlow, PyTorch, XGBoost, etc), wherever you currently run ML code (e.g. in notebooks, standalone applications or the cloud).

## Dataset 💾
The dataset used in this project is the Rossmann Store Sales data from Kaggle:
https://www.kaggle.com/competitions/rossmann-store-sales

## Plots 🖼
![mlflow](https://github.com/juste97/darts-forecasting-pipeline/blob/main/data/plots/mlflow.png?raw=true)
![Forecast: Store 1](https://github.com/juste97/darts-pipeline/blob/main/data/plots/prob_forecast_store_1.png?raw=true)
