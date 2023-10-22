import mlflow
from datetime import datetime

# Start MLFLow server like this (replace path with mlruns folder in outputs folder):
# mlflow server --backend-store-uri file:C:\Users\steng\Github\darts-forecasting-pipeline\outputs\mlruns


def start_run(cfg) -> None:
    """
    Initializes an MLflow run using the given configuration.

    Parameters:
    - cfg (Config object): The configuration object containing experiment and model details.

    Returns:
    None
    """
    tracking_uri = f"file:{cfg.mlflow.tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    run_name = cfg.models["model"]["_target_"].split(".")[-1]
    mlflow.start_run(
        run_name=(f'{run_name}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    )


def end_run(cfg, results) -> None:
    """
    Logs the results and model parameters to the current MLflow run and then ends the run.

    Parameters:
    - cfg (Config object): The configuration object containing model details.
    - results (dict): Dictionary containing the results/metrics to be logged.

    Returns:
    None
    """
    for key, value in results.items():
        mlflow.log_metric(key, value)

    model_params = dict(cfg.models.model)
    del model_params["_target_"]
    for key, value in model_params.items():
        mlflow.log_param(key, value)

    mlflow.log_dict(model_params, "model_parameters.json")
    model_name = cfg.models.model._target_.split(".")[-1]
    mlflow.log_param("model_name", model_name)
    mlflow.end_run()
