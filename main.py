import hydra
from omegaconf import DictConfig
import numpy as np
from codes.utility.mlflow_utils import *
from codes.utility.evaluator import evaluator
import sys
import numpy as np
from optuna.trial import Trial

# from optuna.integration import PyTorchLightningPruningCallback
# from pytorch_lightning.callbacks import EarlyStopping

import mlflow

HYDRA_FULL_ERROR = 1


@hydra.main(
    version_base="1.2",
    config_path=r"C:\Users\steng\Github\darts-forecasting-pipeline\configs",
    config_name="config",
)
def main(cfg: DictConfig):
    start_run(cfg)

    dataset = hydra.utils.instantiate(cfg.datasets.dataset)
    y, y_scaler, train, test, covariates = dataset.return_dataset()

    model = hydra.utils.instantiate(cfg.models.model)

    model.fit(series=train, past_covariates=covariates)

    forecasts = model.predict(
        series=train, past_covariates=covariates, n=cfg.forecast_horizon
    )

    results = evaluator(cfg, test, forecasts)
    result = results[cfg.parameter]

    if not np.isfinite(results[cfg.parameter]):
        result = float("inf")

    end_run(cfg, results)

    print(f"This trials result is: {result}")

    return result


if __name__ == "__main__":
    main()
