from darts.metrics import *
import numpy as np


def evaluator(cfg, test_sets, forecasts):
    smapes = smape(test_sets, forecasts, n_jobs=-1, verbose=False)
    maes = mae(test_sets, forecasts, n_jobs=-1, verbose=False)

    # TODO: Loop over metrics dict and replace values with mean
    smapes = np.mean(smapes)
    maes = np.mean(maes)

    metrics = {
        "SMAPE": smapes,
        "MAE": maes,
    }

    return metrics
