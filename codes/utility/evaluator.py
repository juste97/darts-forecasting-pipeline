from darts.metrics import *
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import numpy as np


class Evaluator:
    """
    Evaluator class for assessing predictions made by Darts.
    This class provides methods to evaluate the performance of forecasting models using various metrics.
    It supports evaluation in different modes (validation or test), allows the specification of an optimization metric (e.g., for Optuna) and plotting the forecasts.
    """

    def __init__(
        self,
        predictions: list,
        train_series: list,
        validation_series: list,
        test_series: list,
        mode: str,
        optimization_metric: str = "Mean Squared Error",
        verbose: bool = False,
    ):
        """
        Initializes the Evaluator object.

        Args:
            predictions (list): List of prediction objects.
            train_series (list): List of training series objects.
            validation_series (list): List of validation series objects.
            test_series (list): List of test series objects.
            mode (str): Mode of evaluation ('validate' or 'test').
            optimization_metric (str, optional): Metric used for optimization, defaults to "Mean Squared Error".
            verbose (bool, optional): Flag to enable verbose output, returns metrics as pd.DataFrames.

        Raises:
            ValueError: If the indices of the predictions and target series do not match.
        """

        self.predictions = predictions
        self.train_series = train_series
        self.validation_series = validation_series
        self.test_series = test_series
        self.mode = mode
        self.optimization_metric = optimization_metric
        self.verbose = verbose

        self.aggregated_metrics = pd.DataFrame()
        self.timestep_metrics = pd.DataFrame()

        if self.mode == "validate":
            self.target_series = self.validation_series
        elif self.mode == "test":
            self.target_series = self.test_series

        if self.predictions[0].time_index[0] != self.target_series[0].time_index[0]:
            raise ValueError(
                "The index of the predictions do not match with the corresponding target series for evaluations (check mode)."
            )

    def get_timestep_metrics(self) -> pd.DataFrame:
        """
        Computes and returns metrics for each timestep of each prediction entity.
        This method iterates over each entity in the predictions and creates a DataFrame with detailed metrics for each timestep.

        Returns:
            self.timestep_metrics (pd.DataFrame): A DataFrame containing detailed timestep metrics for each prediction entity.
        """

        dataframes_entity_metrics_timestep = []

        for index, entity in enumerate(self.predictions):
            entity_name = entity.static_covariates.iloc[0, 0]
            entity_df = pd.DataFrame(
                index=pd.date_range(
                    start=entity.time_index.min(), end=entity.time_index.max(), freq="D"
                )
            )
            for timestep_index, entity_preds in enumerate(entity):
                actual_values = self.target_series[index][timestep_index]
                timestep = entity_preds.time_index
                entity_df["Entity"] = entity_name
                entity_df.loc[timestep, ["Timestep"]] = timestep_index + 1

                # Actual Value
                entity_df.loc[timestep, ["Actual Value"]] = actual_values.all_values()[
                    0
                ][0][0]

                # Mean Forecast
                entity_df.loc[timestep, ["Mean Forecast"]] = np.mean(
                    entity_preds.all_values()[0][0]
                )

                # Quantile-based Forecasts
                quantiles = np.quantile(
                    entity_preds.all_values()[0][0],
                    [0.05, 0.25, 0.5, 0.75, 0.95],
                    axis=0,
                )
                entity_df.loc[
                    timestep,
                    [
                        "5th Percentile",
                        "25th Percentile",
                        "50th Percentile",
                        "75th Percentile",
                        "95th Percentile",
                    ],
                ] = quantiles

                # Error Metrics
                entity_df.loc[timestep, ["Mean Absolute Error"]] = mae(
                    actual_values, entity_preds, intersect=True
                )
                entity_df.loc[timestep, ["Root Mean Squared Error"]] = rmse(
                    actual_values, entity_preds, intersect=True
                )
                entity_df.loc[timestep, ["Root Mean Squared Log Error"]] = rmsle(
                    actual_values, entity_preds, intersect=True
                )
                entity_df.loc[
                    timestep, ["Symmetric Mean Absolute Percentage Error"]
                ] = smape(actual_values, entity_preds, intersect=True)

                # Specialized Error Metric
                entity_df.loc[timestep, ["Pinball Loss"]] = quantile_loss(
                    actual_values, entity_preds, intersect=True
                )

            dataframes_entity_metrics_timestep.append(entity_df)

        self.timestep_metrics = pd.concat(dataframes_entity_metrics_timestep, axis=0)
        self.timestep_metrics = self.timestep_metrics.set_index(
            ["Entity", self.timestep_metrics.index]
        )

        if self.verbose:
            return self.timestep_metrics

    def get_aggregated_metrics(self) -> pd.DataFrame:
        """
        Computes and returns aggregated metrics for each entity aggregated along the forecast horizon.
        This method calculates various error metrics and statistical measures for the predictions, such as Mean Absolute Error, Root Mean Squared Error, Symmetric Mean Absolute Percentage Error, and others.

        Returns:
            self.aggregated_metrics (pd.DataFrame): A DataFrame containing aggregated error metrics and statistical measures for the predictions.

        """
        index_names = [
            entity.static_covariates.iloc[0, 0] for entity in self.predictions
        ]
        self.aggregated_metrics["Entity"] = index_names

        # Forecast Specific Metric
        self.aggregated_metrics["Forecast Horizon"] = len(self.predictions[0])

        # Mean Error Metrics
        self.aggregated_metrics["Mean Absolute Error"] = mae(
            self.target_series, self.predictions, intersect=True
        )
        # self.aggregated_metrics['Mean Absolute Percentage Error'] = mape(self.target_series, self.predictions, intersect=True) # Commented out in your code
        self.aggregated_metrics["Mean Absolute Ranged Relative Error"] = marre(
            self.target_series, self.predictions, intersect=True
        )
        self.aggregated_metrics["Mean Absolute Scaled Error"] = mase(
            self.target_series,
            self.predictions,
            intersect=True,
            insample=self.train_series,
        )
        self.aggregated_metrics["Mean Squared Error"] = mse(
            self.target_series, self.predictions, intersect=True
        )

        # Root Mean Error Metrics
        self.aggregated_metrics["Root Mean Squared Error"] = rmse(
            self.target_series, self.predictions, intersect=True
        )
        self.aggregated_metrics["Root Mean Squared Log Error"] = rmsle(
            self.target_series, self.predictions, intersect=True
        )

        # Symmetric Error Metric
        self.aggregated_metrics["Symmetric Mean Absolute Percentage Error"] = smape(
            self.target_series, self.predictions, intersect=True
        )

        # Aggregate Error Metrics
        self.aggregated_metrics["Overall Percentage Error"] = ope(
            self.target_series, self.predictions, intersect=True
        )
        self.aggregated_metrics["Coefficient of Variation"] = coefficient_of_variation(
            self.target_series, self.predictions, intersect=True
        )
        self.aggregated_metrics["Dynamic Time Warping"] = dtw_metric(
            self.target_series, self.predictions
        )

        # Statistical Measure Metrics
        self.aggregated_metrics["Coefficient of Determination"] = r2_score(
            self.target_series, self.predictions, intersect=True
        )
        self.aggregated_metrics["Rho-Risk"] = rho_risk(
            self.target_series, self.predictions, intersect=True
        )

        # Specialized Error Metrics
        self.aggregated_metrics["Pinball Loss"] = quantile_loss(
            self.target_series, self.predictions, intersect=True
        )

        mean_values = self.aggregated_metrics.drop(columns=["Entity"]).mean()
        mean_row = pd.DataFrame([mean_values], index=["Mean"])
        mean_row["Entity"] = "Mean"
        self.aggregated_metrics = pd.concat([self.aggregated_metrics, mean_row])
        self.aggregated_metrics = self.aggregated_metrics.set_index("Entity", drop=True)

        if self.verbose:
            return self.aggregated_metrics

    def get_optimization_metric(self) -> int:
        """
        Retrieves the optimization metric's mean value from the aggregated metrics.

        Returns:
            metric (int): The mean value of the specified optimization metric.
        """
        if self.aggregated_metrics.empty:
            self.get_aggregated_metrics()

        metric = self.aggregated_metrics.loc["Mean", self.optimization_metric]

        print(f"\n{self.optimization_metric} is {metric}")

        return metric

    def plot_predictions(self, name: str = "", reduction_factor: int = None):
        """
        Plots the training, validation or test series along with the forecasted predictions.
        This method generates a plot for each entity in the predictions list.

        Args:
            name (str, optional): An optional string to prepend to the title of each plot.
            reduction_factor (str, optinal): Shortens the training series by len(training_series)/reduction_factor.
        """

        T = len(self.train_series[0])
        P = len(self.predictions[0])

        for index, _ in enumerate(self.predictions):
            plt.figure(figsize=(15, 5))

            if reduction_factor:
                self.train_series[index][-T // reduction_factor :].plot(
                    label="Train", lw=1
                )
            else:
                self.train_series[index].plot(label="Train", lw=1)

            if self.mode == "validate":
                self.validation_series[index][:P].plot(label="Validation", lw=1)
            elif self.mode == "test":
                self.validation_series[index].plot(label="Validation", lw=1)
                self.test_series[index][:P].plot(label="Test Series", lw=1)

            self.predictions[index].plot(label="Forecast", lw=1)

            plt.title(f"{name} {self.predictions[index].static_covariates.iloc[0, 0]}")
            plt.legend()

            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

            plt.subplots_adjust(right=0.75)

            if self.verbose:
                plt.show()
