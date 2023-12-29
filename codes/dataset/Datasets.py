import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from sklearn.preprocessing import MaxAbsScaler
import fastparquet
import pyarrow
import pandas as pd


class TimeSeriesDataLoader:
    """
    Handles loading, splitting, and preparing time series data for forecasting with Darts.
    """

    def __init__(
        self,
        path,
        time_col,
        group_cols,
        value_col,
        features,
        frequency,
        train_ratio=0.6,
        test_ratio=0.2,
        validation_ratio=0.2,
    ):
        """
        Initializes the TimeSeriesDataLoader with data source and parameters for processing.

        Args:
            path (str): Path to the dataset file.
            time_col (str): Name of the column containing time data (needs to be in pd.DateTime).
            group_cols (str or list): Column name(s) defining groups in the dataset.
            value_col (str): Name of the target variable column.
            features (list): List of feature column names.
            frequency (str): Frequency of the time series data.
            train_ratio (float, optional): Proportion of data to be used for training. Defaults to 0.6.
            test_ratio (float, optional): Proportion of data for testing. Defaults to 0.2.
            validation_ratio (float, optional): Proportion of data for validation. Defaults to 0.2.
        """
        self.path = path
        self.time_col = time_col
        self.group_cols = group_cols
        self.value_col = value_col
        self.features = features
        self.frequency = frequency
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio

    def load_data(self):
        """
        Loads the dataset from the specified path and converts the time column to datetime format.
        """
        self.dataset = pd.read_parquet(self.path)
        self.dataset[self.time_col] = pd.to_datetime(self.dataset[self.time_col])

    def split_data(self):
        """
        Splits the dataset into training, validation, and test sets based on specified ratios.
        It calculates the split dates based on the total duration of the dataset and creates TimeSeries objects for the target and covariate data.
        """
        total_days = (
            self.dataset[self.time_col].max() - self.dataset[self.time_col].min()
        ).days
        train_end = self.dataset[self.time_col].min() + pd.to_timedelta(
            int(total_days * self.train_ratio), unit="d"
        )
        val_start = train_end - pd.to_timedelta(1, unit="d")
        val_end = val_start + pd.to_timedelta(
            int(total_days * self.validation_ratio), unit="d"
        )
        test_start = val_end - pd.to_timedelta(1, unit="d")

        target = TimeSeries.from_group_dataframe(
            self.dataset,
            time_col=self.time_col,
            group_cols=[self.group_cols],
            value_cols=[self.value_col],
            freq=self.frequency,
        )

        self.covariates = TimeSeries.from_group_dataframe(
            self.dataset,
            time_col=self.time_col,
            group_cols=[self.group_cols],
            value_cols=self.features,
            freq=self.frequency,
        )

        self.train_series = [series.drop_after(train_end) for series in target]
        self.validation_series = [
            series.drop_before(val_start).drop_after(val_end) for series in target
        ]
        self.test_series = [series.drop_before(test_start) for series in target]

    def scale_data(self):
        """
        Scales the train, validation, and test series using the Max Absolute Scaler.
        Applies scaling separately to the target series and covariates, fitting the scaler on the training data.
        """
        self.target_scaler = Scaler(scaler=MaxAbsScaler(), global_fit=False).fit(
            self.train_series
        )
        self.train_series_scaled = self.target_scaler.transform(self.train_series)
        self.validation_series_scaled = self.target_scaler.transform(
            self.validation_series
        )
        self.test_series_scaled = self.target_scaler.transform(self.test_series)

        self.covariates_scaler = Scaler(scaler=MaxAbsScaler(), global_fit=False).fit(
            [
                series.drop_after(self.train_series[0].time_index.max())
                for series in self.covariates
            ]
        )
        self.covariates_scaled = self.covariates_scaler.transform(self.covariates)

    def get_data(self) -> tuple:
        """
        Starts the loading, splitting, and scaling of the dataset.

        Returns:
            tuple: Contains scaled training, validation, and test series, along with the scalers for target and covariates.
        """
        self.load_data()
        self.split_data()
        self.scale_data()

        return (
            self.train_series_scaled,
            self.validation_series_scaled,
            self.test_series_scaled,
            self.covariates_scaled,
            self.target_scaler,
            self.covariates_scaler,
        )
