import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from sklearn.preprocessing import MaxAbsScaler
import fastparquet
import pyarrow
import pandas as pd


class Datasets:
    def __init__(
        self,
        split_key: int,
        path: str,
        freq: str,
        group_cols: str,
        time_col: str,
        value_col: str,
    ) -> None:
        self.split_key = split_key
        self.path = path
        self.freq = freq
        self.time_col = time_col
        self.group_cols = [group_cols]
        self.value_col = value_col

    def read_data_from_path(self):
        """
        Reads data from the specified path. Infers the file type from the path extension.

        Returns:
        - pd.DataFrame: The data read from the file.
        """
        # Infer file type from the path extension
        file_type = self.path.split(".")[-1]

        if file_type == "xlsx":
            dataframe = pd.read_excel(self.path)
        elif file_type == "csv":
            dataframe = pd.read_csv(self.path)
        elif file_type == "parquet":
            dataframe = pd.read_parquet(self.path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return dataframe

    def extract_features_from_dataframe(self, dataframe):
        """
        Extracts feature columns from the dataframe.

        Returns:
        - list: List of feature column names.
        """
        exclude_cols = [self.time_col, self.value_col] + self.group_cols
        feature_cols = [col for col in dataframe.columns if col not in exclude_cols]
        return feature_cols

    def create_time_series(self, dataframe, features):
        """
        Creates a TimeSeries object for the target column and another for the features (covariates).

        Parameters:
        - dataframe (pd.DataFrame): The input dataframe containing the data.
        - features (list): List of feature column names.

        Returns:
        - tuple: A tuple containing two TimeSeries objects: (y, covariates).
        """
        y = TimeSeries.from_group_dataframe(
            dataframe,
            time_col=self.time_col,
            group_cols=self.group_cols,
            value_cols=self.value_col,
            freq=self.freq,
        )

        covariates = TimeSeries.from_group_dataframe(
            dataframe,
            time_col=self.time_col,
            group_cols=self.group_cols,
            value_cols=features,
            freq=self.freq,
        )

        return y, covariates

    def train_test_split(self, y, covariates):
        """
        Splits the time series into training and test sets, and scales them.

        Parameters:
        - y (TimeSeries): TimeSeries object for the target column.
        - covariates (TimeSeries): TimeSeries object for the features.

        Returns:
        - tuple: A tuple containing the original time series, scaler for y, scaled training and test sets,
                and scaled covariates.
        """
        y_train_list = [ts[: -self.split_key] for ts in y]
        y_test_list = [ts[-self.split_key :] for ts in y]

        y_scaler = Scaler(scaler=MaxAbsScaler(), global_fit=True)
        covariates_scaler = Scaler(scaler=MaxAbsScaler(), global_fit=True)

        train = y_scaler.fit_transform(y_train_list)
        test = y_scaler.transform(y_test_list)

        covariates = covariates_scaler.fit_transform(covariates)

        return y, y_scaler, train, test, covariates

    def return_dataset(self):
        """
        Reads data, extracts features, creates time series, splits the data into train and test sets,
        and scales them.

        Returns:
        - tuple: A tuple containing the original time series, scaler for y, scaled training and test sets,
                and scaled covariates.
        """

        print(
            "Reading data, extracting features, creating time series, applying train test split."
        )

        dataframe = self.read_data_from_path()
        feature_cols = self.extract_features_from_dataframe(dataframe)
        y, covariates = self.create_time_series(
            dataframe=dataframe, features=feature_cols
        )
        y, y_scaler, train, test, covariates = self.train_test_split(y, covariates)

        return y, y_scaler, train, test, covariates
