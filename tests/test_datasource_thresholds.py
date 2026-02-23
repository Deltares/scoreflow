"""Test the threshold datasource."""

from pathlib import Path

import pandas as pd

from dpyverification.configuration.default.thresholds import CsvFileConfig
from dpyverification.datasources.thresholds import CsvFile


def test_fetch_thresholds_from_csv(dummy_threshold_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test we can fetch thresholds from csv file."""
    file_path = tmp_path / "thresholds.csv"
    dummy_threshold_df.to_csv(file_path, index=False)
    config = CsvFileConfig(
        kind="csv",
        directory=file_path.parent,
        filename=file_path.name,
        stations=["station_2"],
        variables=["variable_1"],
        thresholds=["warn_1"],
    )
    instance = CsvFile(config)
    instance.fetch_data()
