"""Datasources to fetch thresholds."""

from pathlib import Path
from typing import Self

import pandas as pd
import xarray as xr

from dpyverification.configuration.default.thresholds import CsvFileConfig
from dpyverification.constants import StandardDim
from dpyverification.datasources.base import BaseThresholdsDatasource


class CsvFile(BaseThresholdsDatasource):
    """Parse thresholds from a csv file."""

    kind: str = ""
    config_class = CsvFileConfig

    def __init__(self, config: CsvFileConfig) -> None:
        self.config: CsvFileConfig = config
        self.data_array = xr.DataArray()

    def fetch_data(self) -> Self:
        """Parse thresholds from csv file."""
        file_path = Path(self.config.directory) / self.config.filename
        threshold_df = pd.read_csv(file_path)

        # Check that the df has the correct structure
        expected_columns = [
            StandardDim.station,
            StandardDim.variable,
            StandardDim.threshold,
            "value",
        ]
        if not all(k in expected_columns for k in threshold_df.columns):
            msg = f"Expected columns: {expected_columns}. Got: {threshold_df.columns}"
            raise ValueError(msg)

        # Convert it to the internal datamodel
        data_array = threshold_df.set_index(
            [StandardDim.station, StandardDim.variable, StandardDim.threshold],
        ).to_xarray()["value"]
        data_array.attrs["timeseries_kind"] = "thresholds"  # type:ignore[misc]
        self.data_array = data_array
        return self
