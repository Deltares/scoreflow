"""Read and write netcdf files in a fews compatible format."""

from pathlib import Path

import xarray as xr

from dpyverification.configuration import (
    CFCompliantNetCDFConfig,
)
from dpyverification.datasinks.base import BaseDatasink


class CFCompliantNetCDF(BaseDatasink):
    """For writing data to a fews netcdf file."""

    kind = "cf_compliant_netcdf"
    config_class = CFCompliantNetCDFConfig

    def __init__(self, config: CFCompliantNetCDFConfig) -> None:
        self.config: CFCompliantNetCDFConfig = config

    def write_data(self, dataset: xr.Dataset) -> None:
        """Write the data in the xarray Dataset to the file as specified in the output config."""
        filepath = Path(self.config.directory) / self.config.filename
        if filepath.exists() and self.config.force_overwrite is False:
            msg = "File already exists: " + str(filepath)
            raise FileExistsError(msg)
        dataset.to_netcdf(filepath)
