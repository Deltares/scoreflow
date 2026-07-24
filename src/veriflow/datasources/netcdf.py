"""Read and write NetCDF files in a fews compatible format."""

from typing import ClassVar, Self

import xarray as xr

from veriflow.configuration.default.datasources import NetCDFConfig
from veriflow.constants import (
    DataType,
)
from veriflow.datasources.base import BaseDatasource

__all__ = [
    "NetCDF",
    "NetCDFConfig",
]


class NetCDF(BaseDatasource):
    """A datasource for reading NetCDF files compatible with the internal datamodel.

    You can validate that the NetCDF file satisfies the internal datamodel using the following
    example:

    .. code-block:: python

        import xarray as xr
        from veriflow.datasources import validate_input_data

        dataset = xr.open_dataset("path/to/netcdf/file/example.nc")
        validated_data = validate_input_data(dataset)

    .. note::
        The data variables in the NetCDF file represent the physical variables to be verified.
        Each data variable should carry a ``units`` attribute. The dataset must carry a
        ``data_type`` attribute that matches one of the supported data types (it will be set
        from the configuration if missing).
    """

    kind = "netcdf"
    config_class = NetCDFConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.observed_historical,
        DataType.simulated_forecast_ensemble,
        DataType.simulated_forecast_single,
        DataType.simulated_forecast_probabilistic,
        DataType.threshold,
    }

    def __init__(self, config: NetCDFConfig) -> None:
        self.config: NetCDFConfig = config

    @property
    def configured_stations(self) -> set[str] | None:
        """Return the internal station identifiers configured for this datasource.

        This is needed for standardization of station identifiers across sources.
        """
        return set(self.config.stations) if self.config.stations is not None else None

    @configured_stations.setter
    def configured_stations(self, stations: set[str]) -> None:
        """Set the internal station identifiers configured for this datasource.

        This is needed for standardization of station identifiers across sources.
        """
        self.config.stations = list(stations)

    @property
    def configured_variables(self) -> set[str] | None:
        """Return the internal variable identifiers configured for this datasource.

        This is needed for standardization of variable identifiers across sources.
        """
        return set(self.config.variables) if self.config.variables is not None else None

    @configured_variables.setter
    def configured_variables(self, variables: set[str]) -> None:
        """Set the internal variable identifiers configured for this datasource.

        This is needed for standardization of variable identifiers across sources.
        """
        self.config.variables = list(variables)

    def fetch_data(self) -> Self:
        """Retrieve NetCDF file content as an xarray Dataset."""
        dataset = xr.open_mfdataset(self.config.paths)  # type:ignore[arg-type] # Generator is accepted by open_mfdataset, but not correctly typed in xarray
        dataset.attrs["data_type"] = self.config.data_type  # type: ignore[misc]
        self.dataset = dataset
        return self
