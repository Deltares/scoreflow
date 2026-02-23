"""Module with the base class that all datasources should inherit from."""

import hashlib
from abc import abstractmethod
from os import R_OK, access
from typing import ClassVar, Self

import xarray
import xarray as xr

from dpyverification.base import Base
from dpyverification.configuration.config import (
    BaseThresholdsDatasourceConfig,
    BaseTimeseriesDatasourceConfig,
)
from dpyverification.configuration.utils import TimePeriod
from dpyverification.constants import FORECAST_TIMESERIES_KINDS, StandardDim, TimeseriesKind


class BaseTimeseriesDatasource(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind: str = ""
    config_class: type[BaseTimeseriesDatasourceConfig] = BaseTimeseriesDatasourceConfig
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = set()

    def __init__(self, config: BaseTimeseriesDatasourceConfig) -> None:
        self.config: BaseTimeseriesDatasourceConfig = config
        self.timeseries_kind = config.timeseries_kind
        self.data_array = xarray.DataArray()

    @property
    def timeseries_kind(self) -> str:
        """Whether the instance represents sim or obs data."""
        return self.config.timeseries_kind

    @timeseries_kind.setter
    def timeseries_kind(self, new_timeseries_kind: TimeseriesKind) -> None:
        if new_timeseries_kind not in self.supported_timeseries_kinds:
            msg = (
                f"Timeseries kind '{new_timeseries_kind}' is not supported ",
                f"by {self.__class__.__name__}",
            )
            raise NotImplementedError(msg)

        self._timeseries_kind = new_timeseries_kind

    @abstractmethod
    def fetch_data(self) -> Self:
        """Fetch data from datasource."""

    @staticmethod
    def _drop_times_outside_vp(
        da: xr.DataArray,
        verification_period_on_time: TimePeriod,
    ) -> xr.DataArray:
        """Mask times outside of verification period with inclusive endpoints."""
        # Mask values outside of verification period
        filtered = da.where(
            (da[StandardDim.time] >= verification_period_on_time.start_datetime64)
            & (da[StandardDim.time] <= verification_period_on_time.end_datetime64),
        )
        # Drop NaN values along frt and fp dims, if all values are NaN
        return filtered.dropna(dim=StandardDim.forecast_reference_time, how="all").dropna(
            dim=StandardDim.forecast_period,
            how="all",
        )

    def get_data(self) -> Self:
        """Get cached data, or fetch and cache."""
        config_json = self.config.model_dump_json().encode("utf-8")
        config_hash = hashlib.sha256(config_json).hexdigest()

        cache_dir = self.config.general.cache_dir

        # Create cache if not exists
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        # If it exists, check it's an accessible dir
        elif not cache_dir.is_dir() and access(cache_dir, R_OK):
            msg = "Cache directory is not an accessible directory."
            raise NotADirectoryError(msg)

        # Define file path for caching
        cached_data_array_path = cache_dir / f"{self.__class__.__name__}_{config_hash}.nc"

        if cached_data_array_path.exists():
            self.data_array = xr.open_dataarray(cached_data_array_path)
            return self

        # Go fetch and cache
        self.fetch_data()
        data_array_original = self.data_array

        # Make sure the name of the array is set to the configured source
        data_array_original.name = self.config.source

        # Apply re-naming based on configured id mapping, if not None
        if self.config.id_mapping is not None:
            data_array_original = self.config.id_mapping.rename_data_array(data_array_original)

        # Additional layer to filter time, frt and fp properly according to config.
        if data_array_original.attrs["timeseries_kind"] in FORECAST_TIMESERIES_KINDS:  # type:ignore[misc]
            # Select only relevant forecast periods for simulations
            data_array_original = data_array_original.sel(
                forecast_period=self.config.forecast_periods.timedelta64,
            )
            # Mask and drop time values outside of the configured vp
            data_array_original = self._drop_times_outside_vp(
                da=data_array_original,
                verification_period_on_time=self.config.verification_period_on_time,
            )
        else:
            # Historical timeseries kind
            data_array_original = data_array_original.sel(
                {
                    StandardDim.time: slice(  # type:ignore[misc]
                        self.config.verification_period_on_time.start,
                        self.config.verification_period_on_time.end,
                    ),
                },
            )

        # Cache
        data_array_original.to_netcdf(cached_data_array_path)

        # Re-open to read from cache and prevent links to original files from which the dataarray
        #   was loaded
        data_array_reloaded = xr.open_dataarray(cached_data_array_path)

        # Explicitly close original backing files
        if hasattr(data_array_original, "close"):
            data_array_original.close()

        # Re-assign from cache
        self.data_array = data_array_reloaded
        return self


class BaseThresholdsDatasource(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind: str = ""
    config_class: type[BaseThresholdsDatasourceConfig] = BaseThresholdsDatasourceConfig

    def __init__(self, config: BaseThresholdsDatasourceConfig) -> None:
        self.config: BaseThresholdsDatasourceConfig = config
        self.data_array = xarray.DataArray()

    @abstractmethod
    def fetch_data(self) -> Self:
        """Fetch data from datasource."""
