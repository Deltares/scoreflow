"""Zarr cache implementation.

This is a special datasource implementation that is used to cache datasets fetched from other
datasources. The cache is implemented as a Zarr store, and is designed to be flexible in
terms of the structure of the store and the types of data that can be cached. The cache is not
meant to be used directly by users, but rather as an internal helper class used in the get_data
method of the BaseDatasource abstract class.

Example structure:

veriflow-cache.zarr/

├── datasets/
        ecmwf_hres_wflow/
        |- discharge (station, forecast_reference_time, lead_time, realization) [chunk lead time]
        |- waterlevel (station, forecast_reference_time, lead_time, realization) [chunk lead time]
        .../
│   ├── observed
│	│	├── discharge (station, time)
│	│	├──	waterlevel (station, time)
│
├── evaluations/
│   ├── ecmwf_ens_wflow__vs__observed/
│   │   ├── metadata/
│   │   ├── crps/
│   │   ├── rank_histogram/
│   │
│   ├── ecmwf_hres_wflow__vs__observed/
│   │   └── metadata/
│	│	└──	rmse
"""

from typing import Literal, Self

import numpy as np
import xarray as xr
from pydantic import BaseModel, model_validator

from veriflow.cache.config import ReadWriteMode, ZarrCacheConfig
from veriflow.cache.utils import combine_cached_and_fetched_data
from veriflow.configuration.base import BaseDatasourceConfig
from veriflow.configuration.utils import LeadTimes, TimePeriod
from veriflow.constants import StandardDim, TimeUnits


class DataRequest(BaseModel):
    """Helper class to track requested and cached data and what data is missing from the cache."""

    requested: set[str] | TimePeriod | LeadTimes | None
    cached: set[str] | TimePeriod | LeadTimes | None

    @model_validator(mode="after")
    def validate_requested_and_cached(self) -> Self:
        """Validate that the requested and cached values are of the same type."""
        if type(self.requested) != type(self.cached):  # noqa: E721
            msg = "Requested and cached values must be of the same type."
            raise ValueError(msg)
        return self

    @staticmethod
    def find_missing_and_available_time_period(
        requested: TimePeriod,
        cached: TimePeriod,
    ) -> tuple[TimePeriod | None, TimePeriod | None]:
        """Find the missing time period between the requested and cached time periods.

        This helper function is used to determine what time period needs to be fetched from the
        datasource given what is already available in the cache.
        """
        # Requested fully contained in (or equal to) cached → nothing missing.
        if requested.start >= cached.start and requested.end <= cached.end:
            return None, requested
        #  RRRR
        #        CCCC
        if requested.end <= cached.start:
            return requested, None
        #        RRRR
        #  CCCC
        if requested.start >= cached.end:
            return requested, None
        #  RRRR
        #   CC      (cached strictly inside requested)
        if requested.start <= cached.start and requested.end >= cached.end:
            return requested, None
        # RRRR
        #   CCCC    (left overlap: requested starts before cached, ends inside cached)
        if requested.start < cached.start <= requested.end < cached.end:
            return TimePeriod(start=requested.start, end=cached.start), TimePeriod(
                start=cached.start,
                end=requested.end,
            )
        #   RRRR
        # CCCC      (right overlap: requested starts inside cached, ends after cached)
        if cached.start < requested.start <= cached.end < requested.end:
            return TimePeriod(start=cached.end, end=requested.end), TimePeriod(
                start=requested.start,
                end=cached.end,
            )
        msg = "Unexpected case of intersecting time periods."
        raise ValueError(msg)

    @property
    def get_missing_and_available(  # noqa: PLR0911
        self,
    ) -> tuple[set[str] | TimePeriod | LeadTimes | None, set[str] | TimePeriod | LeadTimes | None]:
        """Return the missing data from the cache."""
        if self.requested is None:
            return None, None
        if self.cached is None:
            return self.requested, None

        # Sets are expected to be used for variables, stations.
        if isinstance(self.requested, set) and isinstance(self.cached, set):
            missing_set = self.requested - self.cached
            available_set = self.requested - missing_set
            return (
                missing_set or None,
                available_set or None,
            )

        # Time periods are expected to be used for forecast reference times and
        # historical time periods.
        if isinstance(self.requested, TimePeriod) and isinstance(self.cached, TimePeriod):
            missing_tp, available_tp = self.find_missing_and_available_time_period(
                requested=self.requested,
                cached=self.cached,
            )
            return missing_tp, available_tp

        # Lead times are expected to be used for forecast lead times.
        if isinstance(self.requested, LeadTimes) and isinstance(
            self.cached,
            LeadTimes,
        ):
            requested_lead_times = self.requested.timedelta64
            cached_lead_times = self.cached.timedelta64

            missing_lt = [lt for lt in requested_lead_times if lt not in cached_lead_times]
            if len(missing_lt) == 0:
                return None, self.requested

            available_lt = [lt for lt in requested_lead_times if lt in cached_lead_times]
            if len(available_lt) == 0:
                return self.requested, None

            return (
                LeadTimes(
                    unit=TimeUnits.nanosecond,
                    values=[int(td / np.timedelta64(1, "ns")) for td in missing_lt],
                ),
                LeadTimes(
                    unit=TimeUnits.nanosecond,
                    values=[int(td / np.timedelta64(1, "ns")) for td in available_lt],
                ),
            )

        msg = "Unexpected case of missing data."
        raise ValueError(msg)


class HistoricalCacheRequest(BaseModel):
    """Helper class to determine what historical data is missing from the cache."""

    variables: DataRequest
    stations: DataRequest
    time_period: DataRequest

    @property
    def _missing(self) -> dict[str, set[str] | TimePeriod | LeadTimes | None]:
        """Return a mapping of field name to its missing data (or ``None`` if nothing missing)."""
        return {
            "variables": self.variables.get_missing_and_available[0],
            "stations": self.stations.get_missing_and_available[0],
            "time_period": self.time_period.get_missing_and_available[0],
        }

    @property
    def missing_dims(self) -> list[str]:
        """Return the dimensions along which historical data is missing from the cache."""
        _map = {
            "variables": "variable",
            "stations": StandardDim.station,
            "time_period": StandardDim.time,
        }
        return [_map[dim] for dim, value in self._missing.items() if value is not None]

    @property
    def missing_count(self) -> int:
        """Return the count of missing historical data from the cache."""
        return sum(value is not None for value in self._missing.values())

    def split_config(
        self,
        config: BaseDatasourceConfig,
    ) -> tuple[BaseDatasourceConfig, BaseDatasourceConfig] | None:
        """Modify the datasource config to fetch the missing data from the cache.

        Return a modified config if the missing data can be fetched from the datasource,
        otherwise return None.
        """
        # Copy the config to avoid mutating the original one
        fetch_from_datasource_config = config.model_copy(deep=True)
        fetch_from_cache_config = config.model_copy(deep=True)

        missing_time, available_time = self.time_period.get_missing_and_available
        missing_variables, available_variables = self.variables.get_missing_and_available
        missing_stations, available_stations = self.stations.get_missing_and_available

        # Only times are missing from cache
        if (
            missing_time is not None
            and available_time is not None
            and missing_variables is None
            and missing_stations is None
            and isinstance(missing_time, TimePeriod)
            and isinstance(available_time, TimePeriod)
        ):
            fetch_from_datasource_config.general.verification_period.start = missing_time.start
            fetch_from_datasource_config.general.verification_period.end = missing_time.end
            fetch_from_cache_config.general.verification_period.start = available_time.start
            fetch_from_cache_config.general.verification_period.end = available_time.end
            return fetch_from_datasource_config, fetch_from_cache_config
        # Only variables are missing from cache
        if (
            missing_time is None
            and missing_variables is not None
            and available_variables is not None
            and missing_stations is None
            and isinstance(missing_variables, set)
            and isinstance(available_variables, set)
        ):
            fetch_from_datasource_config.variables = sorted(missing_variables)
            fetch_from_cache_config.variables = sorted(available_variables)
            return fetch_from_datasource_config, fetch_from_cache_config

        # Only stations are missing from cache
        if (
            missing_time is None
            and missing_variables is None
            and missing_stations is not None
            and available_stations is not None
            and isinstance(missing_stations, set)
            and isinstance(available_stations, set)
        ):
            fetch_from_datasource_config.stations = sorted(missing_stations)
            fetch_from_cache_config.stations = sorted(available_stations)
            return fetch_from_datasource_config, fetch_from_cache_config

        return None


class ForecastCacheRequest(BaseModel):
    """Helper class to determine what forecast data is missing from the cache."""

    variables: DataRequest
    stations: DataRequest
    frt_period: DataRequest
    lead_times: DataRequest

    @property
    def _missing(self) -> dict[str, set[str] | TimePeriod | LeadTimes | None]:
        """Return a mapping of field name to its missing data (or ``None`` if nothing missing)."""
        return {
            "variables": self.variables.get_missing_and_available[0],
            "stations": self.stations.get_missing_and_available[0],
            "frt_period": self.frt_period.get_missing_and_available[0],
            "lead_times": self.lead_times.get_missing_and_available[0],
        }

    @property
    def missing_count(self) -> int:
        """Return the count of missing forecast data from the cache."""
        return sum(value is not None for value in self._missing.values())

    @property
    def missing_dims(self) -> list[str]:
        """Return the dimensions along which forecast data is missing from the cache."""
        _map = {
            "variables": "variable",
            "stations": StandardDim.station,
            "frt_period": StandardDim.forecast_reference_time,
            "lead_times": StandardDim.lead_time,
        }
        return [_map[dim] for dim, value in self._missing.items() if value is not None]

    def split_config(
        self,
        config: BaseDatasourceConfig,
    ) -> tuple[BaseDatasourceConfig, BaseDatasourceConfig] | None:
        """Modify the datasource config to fetch the missing data from the cache.

        Return a modified config if the missing data can be fetched from the datasource,
        otherwise return None.
        """
        # Copy the config to avoid mutating the original one
        fetch_from_datasource_config = config.model_copy(deep=True)
        fetch_from_cache_config = config.model_copy(deep=True)

        missing_frt, available_frt = self.frt_period.get_missing_and_available
        missing_variables, available_variables = self.variables.get_missing_and_available
        missing_stations, available_stations = self.stations.get_missing_and_available
        missing_lead_times, available_lead_times = self.lead_times.get_missing_and_available

        # Only forecast reference times are missing from cache
        if (
            missing_frt is not None
            and available_frt is not None
            and missing_variables is None
            and missing_stations is None
            and missing_lead_times is None
            and isinstance(missing_frt, TimePeriod)
            and isinstance(available_frt, TimePeriod)
        ):
            fetch_from_datasource_config.general.verification_period.start = missing_frt.start
            fetch_from_datasource_config.general.verification_period.end = missing_frt.end
            fetch_from_cache_config.general.verification_period.start = available_frt.start
            fetch_from_cache_config.general.verification_period.end = available_frt.end
            return fetch_from_datasource_config, fetch_from_cache_config

        # Only variables are missing from cache
        if (
            missing_frt is None
            and missing_variables is not None
            and available_variables is not None
            and missing_stations is None
            and missing_lead_times is None
            and isinstance(missing_variables, set)
            and isinstance(available_variables, set)
        ):
            fetch_from_datasource_config.variables = sorted(missing_variables)
            fetch_from_cache_config.variables = sorted(available_variables)
            return fetch_from_datasource_config, fetch_from_cache_config
        # Only stations are missing from cache
        if (
            missing_frt is None
            and missing_variables is None
            and missing_stations is not None
            and available_stations is not None
            and missing_lead_times is None
            and isinstance(missing_stations, set)
            and isinstance(available_stations, set)
        ):
            fetch_from_datasource_config.stations = sorted(missing_stations)
            fetch_from_cache_config.stations = sorted(available_stations)
            return fetch_from_datasource_config, fetch_from_cache_config

        # Only lead times are missing from cache
        if (
            missing_frt is None
            and missing_variables is None
            and missing_stations is None
            and missing_lead_times is not None
            and available_lead_times is not None
            and isinstance(missing_lead_times, LeadTimes)
            and isinstance(available_lead_times, LeadTimes)
        ):
            fetch_from_datasource_config.general.lead_times = missing_lead_times
            fetch_from_cache_config.general.lead_times = available_lead_times
            return fetch_from_datasource_config, fetch_from_cache_config

        return None


class ZarrCache:
    """The veriflow cache."""

    store = "veriflow-cache.zarr"

    def __init__(
        self,
        config: ZarrCacheConfig,
    ) -> None:
        self.config = config
        self.storage_options = self._build_storage_options()

    def _build_storage_options(self) -> dict[str, object] | None:
        """Build storage_options for xr.open_zarr based on path and config.

        Returns ``None`` for non-remote (local) paths so that xarray opens the store
        directly from the local filesystem.
        """
        if not self.config.is_remote_path():
            return None

        options: dict[str, object] = {}
        if self.config.auth_config is not None:
            options.update(self.config.auth_config.to_storage_options())
        if self.config.storage_options is not None:
            options.update(self.config.storage_options)
        return options

    def get_dataset(self, source: str) -> xr.Dataset:
        """Open a dataset from the cache."""
        try:
            return xr.open_zarr(  # type: ignore[no-any-return, misc]
                self.config.path,
                group=f"datasets/{source}",
                storage_options=self.storage_options,
                consolidated=self.config.consolidated,
            )
        except (FileNotFoundError, KeyError):
            return xr.Dataset()

    def append(
        self,
        new_dataset: xr.Dataset,
        source: str,
        dim: Literal[
            StandardDim.station,
            StandardDim.forecast_reference_time,
            StandardDim.lead_time,
            StandardDim.time,
            "variable",
        ],
    ) -> None:
        """Update the cache for a given dataset, by updating on coordinates or variables."""
        cached_ds = self.get_dataset(source)
        updated_ds = combine_cached_and_fetched_data(cached_ds, new_dataset, dim)
        updated_ds.to_zarr(  # type: ignore[call-overload]
            self.config.path,
            group=f"datasets/{source}",
            mode="w",
            storage_options=self.storage_options,
            consolidated=self.config.consolidated,
        )

    @property
    def is_remote(self) -> bool:
        """Return True if ``path`` looks like a remote/fsspec URL (e.g. ``s3://``)."""
        return "://" in self.config.path

    @property
    def is_writable(self) -> bool:
        """Return True if the cache is writable."""
        return self.config.read_write_mode == ReadWriteMode.write
