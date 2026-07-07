"""Module with the base class that all datasources should inherit from."""

from abc import abstractmethod
from typing import ClassVar, Self, cast

import pandas as pd
import xarray as xr

from veriflow.base import Base
from veriflow.cache.cache import (
    DataRequest,
    ForecastCacheRequest,
    HistoricalCacheRequest,
    ZarrCache,
)
from veriflow.cache.utils import combine_cached_and_fetched_data
from veriflow.configuration.base import (
    BaseDatasourceConfig,
)
from veriflow.configuration.utils import LeadTimes, TimePeriod
from veriflow.constants import (
    FORECAST_DATA_TYPES,
    HISTORICAL_DATA_TYPES,
    DataType,
    StandardDim,
    TimeUnits,
)

__all__ = [
    "BaseDatasource",
    "BaseDatasourceConfig",
]

CACHABLE_DATA_TYPES = FORECAST_DATA_TYPES + HISTORICAL_DATA_TYPES  # type: ignore[misc]


class BaseDatasource(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind: str = ""
    config_class: type[BaseDatasourceConfig] = BaseDatasourceConfig
    supported_data_types: ClassVar[set[DataType]] = set()

    def __init__(self, config: BaseDatasourceConfig) -> None:
        self.config: BaseDatasourceConfig = config
        self.data_type = config.data_type
        self.dataset: xr.Dataset = xr.Dataset()

    @property
    def data_type(self) -> DataType:
        """Whether the instance represents sim or obs data."""
        return self.config.data_type

    @data_type.setter
    def data_type(self, new_data_type: DataType) -> None:
        if new_data_type not in self.supported_data_types:
            msg = (
                f"Data type '{new_data_type}' is not supported ",
                f"by {self.__class__.__name__}",
            )
            raise NotImplementedError(msg)

        self._data_type = new_data_type

    @property
    def cache(self) -> ZarrCache | None:
        """The cache instance if configured, else None."""
        return (
            ZarrCache(self.config.general.cache) if self.config.general.cache is not None else None
        )

    @abstractmethod
    def fetch_data(self) -> Self:
        """Fetch data from datasource."""

    def _validate_data_type(self) -> None:
        # Check that the datatype is defined, and consistent with the config
        if "data_type" not in self.dataset.attrs:  # type:ignore[misc]
            msg = "The fetched dataset does not have a 'data_type' attribute."
            raise ValueError(msg)

        if self.dataset.attrs["data_type"] != self.data_type:  # type:ignore[misc]
            msg = (
                f"The data type of the fetched dataset "
                f"({self.dataset.attrs['data_type']}) does not match the expected data "  # type:ignore[misc]
                f"type ({self.config.data_type})."
            )
            raise ValueError(msg)

    def _validate_source(self) -> None:
        # Make sure the source attribute is set to the expected source
        self.dataset.attrs["source"] = self.config.source  # type:ignore[misc]

    def _validate_lead_times(self) -> None:
        """Check that lead times are provided for forecast data types."""
        if not self.config.lead_times and self.data_type in FORECAST_DATA_TYPES:
            msg = "Lead times must be provided in the config for forecast data types, but got None."
            raise ValueError(msg)

    @staticmethod
    def _filter_lead_times(
        dataset: xr.Dataset,
        lead_times: LeadTimes | None,
    ) -> xr.Dataset:
        """Filter forecast dataset on lead times."""
        if dataset.attrs["data_type"] in FORECAST_DATA_TYPES and lead_times is not None:  # type:ignore[misc]
            # Select only relevant lead times for simulations
            dataset = dataset.sel(
                lead_time=lead_times.timedelta64,
            )
        return dataset

    @staticmethod
    def _filter_times(dataset: xr.Dataset, verification_period_on_time: TimePeriod) -> xr.Dataset:
        """Filter the times outside the verification period and lead times."""
        data_type = dataset.attrs["data_type"]  # type:ignore[misc]

        if data_type in FORECAST_DATA_TYPES:  # type:ignore[misc]
            # Mask and drop time values outside of the configured vp
            filtered = dataset.where(
                (dataset[StandardDim.time] >= verification_period_on_time.start_datetime64)
                & (dataset[StandardDim.time] <= verification_period_on_time.end_datetime64),
            )
            # Drop NaN values along frt and fp dims, if all values are NaN
            return filtered.dropna(dim=StandardDim.forecast_reference_time, how="all").dropna(
                dim=StandardDim.lead_time,
                how="all",
            )
        if data_type in HISTORICAL_DATA_TYPES:  # type:ignore[misc]
            # Mask and drop time values outside of the configured vp
            # Historical data type
            dataset = dataset.sel(
                {
                    StandardDim.time: slice(  # type:ignore[misc]
                        verification_period_on_time.start,
                        verification_period_on_time.end,
                    ),
                },
            )
        return dataset

    def validate_fetched_data(self) -> None:
        """Validate that the dataset is consistent with the config, and apply filtering."""
        self._validate_data_type()
        self._validate_source()
        self._validate_lead_times()

    def filter_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Filter the dataset on lead times and times outside the verification period."""
        # Filter lead_times
        dataset = self._filter_lead_times(
            dataset,
            self.config.lead_times,
        )
        # Filter times outside verification period
        return self._filter_times(
            dataset,
            self.config.verification_period_on_time,
        )

    def fetch_validate_filter(self) -> Self:
        """High-level wrapper to fetch, validate, filter and apply id mapping to the dataset."""
        self.fetch_data()
        self.validate_fetched_data()
        dataset = self.filter_dataset(self.dataset)
        if self.config.id_mapping is not None:
            self.dataset = self.config.id_mapping.apply(dataset)
        return self

    def create_cache_request(
        self,
        cached_dataset: xr.Dataset,
    ) -> HistoricalCacheRequest | ForecastCacheRequest:
        """Get the cache request based on the cached dataset and the config."""
        # Determine what data is missing from the cache
        if self.data_type in HISTORICAL_DATA_TYPES:
            return HistoricalCacheRequest(
                variables=DataRequest(
                    requested=set(self.config.variables),
                    cached=cast("set[str]", set(cached_dataset.data_vars)),
                ),
                stations=DataRequest(
                    requested=set(self.config.stations),
                    cached=set(cached_dataset[StandardDim.station].values),  # type: ignore[misc]
                ),
                time_period=DataRequest(
                    requested=TimePeriod(
                        start=self.config.verification_period_on_time.start,
                        end=self.config.verification_period_on_time.end,
                    ),
                    cached=TimePeriod(
                        start=pd.Timestamp(
                            cached_dataset[StandardDim.time].min().item(),  # type: ignore[misc]
                        ).to_pydatetime(),
                        end=pd.Timestamp(
                            cached_dataset[StandardDim.time].max().item(),  # type: ignore[misc]
                        ).to_pydatetime(),
                    ),
                ),
            )
        if self.data_type in FORECAST_DATA_TYPES:
            if self.config.lead_times is None:
                msg = "lead_times must be configured for forecast data types."
                raise ValueError(msg)
            cached_lead_times_array = cached_dataset[StandardDim.lead_time].to_numpy()  # type: ignore[misc]
            return ForecastCacheRequest(
                variables=DataRequest(
                    requested=set(self.config.variables),
                    cached=cast("set[str]", set(cached_dataset.data_vars)),
                ),
                stations=DataRequest(
                    requested=set(self.config.stations),
                    cached=set(cached_dataset[StandardDim.station].values),  # type: ignore[misc]
                ),
                frt_period=DataRequest(
                    requested=TimePeriod(
                        start=self.config.verification_period_on_frt.start,
                        end=self.config.verification_period_on_frt.end,
                    ),
                    cached=TimePeriod(
                        start=pd.Timestamp(
                            cached_dataset[StandardDim.forecast_reference_time].min().item(),  # type: ignore[misc]
                        ).to_pydatetime(),
                        end=pd.Timestamp(
                            cached_dataset[StandardDim.forecast_reference_time].max().item(),  # type: ignore[misc]
                        ).to_pydatetime(),
                    ),
                ),
                lead_times=DataRequest(
                    requested=self.config.lead_times,
                    cached=LeadTimes(
                        unit=TimeUnits.nanosecond,
                        values=[  # type: ignore[misc]
                            int(td)  # type: ignore[misc]
                            for td in cached_lead_times_array.astype(  # type: ignore[misc]
                                "timedelta64[ns]",
                            ).astype("int64")
                        ],
                    ),
                ),
            )
        msg = f"Unsupported data type '{self.data_type}' for creating cache request."
        raise NotImplementedError(msg)

    @staticmethod
    def get_cached_data(
        cached_dataset: xr.Dataset,
        config: BaseDatasourceConfig,
    ) -> xr.Dataset:
        """Get the dataset from the cache based on the config."""
        # If all requested data is available in the cache, load directly.
        if config.data_type in FORECAST_DATA_TYPES:
            if config.lead_times is None:
                msg = "lead_times must be configured for forecast data types."
                raise ValueError(msg)
            dataset = cached_dataset[config.variables].sel(
                {  # type: ignore[misc]
                    StandardDim.station: config.stations,
                    StandardDim.forecast_reference_time: slice(  # type: ignore[misc]
                        config.verification_period_on_frt.start,
                        config.verification_period_on_frt.end,
                    ),
                    StandardDim.lead_time: config.lead_times.timedelta64,
                },
            )
        elif config.data_type in HISTORICAL_DATA_TYPES:
            # All requested data is available in the cache, so we can load it directly
            dataset = cached_dataset[config.variables].sel(
                {  # type: ignore[misc]
                    StandardDim.station: config.stations,
                    StandardDim.time: slice(  # type: ignore[misc]
                        config.verification_period_on_time.start,
                        config.verification_period_on_time.end,
                    ),
                },
            )
        else:
            msg = f"Unsupported data type '{config.data_type}' for loading from cache."
            raise NotImplementedError(msg)
        return dataset

    def get_data(self) -> Self:
        """Get data and make use of cache if configured."""
        # If no cache is configured, or the data type is not cacheable, fetch and process the data
        # directly from the datasource, without using the cache.
        if self.cache is None or self.data_type not in CACHABLE_DATA_TYPES:  # type: ignore[misc]
            self.fetch_validate_filter()
            return self

        # Get the dataset from cache
        cached_dataset = self.cache.get_dataset(source=self.config.source)

        # If no data is found in the cache for the given source, fetch and process the data directly
        # from the datasource, and write to cache if configured to do so.
        if not cached_dataset.data_vars:
            # No data from cache, fetch from datasource
            self.fetch_validate_filter()
            if self.cache.is_writable:
                self.cache.append(
                    new_dataset=self.dataset,
                    source=self.config.source,
                    dim="variable",
                )
            return self

        cache_request = self.create_cache_request(cached_dataset)

        # No data is missing from the cache, so we can load it directly and skip fetching from the
        # datasource.
        if cache_request.missing_count == 0:
            cached_dataset = self.get_cached_data(
                cached_dataset=cached_dataset,
                config=self.config,
            )
            self.dataset = cached_dataset
            return self

        # Some data is missing from the cache. Try to split the config so we only fetch the
        # missing data (works only when exactly one dim is missing). If split_config returns
        # None — either because multiple dims are missing or the missing data isn't expressible
        # by tweaking the config — fall back to fetching the full requested dataset.
        split_result = cache_request.split_config(self.config)
        if split_result is None:
            self.fetch_validate_filter()
            if self.cache.is_writable:
                self.cache.append(
                    new_dataset=self.dataset,
                    source=self.config.source,
                    dim="variable",
                )
            return self

        missing_dim = cache_request.missing_dims[0]
        fetch_from_datasource_config, fetch_from_cache_config = split_result

        # If we get here, it means that some data is missing from the cache, but it can be
        # fetched by modifying the config to only fetch the missing data.
        cached_dataset_filtered = self.get_cached_data(
            cached_dataset=cached_dataset,
            config=fetch_from_cache_config,
        )

        datasource_modified = self.from_config(fetch_from_datasource_config.model_dump())  # type: ignore[misc]
        datasource_modified.fetch_validate_filter()
        newly_fetched_dataset = datasource_modified.dataset

        # After fetching the missing data, we can combine it with the cached data to get the full
        # dataset.
        self.dataset = combine_cached_and_fetched_data(
            cached_dataset=cached_dataset_filtered,
            fetched_dataset=newly_fetched_dataset,
            dim=missing_dim,
        )

        # If the cache is writable, we also write the newly fetched data to the cache for
        # future use. We write
        if self.cache.is_writable:
            self.cache.append(
                new_dataset=newly_fetched_dataset,
                source=self.config.source,
                dim=missing_dim,  # type: ignore[arg-type]
            )

        return self
