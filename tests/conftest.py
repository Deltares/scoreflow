"""Shared resources across the test suite."""

# mypy: ignore-errors

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.default.datasources import FewsNetcdfKind
from dpyverification.configuration.utils import ForecastPeriods, TimePeriod, TimeUnits
from dpyverification.constants import StandardCoord, StandardDim
from dpyverification.datamodel.main import SimObsDataset
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile

rng = np.random.default_rng(seed=42)

# Dims
n_time = 60
n_frt = 10  # One forecast every 6 hours
n_forecast_period = 4  # Hours
n_realization = 10
n_stations = 3

# Coords
start_date = "2025-01-01T00:00"
time = pd.date_range(start_date, periods=n_time, freq="h")
station_ids = [f"station{n}" for n in range(n_stations)]
x = rng.uniform(0, 100, size=n_stations)
y = rng.uniform(0, 100, size=n_stations)
z = rng.uniform(0, 10, size=n_stations)
lat = y
lon = x
realization = np.arange(1, n_realization + 1)

# One forecast every 6 hours
forecast_reference_time = pd.date_range(
    start_date,
    periods=n_frt,
    freq="6h",
)
forecast_period = np.array([np.timedelta64(i, "h") for i in range(1, n_forecast_period + 1)])


@pytest.fixture()
def xarray_dataset_observations() -> xr.Dataset:
    """Return example observations."""
    # Create observation data
    obs_data = rng.random((len(time), len(station_ids)))

    # Create dataset
    return xr.Dataset(
        data_vars={
            "obs": ((StandardDim.time, StandardDim.station), obs_data),
        },
        coords={
            StandardCoord.time.name: time,
            StandardCoord.station_id.name: (StandardDim.station, station_ids),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
    )


@pytest.fixture()
def xarray_dataset_simulations_forecast_reference_time() -> xr.Dataset:
    """Return example simulations compatible with the internal datamodel."""
    # Generate random forecast data using Generator
    data = rng.random((n_time, n_frt, n_realization, n_stations))

    # Create Dataset
    ds = xr.Dataset(
        {
            "forecast": (
                (
                    StandardDim.time,
                    StandardDim.forecast_reference_time,
                    StandardDim.realization,
                    StandardDim.station,
                ),
                data,
            ),
        },
        coords={
            StandardCoord.time.name: time,
            StandardCoord.forecast_reference_time.name: forecast_reference_time,
            StandardCoord.realization.name: realization,
            StandardCoord.station_id.name: (StandardDim.station, station_ids),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
    )

    # Mask some forecast values to be more realistic
    mask = (ds[StandardDim.time] >= ds[StandardDim.forecast_reference_time]) & (
        ds[StandardDim.time] <= ds[StandardDim.forecast_reference_time] + max(forecast_period)
    )
    return ds.where(mask)


@pytest.fixture()
def xarray_dataset_simulations_forecast_period() -> xr.Dataset:
    """Return example simulations compatible with the internal datamodel.

    Uses forecast_period as dimension and coordinates.
    """
    data = rng.random((n_time, n_forecast_period, n_realization, n_stations))

    return xr.Dataset(
        {
            "forecast": (
                (
                    StandardDim.time,
                    StandardDim.forecast_period,
                    StandardDim.realization,
                    StandardDim.station,
                ),
                data,
            ),
        },
        coords={
            StandardCoord.time.name: time,
            StandardCoord.forecast_period.name: forecast_period,
            StandardCoord.realization.name: realization,
            StandardCoord.station_id.name: (StandardDim.station, station_ids),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
    )


@pytest.fixture()
def testconfig_general_info_simobsdataset_from_dummy_data() -> GeneralInfoConfig:
    """General info config to be used across tests."""
    return GeneralInfoConfig(
        verification_period=TimePeriod(
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 3, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.HOUR, values=[1, 2, 3, 4]),
    )


@pytest.fixture()
def general_info_config_fewsnetcdf() -> GeneralInfoConfig:
    """Get general info config matching the test data."""
    return GeneralInfoConfig(
        verification_period=TimePeriod(
            start=datetime(2025, 10, 1, tzinfo=timezone.utc),
            end=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.HOUR, values=[3, 6, 9, 12]),
    )


@pytest.fixture()
def datasource_fewnetcdf_obs(general_info_config_fewsnetcdf: GeneralInfoConfig) -> FewsNetcdfFile:
    """Fewsnetcdf datasource obs config."""
    return FewsNetcdfFile.from_config(
        {
            "kind": "fewsnetcdf",
            "simobskind": "obs",
            "netcdf_kind": "observation",
            "directory": "tests/data/webservice_responses_netcdf/obs",
            "filename_pattern": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "general": general_info_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def datasource_fewnetcdf_sim(general_info_config_fewsnetcdf: GeneralInfoConfig) -> FewsNetcdfFile:
    """Fewsnetcdf datasource sim config."""
    return FewsNetcdfFile.from_config(
        {
            "kind": "fewsnetcdf",
            "simobskind": "sim",
            "netcdf_kind": FewsNetcdfKind.one_full_simulation,
            "directory": "tests/data/webservice_responses_netcdf/sim",
            "filename_pattern": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "general": general_info_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def datasource_fewsnetcdf_obs(
    datasource_fewnetcdf_obs: dict[
        str,
        str | list[str] | dict[str, dict[str, str | list[str]]],
    ],
) -> FewsNetcdfFile:
    """Get a fews netcdf datasource."""
    return FewsNetcdfFile(datasource_fewnetcdf_obs).get_data()


@pytest.fixture()
def datasource_fewsnetcdf_sim(
    datasource_fewnetcdf_sim: dict[
        str,
        str | list[str] | dict[str, dict[str, str | list[str]]],
    ],
) -> FewsNetcdfFile:
    """Get a fews netcdf datasource."""
    return FewsNetcdfFile(datasource_fewnetcdf_sim).get_data()


@pytest.fixture()
def simobsdataset_forecast_reference_time(
    xarray_dataset_observations: xr.Dataset,
    xarray_dataset_simulations_forecast_reference_time: xr.Dataset,
    testconfig_general_info_simobsdataset_from_dummy_data: GeneralInfoConfig,
) -> SimObsDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return SimObsDataset(
        data=[xarray_dataset_observations, xarray_dataset_simulations_forecast_reference_time],
        general_config=testconfig_general_info_simobsdataset_from_dummy_data,
    )
