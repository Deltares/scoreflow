"""Shared resources across the test suite."""

# mypy: ignore-errors

import numpy as np
import pandas as pd
import pytest
import xarray as xr

rng = np.random.default_rng(seed=42)

# Dims
n_time = 60
n_frt = 10  # One forecast every 6 hours
n_forecast_period = 4
n_realization = 10
n_stations = 3

# Coords
start_date = "2025-01-01T00:00"
time = pd.date_range(start_date, periods=n_time, freq="H")
stations = [f"station{n}" for n in range(n_stations)]
x = rng.uniform(0, 100, size=n_stations)
y = rng.uniform(0, 100, size=n_stations)
z = rng.uniform(0, 10, size=n_stations)
realization = np.arange(1, n_realization + 1)

# One forecast every 6 hours
forecast_reference_time = pd.date_range(start_date, periods=n_frt, freq="6H")
forecast_period = np.array([np.timedelta64(i, "h") for i in range(1, n_forecast_period + 1)])


@pytest.fixture()
def xarray_dataset_observations() -> xr.Dataset:
    """Return example observations."""
    # Create observation data
    obs_data = rng.random((len(time), len(stations)))

    # Create dataset
    return xr.Dataset(
        data_vars={
            "obs": (("time", "stations"), obs_data),
        },
        coords={
            "time": time,
            "stations": stations,
            "x": ("stations", x),
            "y": ("stations", y),
            "z": ("stations", z),
        },
    )


@pytest.fixture()
def xarray_dataset_simulations_forecast_reference_time() -> xr.Dataset:
    """Return example simulations compatible with the internal datamodel."""
    # Generate random forecast data using Generator
    data = rng.random((n_time, n_frt, n_realization, n_stations))

    # Create Dataset
    return xr.Dataset(
        {
            "forecast": (("time", "forecast_reference_time", "realization", "stations"), data),
        },
        coords={
            "time": time,
            "forecast_reference_time": forecast_reference_time,
            "realization": realization,
            "stations": stations,
            "x": ("stations", x),
            "y": ("stations", y),
            "z": ("stations", z),
        },
    )


@pytest.fixture()
def xarray_dataset_simulations_forecast_period() -> xr.Dataset:
    """Return example simulations compatible with the internal datamodel.

    Uses forecast_period as dimension and coordinates.
    """
    data = rng.random((n_time, n_forecast_period, n_realization, n_stations))

    return xr.Dataset(
        {
            "forecast": (("time", "forecast_period", "realization", "stations"), data),
        },
        coords={
            "time": time,
            "forecast_period": forecast_period,
            "realization": realization,
            "stations": stations,
            "x": ("stations", x),
            "y": ("stations", y),
            "z": ("stations", z),
        },
    )
