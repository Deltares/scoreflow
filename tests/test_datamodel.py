"""Test the dpyverification.datamodel package."""

from datetime import datetime, timezone

import numpy as np
import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.utils import LeadTimes, TimePeriod, TimeUnits
from dpyverification.datamodel.main import SimObsDataset

# mypy: disable-error-code="misc"


def test_init_datamodel_fp(
    xarray_dataset_observations: xr.Dataset,
    xarray_dataset_simulations_forecast_period: xr.Dataset,
) -> None:
    """Test the datamodel initializes succefully with forecast period (fp) input."""
    general_config = GeneralInfoConfig(
        verificationperiod=TimePeriod(
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 3, tzinfo=timezone.utc),
        ),
        leadtimes=LeadTimes(unit=TimeUnits.HOUR, values=[1, 2, 3, 4]),
    )

    datamodel = SimObsDataset(
        data=[xarray_dataset_observations, xarray_dataset_simulations_forecast_period],
        general_config=general_config,
    )

    _ = datamodel


def test_init_datamodel_frt(
    datamodel_forecast_reference_time: SimObsDataset,
) -> None:
    """Test the datamodel initializes succefully with forecast ref time (frt) input."""
    assert isinstance(datamodel_forecast_reference_time.dataset, xr.Dataset)


def test_datamodel_frt_structure_correct(
    datamodel_forecast_reference_time: SimObsDataset,
    xarray_dataset_simulations_forecast_reference_time: xr.Dataset,
) -> None:
    """Test the data structure of the datamodel for input sim based on frt."""
    # Test that the second value in of the forecast with the first forecast_reference_time
    #   is equal to the value of the first lead time (frt + 1hr)
    value_in_input_data = xarray_dataset_simulations_forecast_reference_time["forecast"].isel(
        realization=0,
        stations=0,
        forecast_reference_time=0,
    )[1]

    value_in_datamodel = datamodel_forecast_reference_time.dataset["forecast"].isel(
        stations=0,
        realization=0,
        forecast_period=1,
    )[1]
    np.testing.assert_array_equal(value_in_datamodel.to_numpy(), value_in_input_data.to_numpy())

    # Test that the timestamp labels for each of the values are the same
    #   in other words: the time dimension stayed intact.
    np.testing.assert_array_equal(
        value_in_datamodel["time"].to_numpy(),
        value_in_input_data["time"].to_numpy(),
    )
