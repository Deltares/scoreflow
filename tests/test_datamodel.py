"""Test the dpyverification.datamodel package."""

from datetime import datetime, timezone

import numpy as np
import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.utils import ForecastPeriods, TimePeriod, TimeUnits
from dpyverification.datamodel.main import SimObsDataset
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile

# mypy: disable-error-code="misc"


def test_init_simobsdataset_fp(
    xarray_dataset_observations: xr.Dataset,
    xarray_dataset_simulations_forecast_period: xr.Dataset,
) -> None:
    """Test the simobsdataset initializes successfully with forecast period (fp) input."""
    general_config = GeneralInfoConfig(
        verification_period=TimePeriod(
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 3, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.HOUR, values=[1, 2, 3, 4]),
    )

    simobsdataset = SimObsDataset(
        data=[xarray_dataset_observations, xarray_dataset_simulations_forecast_period],
        general_config=general_config,
    )

    _ = simobsdataset


def test_init_simobsdataset_frt(
    simobsdataset_forecast_reference_time: SimObsDataset,
) -> None:
    """Test the simobsdataset initializes successfully with forecast ref time (frt) input."""
    assert isinstance(simobsdataset_forecast_reference_time.dataset, xr.Dataset)


def test_init_simobsdataset_fewsnetcdf(
    datasource_fewnetcdf_obs: FewsNetcdfFile,
    datasource_fewnetcdf_sim: FewsNetcdfFile,
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> None:
    """Test the fewsnetcdf is accepted by the simobsdataset."""
    SimObsDataset(
        data=[
            datasource_fewnetcdf_obs.get_data().dataset,
            datasource_fewnetcdf_sim.get_data().dataset,
        ],
        general_config=general_info_config_fewsnetcdf,
    )


def test_simobsdataset_frt_structure_correct(
    simobsdataset_forecast_reference_time: SimObsDataset,
    xarray_dataset_simulations_forecast_reference_time: xr.Dataset,
) -> None:
    """Test the data structure of the simobsdataset for input sim based on frt."""
    # Test that the second value in of the forecast with the first forecast_reference_time
    #   is equal to the value of the first lead time (frt + 1hr)
    value_in_input_data = xarray_dataset_simulations_forecast_reference_time["forecast"].isel(
        realization=0,
        station=0,
        forecast_reference_time=0,
    )[1]

    value_in_simobsdataset = simobsdataset_forecast_reference_time.dataset["forecast"].isel(
        station=0,
        realization=0,
        forecast_period=1,
    )[1]
    np.testing.assert_array_equal(value_in_simobsdataset.to_numpy(), value_in_input_data.to_numpy())

    # Test that the timestamp labels for each of the values are the same
    #   in other words: the time dimension stayed intact.
    np.testing.assert_array_equal(
        value_in_simobsdataset["time"].to_numpy(),
        value_in_input_data["time"].to_numpy(),
    )
