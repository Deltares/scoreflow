"""Test the veriflow.datamodel package."""

import xarray as xr
from veriflow.datamodel.input import InputDataset
from veriflow.datasources.csv import Csv
from veriflow.datasources.fewsnetcdf import FewsNetCDF

# mypy: disable-error-code="misc"


def test_init_input_dataset_simulated_forecast_ensemble(
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Test the input_dataset initializes successfully with lead time (fp) input."""
    _ = InputDataset(
        data=[xarray_observed_historical, xarray_simulated_forecast_ensemble],
    )


def test_input_dataset_obs_mapper(
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Test the function that maps observations into forecast space."""
    variable = "var_0"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_ensemble[variable]

    # Map the obs into forecast space
    obs_reprojected = InputDataset.map_historical_into_forecast_space(obs, sim)

    # Get a subset of obs and sim
    sim_subset = obs_reprojected.isel(station=0, lead_time=0)
    obs_subset = obs.isel(station=0).sel(time=sim_subset.forecast_reference_time)

    # We expect all values at forecast_reference_time=0 to match the observed values
    assert all(sim_subset.to_numpy() == obs_subset.to_numpy())


def test_init_input_dataset_simulated_forecast_single(
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_single: xr.Dataset,
) -> None:
    """Test the input_dataset initializes successfully with lead time (fp) input."""
    _ = InputDataset(
        data=[xarray_observed_historical, xarray_simulated_forecast_single],
    )


def test_init_input_dataset_fewsnetcdf(
    fews_netcdf_observed_historical: FewsNetCDF,
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
) -> None:
    """Test the fewsnetcdf is accepted by the input_dataset."""
    InputDataset(
        data=[
            fews_netcdf_observed_historical.get_data().dataset,
            fews_netcdf_simulated_forecast_ensemble_frt.get_data().dataset,
        ],
    )


def test_init_input_dataset_thresholds(
    xarray_thresholds: Csv,
) -> None:
    """Test the fewsnetcdf is accepted by the input_dataset."""
    InputDataset(
        data=[
            xarray_thresholds.dataset,
        ],
    )
