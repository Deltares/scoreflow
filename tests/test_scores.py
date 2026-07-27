"""Module to test the available scores."""

import importlib
import sys
from copy import deepcopy

import pytest
import xarray as xr

from veriflow.configuration.default.scores import (
    CategoricalScoresConfig,
    ContinuousScoresConfig,
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
    SALScoreConfig,
)
from veriflow.constants import DataType, ScoreKind, SpatialType
from veriflow.datamodel.main import InputDataset
from veriflow.datasources.fewsnetcdf import FewsNetCDF
from veriflow.scores.categorical import CategoricalScores
from veriflow.scores.continuous import ContinuousScores
from veriflow.scores.probabilistic import CrpsCDF, CrpsForEnsemble, RankHistogram
from veriflow.scores.spatial import SALScore

pysteps_available = importlib.util.find_spec("pysteps") is not None

_SKIP_PYSTEPS = (
    sys.version_info >= (3, 13) or not pysteps_available
)  # pysteps pip install build fails on Windows for Python 3.13/3.14


def test_ensemble_crps(
    score_config_crps: CrpsForEnsembleConfig,
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Test CRPS."""
    variable = "var_0"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_ensemble[variable]
    obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]  # type:ignore[misc]
    sim.attrs["data_type"] = xarray_simulated_forecast_ensemble.attrs["data_type"]  # type:ignore[misc]
    obs_reprojected = InputDataset.map_historical_into_forecast_space(obs, sim)

    result = CrpsForEnsemble(score_config_crps).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert result.name == score_config_crps.score_adapter  # type:ignore[misc]


def test_ensemble_rank_histogram(
    score_config_rank_histogram: RankHistogramConfig,
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Test CRPS."""
    variable = "var_0"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_ensemble[variable]
    obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]  # type:ignore[misc]
    sim.attrs["data_type"] = xarray_simulated_forecast_ensemble.attrs["data_type"]  # type:ignore[misc]
    obs_reprojected = InputDataset.map_historical_into_forecast_space(obs, sim)

    result = RankHistogram(score_config_rank_histogram).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert result.name == "histogram_rank"  # type:ignore[misc]


def test_probabilistic_crps_cdf(
    score_config_crps_cdf: CrpsCDFConfig,
    fews_netcdf_simulated_forecast_probabilistic_fp: FewsNetCDF,
) -> None:
    """Test CRPS."""
    sim_ds = fews_netcdf_simulated_forecast_probabilistic_fp.get_data().dataset
    variable = next(iter(sim_ds.data_vars))
    sim = sim_ds[variable]
    sim.attrs["data_type"] = sim_ds.attrs["data_type"]  # type:ignore[misc]

    # Synthetic obs
    mean_sim = sim.threshold.mean()  # type:ignore[misc]
    obs_dummy = xr.full_like(sim.mean(["threshold", "lead_time"]), mean_sim)  # type:ignore[misc]
    obs_dummy.name = "source_observation"
    obs_dummy.attrs.update({"data_type": DataType.observed_historical})  # type:ignore[misc]

    config_instance = deepcopy(score_config_crps_cdf.model_dump())  # type:ignore[misc]
    conf = config_instance  # type:ignore[misc]
    conf["general"]["verification_pairs"][0].update(  # type:ignore[misc]
        {
            "id": "pair1",
            "obs": "source_observation",
            "sim": "source_probabilistic",
            "variable": variable,
        },
    )

    score = CrpsCDF(CrpsCDFConfig(**conf))  # type:ignore[misc]
    result = score.validate_and_compute(obs=obs_dummy, sim=sim)
    assert score_config_crps_cdf.score_adapter in result


def test_single_continuous_scores(
    score_config_continuous: ContinuousScoresConfig,
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_single: xr.Dataset,
) -> None:
    """Test CRPS."""
    variable = "var_0"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_single[variable]
    obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]  # type:ignore[misc]
    sim.attrs["data_type"] = xarray_simulated_forecast_single.attrs["data_type"]  # type:ignore[misc]
    obs_reprojected = InputDataset.map_historical_into_forecast_space(obs, sim)

    result = ContinuousScores(score_config_continuous).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert isinstance(result, xr.Dataset)  # type:ignore[misc]
    assert "mae" in result
    assert "rmse" in result
    assert "nse" in result
    assert "kge" in result


def test_categorical_scores(
    score_config_categorical: CategoricalScoresConfig,
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_single: xr.Dataset,
    xarray_thresholds: xr.DataArray,
) -> None:
    """Test the categorical scores config."""
    variable = "var_1"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_single[variable]
    obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]  # type:ignore[misc]
    sim.attrs["data_type"] = xarray_simulated_forecast_single.attrs["data_type"]  # type:ignore[misc]
    instance = CategoricalScores(config=score_config_categorical)
    instance.validate_and_compute(
        obs=obs,
        sim=sim,
        thresholds=xarray_thresholds.dataset[variable],  # type:ignore[misc]
    )


@pytest.mark.skipif(
    _SKIP_PYSTEPS,
    reason="pysteps not installed or Python version >= 3.13 (pysteps pip install build fails on "
    "Windows for Python 3.13/3.14)",
)
def test_sal_score_computes(
    xarray_simulated_forecast_single_gridded: xr.Dataset,
    xarray_general_info_config: object,
) -> None:
    """The SAL score computes structure/amplitude/location on gridded forecasts."""
    pytest.importorskip("pysteps")

    variable = "var_0"
    sim = xarray_simulated_forecast_single_gridded[variable]
    obs = xarray_simulated_forecast_single_gridded["var_1"]
    sim.attrs.update(  # type:ignore[misc]
        {"data_type": DataType.simulated_forecast_single, "spatial_type": SpatialType.gridded},  # type:ignore[misc]
    )
    obs.attrs.update(  # type:ignore[misc]
        {"data_type": DataType.observed_historical, "spatial_type": SpatialType.gridded},  # type:ignore[misc]
    )

    config = SALScoreConfig(
        score_adapter=ScoreKind.sal,
        general=xarray_general_info_config.model_dump(),  # type:ignore[misc, attr-defined]
    )
    result = SALScore(config).validate_and_compute(obs=obs, sim=sim)

    assert isinstance(result, xr.Dataset)  # type:ignore[misc]
    assert set(result.data_vars) == {"structure", "amplitude", "location"}
