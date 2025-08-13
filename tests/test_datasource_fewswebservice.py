"""Test the fewswebservice module of the dpyverification.datasources package."""

# mypy: ignore-errors

import io
import os
import time
import zipfile
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import requests
import xarray as xr
import yaml
from dpyverification.api.fewswebservice import FewsWebserviceClient, TimeseriesType
from dpyverification.configuration import ConfigFile
from dpyverification.configuration.utils import ForecastPeriods
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile, Preprocessor
from dpyverification.datasources.fewswebservice import FewsWebservice

from tests import TESTS_CONFIGURATION_FILE

SIM_TIME_DIM_LENGTH = 373
OBS_TIME_DIM_LENGTH = 721
VALID_RESPONSE_CODE = 200
TASK_START_SUCCESS_TEXT = '{"started":true,"message":"Task started"}'

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="module", autouse=False)
def _initialize_archive() -> None:
    @dataclass
    class _ArchiveTask:
        name: str
        task_id: str

    clear_catalogue = _ArchiveTask("Clear internal catalogue", "clear internal catalogue")
    internal_harvester = _ArchiveTask("Internal harvester", "harvester internal catalogue")

    def get_archive_task_status(archive_task: _ArchiveTask) -> dict[str, bool | str]:
        archive_status_url = "http://localhost:8080/deltares-archive-server/api/v1/archive/status"

        archive_status_response = requests.get(archive_status_url, timeout=10)
        assert archive_status_response.status_code == VALID_RESPONSE_CODE
        archive_status: dict[str, list[dict[str, bool | str]]] = yaml.safe_load(
            archive_status_response.text,
        )

        for task in archive_status["list"]:
            if task["name"] == archive_task.name:
                return task
        msg = (
            f"Task with name {archive_task.name} not found in archive status"
            f" information: {archive_status_response.text}"
        )
        raise ValueError(msg)

    def start_and_wait_for_task(archive_task: _ArchiveTask) -> None:
        archive_task_post_url = "http://localhost:8080/deltares-archive-server/api/v1/runtask"

        # The taskId should match the predefinedArchiveTask entry in the ArchiveTasksSchedule.xml
        body = {"taskId": archive_task.task_id}
        # Use argument 'data' (that will pass the data as application/x-www-form-urlencoded), and
        #  NOT 'json', as that is not properly processed by the other side
        archive_task_post_response = requests.post(archive_task_post_url, data=body, timeout=10)
        assert archive_task_post_response.status_code == VALID_RESPONSE_CODE
        assert archive_task_post_response.text == TASK_START_SUCCESS_TEXT

        task_status = get_archive_task_status(archive_task)
        max_wait = 15.0
        waited_time = 0.0
        sleep_time = 0.5
        while task_status["running"] and waited_time < max_wait:
            # running, wait for finish
            time.sleep(sleep_time)
            waited_time = waited_time + sleep_time
            task_status = get_archive_task_status(archive_task)

        assert "finished" in task_status["status"]  # type: ignore[operator] # Indeed the use of in does not fully match with our faked type def of task_status

    # Check archive is up by requesting status
    _ = get_archive_task_status(clear_catalogue)
    # Always run these two tasks, before any of the tests on the webservice
    #   Do not check lastruntime or running status beforehand, unnecessary complication
    start_and_wait_for_task(clear_catalogue)
    start_and_wait_for_task(internal_harvester)


@pytest.fixture()
def _fews_webservice_mock_env(
    monkeypatch: Generator[pytest.MonkeyPatch, None, None],
) -> None:
    """Create a mock environment for testing secret env vars."""
    # The dummy url, username and password
    url = "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
    monkeypatch.setenv("FEWSWEBSERVICE_URL", url)  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_USERNAME", "")  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_PASSWORD", "")  # type: ignore  # noqa: PGH003


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_webservice_live() -> None:
    """Test that a webservice is live and can find filters."""
    url = "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
    endpoint = "archive/locations"
    test_endpoint_url = url + "/" + endpoint
    response = requests.get(test_endpoint_url, timeout=10)
    assert response.status_code == VALID_RESPONSE_CODE


@pytest.mark.usefixtures("_fews_webservice_mock_env")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_obs_netcdf(
    tmp_path: Path,
) -> None:
    """Check that the webservice gives expected outcome for obs."""
    verification_period = {
        "start": "2024-11-01T00:00:00Z",
        "end": "2024-12-01T00:00:00Z",
    }

    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)

    testconf["general"]["verification_period"] = verification_period  # type: ignore[call-overload] # Indeed this assignment does not match with our faked type def of testconf

    testconf["datasources"][0] = {
        "simobskind": "obs",
        "kind": "fewswebservice",
        "location_ids": ["H-RN-0001"],
        "parameter_ids": ["Q_m"],
        "module_instance_ids": ["Hydro_Prep"],
        "auth_config": {
            "url": os.environ.get("FEWSWEBSERVICE_URL"),
            "username": os.environ.get("FEWSWEBSERVICE_USERNAME"),
            "password": os.environ.get("FEWSWEBSERVICE_PASSWORD"),
        },
    }

    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")
    instance = FewsWebservice.from_config(conf.content.datasources[0].model_dump()).get_data()  # type: ignore[misc] # Yes, allow any
    assert "Q_m" in instance.dataset
    np.testing.assert_array_equal(instance.dataset["lat"].values, np.float64(51.85059))


@pytest.mark.usefixtures("_fews_webservice_mock_env")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_sim_netcdf(
    tmp_path: Path,
) -> None:
    """Check that the webservice gives expected outcome for sim."""
    forecast_periods = {"unit": "h", "values": [24, 48, 72, 96]}
    verification_period = {
        "start": "2024-11-01T00:00:00Z",
        "end": "2024-12-01T00:00:00Z",
    }

    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)

    testconf["general"]["verification_period"] = verification_period  # type: ignore[call-overload] # Indeed this assignment does not match with our faked type def of testconf
    testconf["general"]["forecast_periods"] = forecast_periods  # type: ignore[call-overload] # Indeed this assignment does not match with our faked type def of testconf

    testconf["datasources"][0] = {
        "simobskind": "sim",
        "kind": "fewswebservice",
        "location_ids": ["H-RN-0001"],
        "parameter_ids": ["Q_fs"],
        "module_instance_ids": ["SBK3_MaxRTK_ECMWF_ENS"],
        "ensemble_id": ["ECMWF_ENS"],
        "auth_config": {
            "url": os.environ.get("FEWSWEBSERVICE_URL"),
            "username": os.environ.get("FEWSWEBSERVICE_USERNAME"),
            "password": os.environ.get("FEWSWEBSERVICE_PASSWORD"),
        },
    }

    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")
    with pytest.raises(NotImplementedError, match="Simulations are not yet supported"):
        FewsWebservice.from_config(conf.content.datasources[0].model_dump()).get_data()  # type: ignore[misc] # Yes, allow any


@pytest.mark.usefixtures("_fews_webservice_mock_env")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_netcdf_storage_forecast_reference_times() -> None:
    """Test the get netcdf storage forecast endpoint."""
    client = FewsWebserviceClient(
        url="http://localhost:8080/FewsWebServices/rest/fewspiservice/v1",
        username=None,
        password=None,
    )
    datetime_list = client.get_netcdf_storage_forecasts_forecast_reference_times(
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert isinstance(datetime_list, list)
    assert isinstance(datetime_list[0], datetime)


def test_get_netcdf_storage_data_markermeer(tmp_path: Path) -> None:
    """Get raw forecasts from the external storage archive."""
    client = FewsWebserviceClient(
        url="http://localhost:8080/FewsWebServices/rest/fewspiservice/v1",
        username=None,
        password=None,
    )
    datetime_list = client.get_netcdf_storage_forecasts_forecast_reference_times(
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    path_list = []
    for dt in datetime_list:
        response = client.get_timeseries(
            location_ids=["GemaalVeendijk"],
            parameter_ids=["waterlevel_model"],
            module_instance_ids=["markermeer4c_ecmwf_eps"],
            ensemble_id="ECMWF-EPS",
            start_forecast_time=client._format_datetime(datetime(2024, 1, 1, tzinfo=timezone.utc)),
            end_forecast_time=client._format_datetime(datetime(2025, 1, 1, tzinfo=timezone.utc)),
            external_forecast_times=[client._format_datetime(dt)],
            timeseries_type=TimeseriesType.EXTERNAL_FORECASTING,
        )

        # Use BytesIO to treat bytes as a file-like object
        zip_bytes = io.BytesIO(response.content)

        # Open the zipfile in memory
        with zipfile.ZipFile(zip_bytes) as zf:
            # Assuming you want the first .nc file in the zip
            try:
                netcdf_filename = next(name for name in zf.namelist() if name.endswith(".nc"))
            except:
                continue

            # Extract that file in memory
            with zf.open(netcdf_filename) as netcdf_file:
                netcdf_data = netcdf_file.read()  # bytes of the .nc file

            # Write the NetCDF file to tmp_path
            netcdf_path = tmp_path / f"{dt.strftime('%Y%m%d%H')}.nc"
            netcdf_path.write_bytes(netcdf_data)
            path_list.append(netcdf_path)
    _ = FewsNetcdfFile._open_mf_dataset(path_list=path_list)


def test_load_50mb_of_data_transform_fp() -> None:
    """Load 50mb of data."""
    path = Path(
        "c:/Users/beunk/OneDrive - Stichting Deltares/Documents/000 - Projects/"
        "tmp/markermeer_files",
    )
    files = path.rglob("*.nc")

    preprocessor = Preprocessor(
        simobskind="sim",
        filter_variables=["waterlevel_model"],
        filter_forecast_periods=ForecastPeriods(unit="h", values=[12, 24, 36, 48]),
        transform_to_forecast_period_based_dataset=True,
    )
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        preprocess=preprocessor,
        coords="minimal",
        compat="override",
        chunks=None,
    )
    _ = ds


def test_load_50mb_of_data_along_frt() -> None:
    """Load 50mb of data."""
    path = Path(
        "c:/Users/beunk/OneDrive - Stichting Deltares/Documents/000 - Projects/"
        "tmp/markermeer_files",
    )
    files = path.rglob("*.nc")

    preprocessor = Preprocessor(
        simobskind="sim",
        filter_variables=["waterlevel_model"],
        filter_forecast_periods=ForecastPeriods(unit="h", values=[12, 24, 36, 48]),
    )
    # ds = xr.open_mfdataset(
    #     files,
    #     combine="nested",
    #     concat_dim="forecast_reference_time",
    #     preprocess=preprocessor,
    #     coords="minimal",
    #     compat="override",
    #     parallel=True,
    #     chunks=None,
    # )

    ds = xr.open_mfdataset(files, preprocess=preprocessor)
    _ = ds
