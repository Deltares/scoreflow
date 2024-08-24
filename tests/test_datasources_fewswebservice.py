"""Test the fewswebservice module of the dpyverification.datasources package."""

import requests
import yaml
from dpyverification.configuration import ConfigSchema
from dpyverification.configuration.schema import LeadTimes
from dpyverification.constants import TimeUnits
from dpyverification.datasources.fewswebservice import FewsWebService

from tests import TESTS_CONFIGURATION_FILE

SIM_TIME_DIM_LENGTH = 49
OBS_TIME_DIM_LENGTH = 49
VALID_RESPONSE_CODE = 200


def test_webservice_live() -> None:
    """Test that a webservice is live and can find filters."""
    url = "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"  # type: ignore[misc]
    endpoint = "filters"
    test_endpoint_url = url + "/" + endpoint
    response = requests.get(test_endpoint_url, timeout=10)
    assert response.status_code == VALID_RESPONSE_CODE


def test_get_timeseries_sim_happy() -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    with TESTS_CONFIGURATION_FILE.open() as cf:
        leadtimes = LeadTimes(unit=TimeUnits.hour, values=[3, 6])
        testconf = yaml.safe_load(cf)
        testconf["general"]["verificationperiod"]["start"]["value"] = "2024-06-01T00:00:00Z"
        testconf["general"]["verificationperiod"]["end"]["value"] = "2024-06-03T00:00:00Z"
        testconf["datasources"][0]["simobstype"] = "sim"
        testconf["datasources"][0]["datasourcetype"] = "fewswebservice"
        testconf["datasources"][0]["url"] = (
            "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
        )
        testconf["datasources"][0]["location_ids"] = ["H-RN-0001"]
        testconf["datasources"][0]["parameter_ids"] = ["Q.fs"]
        testconf["datasources"][0]["module_instance_ids"] = ["SBK3_MaxRTK_ECMWF_ENS"]
        testconf["datasources"][0]["qualifier_ids"] = []
        testconf["datasources"][0]["document_format"] = "PI_XML"
        testconf["datasources"][0]["document_version"] = "1.32"
        testconf["datasources"][0]["leadtimes"] = leadtimes
        testconf["datasources"][0]["document_version"] = "1.32"

    parsed_content = ConfigSchema(**testconf)

    data = FewsWebService.get_data(parsed_content.datasources[0], giconfig=parsed_content.general)

    assert len(data[0].xarray.time) == SIM_TIME_DIM_LENGTH


def test_get_timeseries_obs_happy() -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf = yaml.safe_load(cf)
        testconf["general"]["verificationperiod"]["start"]["value"] = "2024-06-01T00:00:00Z"
        testconf["general"]["verificationperiod"]["end"]["value"] = "2024-06-03T00:00:00Z"
        testconf["datasources"][0]["simobstype"] = "obs"
        testconf["datasources"][0]["datasourcetype"] = "fewswebservice"
        testconf["datasources"][0]["url"] = (
            "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
        )
        testconf["datasources"][0]["location_ids"] = ["H-RN-0001"]
        testconf["datasources"][0]["parameter_ids"] = ["Q.m"]
        testconf["datasources"][0]["module_instance_ids"] = ["Import_LMW"]
        testconf["datasources"][0]["qualifier_ids"] = []
        testconf["datasources"][0]["document_format"] = "PI_XML"
        testconf["datasources"][0]["document_version"] = "1.32"

    parsed_content = ConfigSchema(**testconf)

    data = FewsWebService.get_data(parsed_content.datasources[0], giconfig=parsed_content.general)

    assert len(data[0].xarray.time) == OBS_TIME_DIM_LENGTH
