"""Test the pixml module of the dpyverification.datasources package."""

import numpy as np
import yaml
from dpyverification.datasources.pixml import PiXmlFile

from tests import TESTS_CONFIGURATION_FILE, TESTS_FORECASTS_FILE


def test_happy() -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
        testconf["datasources"][0]["directory"] = str(TESTS_FORECASTS_FILE.parent)
        testconf["datasources"][0]["filename"] = TESTS_FORECASTS_FILE.name

    forecastdata = PiXmlFile.get_data(testconf["datasources"][0])

    assert (
        forecastdata.loc[  # type: ignore[misc]
            np.datetime64("2024-07-03T05:00"),
            "LOC2",
            "Q.fs",
            frozenset(),
            20:21,
        ].data.tolist()
        == [468.22, 468.27]
    )
