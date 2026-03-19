"""Test the threshold datasource."""

import numpy as np
import pytest

from dpyverification.datasources.csv import Csv


def test_fetch_thresholds(
    xarray_thresholds: Csv,
) -> None:
    """Test we can fetch thresholds from csv file."""
    xarray_thresholds.fetch_data()

    # Test the threshold ids are as expected (coordinate values)
    np.testing.assert_array_equal(
        xarray_thresholds.data_array.threshold.to_numpy(),  # type:ignore[misc]
        np.array(["warn_1"]),  # type:ignore[misc]
    )

    # Test one threshold value is as expected
    expected_value = np.array(0.1542894920675478)  # type:ignore[misc]
    np.testing.assert_approx_equal(
        xarray_thresholds.data_array.isel(station=0, variable=0, threshold=0).to_numpy(),  # type:ignore[misc]
        expected_value,  # type:ignore[misc]
    )


def test_fetch_thresholds_raises_error_on_invalid_location_config(
    xarray_thresholds: Csv,
) -> None:
    """Test we can fetch thresholds from csv file."""
    config = xarray_thresholds.config
    config.stations = ["non_existent_station"]
    instance = Csv(config)

    expected_error_msg = "One of the configured station, variable or threshold ids was not found *"

    with pytest.raises(ValueError, match=expected_error_msg):
        instance.fetch_data()
