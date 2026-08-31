"""Module for tests of datasinks."""

from pathlib import Path
from typing import cast

import pytest
import xarray as xr

from veriflow.configuration.base import GeneralInfoConfig
from veriflow.constants import DataSinkKind
from veriflow.datasinks.cf_compliant_netcdf import CFCompliantNetCDF, CFCompliantNetCDFConfig
from veriflow.datasinks.cf_compliant_zarr import CFCompliantZarr, CFCompliantZarrConfig


@pytest.mark.parametrize(
    "dt_fixture",
    [
        "output_datatree_without_scores",
        "output_datatree_with_scores",
    ],
)
def test_cf_compliant_netcdf_write(
    request: pytest.FixtureRequest,
    dt_fixture: str,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test writing data to a cf-compliant NetCDF file."""
    output_datatree: xr.DataTree = request.getfixturevalue(dt_fixture)
    datasink_cf_compliant_netcdf = CFCompliantNetCDF(
        CFCompliantNetCDFConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_netcdf,
            crs="EPSG:4326",
            directory=Path(tmpdir),
            filename="test_output.nc",
            general=xarray_general_info_config,
        ),
    )
    assert output_datatree is not None
    assert isinstance(output_datatree, xr.DataTree)
    datasink_cf_compliant_netcdf.write_data(
        output_datatree,
    )
    # File name is derived from the configured filename plus the verification pair id.
    assert (Path(tmpdir) / "test_output_test_pair.nc").exists()


def test_cf_compliant_netcdf_write_multiple_pairs(
    output_datatree_with_multiple_pairs: xr.DataTree,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test that one distinct NetCDF file is written per verification pair."""
    datasink_cf_compliant_netcdf = CFCompliantNetCDF(
        CFCompliantNetCDFConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_netcdf,
            crs="EPSG:4326",
            directory=Path(tmpdir),
            filename="test_output.nc",
            general=xarray_general_info_config,
        ),
    )
    datasink_cf_compliant_netcdf.write_data(output_datatree_with_multiple_pairs)

    for pair_id in ("test_pair_1", "test_pair_2"):
        filepath = Path(tmpdir) / f"test_output_{pair_id}.nc"
        assert filepath.exists()
        with xr.open_dataset(filepath) as written:
            assert cast("str", written.attrs["institution"]) == "Test Institution"  # type: ignore[misc]


@pytest.mark.parametrize(
    "dt_fixture",
    [
        "output_datatree_without_scores",
        "output_datatree_with_scores",
    ],
)
def test_cf_compliant_zarr_write_local(
    request: pytest.FixtureRequest,
    dt_fixture: str,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test writing data to a cf-compliant Zarr store."""
    output_datatree: xr.DataTree = request.getfixturevalue(dt_fixture)
    datasink_cf_compliant_zarr = CFCompliantZarr(
        CFCompliantZarrConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_zarr,
            crs="EPSG:4326",
            path=f"{tmpdir!s}/test_output.zarr",
            general=xarray_general_info_config,
        ),
    )
    assert output_datatree is not None
    assert isinstance(output_datatree, xr.DataTree)
    datasink_cf_compliant_zarr.write_data(
        output_datatree,
    )
    assert (Path(datasink_cf_compliant_zarr.config.path)).exists()
