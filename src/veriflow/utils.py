"""General utility functions for Veriflow."""

import xarray as xr


def convert_byte_string_coord_to_utf8(ds: xr.Dataset, coord: str) -> xr.Dataset:
    """Convert byte strings in the dataset to regular strings."""
    if ds[coord].dtype.kind == "S":  # type:ignore[misc]
        ds = ds.assign_coords({coord: ds[coord].astype(str)})  # type:ignore[misc]
    return ds
