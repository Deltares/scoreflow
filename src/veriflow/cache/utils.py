"""Utility functions for cache management."""

import xarray as xr

from veriflow.constants import StandardDim


def combine_cached_and_fetched_data(
    cached_dataset: xr.Dataset,
    fetched_dataset: xr.Dataset,
    dim: str,
) -> xr.Dataset:
    """Combine the cached and fetched dataset along the given dimensions."""
    if dim == "variable":
        # Add a variable to the dataset, or update it if it already exists.
        return xr.merge(  # type: ignore[no-any-return, call-overload, misc]
            [cached_dataset, fetched_dataset],
            combine_attrs="override",
        )
    if dim in [
        StandardDim.station,
        StandardDim.forecast_reference_time,
        StandardDim.lead_time,
        StandardDim.time,
    ]:
        # Update the dataset by updating values along the specified dimension.
        return xr.concat([cached_dataset, fetched_dataset], dim=dim)
    msg = (
        f"Unsupported dimension '{dim}' for cache update. Must be one of 'variable', "
        f"'{StandardDim.station}', '{StandardDim.forecast_reference_time}', "
        f"'{StandardDim.lead_time}' or '{StandardDim.time}'."
    )
    raise ValueError(msg)
