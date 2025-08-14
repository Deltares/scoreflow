"""Utility functions for shared across the scores module."""

from typing import TypeAlias, Union, cast

import xarray as xr

from dpyverification.configuration.base import BaseConfig

NestedDict: TypeAlias = dict[str, Union[str, list[str], "NestedDict"]]


def set_data_array_attributes(
    da: xr.DataArray,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    config: BaseConfig | None = None,
) -> xr.DataArray:
    """Set configuration attributes on xr.DataArray."""
    cf_attrs: dict[str, str] = {
        "long_name": long_name,
        "units": units,
    }

    if standard_name is not None:
        cf_attrs.update({"standard_name": standard_name})

    union: NestedDict | dict[str, str]

    if config is not None:
        config_attrs = cast(NestedDict, config.__dict__)
        union = cf_attrs | config_attrs
    else:
        union = cf_attrs

    return da.assign_attrs(union)
