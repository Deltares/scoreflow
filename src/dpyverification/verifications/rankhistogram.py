"""Compute the rank histogram (Talagrand diagram) over the specified dimensions.

For external documentation, see below:
https://xskillscore.readthedocs.io/en/stable/api/xskillscore.rank_histogram.html?highlight=rank%20histogram#xskillscore.rank_histogram
"""

import numpy as np
import xarray as xr
from xskillscore import rank_histogram as _rank_histogram  # type: ignore[import-untyped]

from dpyverification.configuration import Calculation, RankHistogram
from dpyverification.constants import (
    DataModelDims,
)
from dpyverification.datamodel import DataModel


def rankhistogram(
    calcconfig: Calculation,
    data: DataModel,
) -> xr.Dataset:
    """Compute the histogram of ranks over the specified dimensions."""
    if not isinstance(calcconfig, RankHistogram):
        msg = "Input calcconfig does not have calculationtype RankHistogram"
        raise TypeError(msg)

    # Select sim and obs.
    obs = data.intermediate[calcconfig.variablepair.obs]
    sim = data.intermediate[calcconfig.variablepair.sim]

    rankhistograms_per_leadtime = []
    for leadtime in sim[DataModelDims.leadtime]:  # type: ignore[misc]
        # Get a subset of the simulations dataset
        sim_subset = sim.sel(leadtime=leadtime)  # type: ignore[misc]

        # Compute the rank for
        _rank: xr.DataArray | xr.Dataset = _rank_histogram(
            observations=obs,
            forecasts=sim_subset,
            dim=calcconfig.reduce_dims,
            member_dim=DataModelDims.ensemble,
        )

        # Check a DataArray is returned
        if not isinstance(_rank, xr.DataArray):  # type: ignore[misc]
            msg = f"Expected xr.DataArray, got {type(_rank)}"
            raise TypeError(msg)

        # Set the variable name with specific lead time
        leadtime_seconds = int(leadtime.to_numpy() / np.timedelta64(1, "s"))  # type: ignore[misc]
        name = f"rank_histogram_leadtime_{leadtime_seconds}s"
        _rank.name = name

        # Set the long_name attribute on the variable
        # Set units to 1 (CF-compliant indication for dimensionless variable)
        _rank.attrs = {"long_name": name, "units": 1}

        # Append to the list
        rankhistograms_per_leadtime.append(_rank)

    # Compute
    data_vars = {k.name: k for k in rankhistograms_per_leadtime}
    ranks_for_all_leadtimes = xr.Dataset(data_vars=data_vars)

    # Set attrs on xr.DataArray
    # For now, store config as dict
    # General config could be stored as xr.Dataset attrs
    ranks_for_all_leadtimes.attrs = {str(k): str(v) for k, v in calcconfig.__dict__.items()}  # type: ignore[misc]

    return ranks_for_all_leadtimes
