"""Create simobspair for specific lead times."""

from typing import TYPE_CHECKING

import xarray
from numpy import timedelta64

from dpyverification.configuration import SimObsPairsConfig
from dpyverification.configuration.base import SimObsVariables
from dpyverification.constants import DataModelCoords, ScoreKind
from dpyverification.datamodel import SimObsDataset
from dpyverification.scores.base import BaseScore

if TYPE_CHECKING:
    from numpy import datetime64


class SimObsPairs(BaseScore):
    """SimObsPairs implementation."""

    kind = "simobspairs"
    config_class = SimObsPairsConfig

    def __init__(self, config: SimObsPairsConfig) -> None:
        self.config: SimObsPairsConfig = config

    def compute(
        self,
        data: SimObsDataset,
    ) -> xarray.Dataset:
        """Create pairs of obs and sim values, for the given leadtimes (default leadtime 0)."""
        if not self.config.leadtimes:
            # When called from pipeline, this should not be possible. However, do need to check in
            # case this function is called from a custom implementation.
            msg = "No leadtimes specified in SimObsPairs configuration"
            raise ValueError(msg)
        # Note that leadtimes from calcconfig are used, that may be different (i.e. a subset) from
        # the leadtimes dimension of the intermediate dataset
        leadtimes = self.config.leadtimes.timedelta64
        variablepairs = self.config.variablepairs
        intermediate_dataset = data.intermediate

        # add checks on input_dataset here?
        return SimObsPairs._simobs(intermediate_dataset, leadtimes, variablepairs)

    @staticmethod
    def _simobs(
        intermediate_dataset: xarray.Dataset,
        leadtimes: list[timedelta64],
        variablepairs: list[SimObsVariables],
    ) -> xarray.Dataset:
        leadsets = []
        for leadtime in leadtimes:
            leadset = intermediate_dataset.coords.to_dataset()
            selecttime: list[datetime64] = list(
                intermediate_dataset[DataModelCoords.simstart.name].data + leadtime,  # type: ignore[misc] # Quite certain that data.input[DataModelCoords.time.name].data will be a 1D array of datetime64
            )
            for pair in variablepairs:
                # Construct variable names:
                #   varnamegeneral_calctypename_varname
                # Where
                # - varnamegeneral is assumed equal to obsvar name
                # - calctypename is taken to be equal to enum string value
                outnamesim = f"{pair.obs}_{ScoreKind.SIMOBSPAIRS}_{pair.sim}"
                outnameobs = f"{pair.obs}_{ScoreKind.SIMOBSPAIRS}_{pair.obs}"

                # TODO(AU): Add unit test on simobspair creation # noqa: FIX002
                #   https://github.com/Deltares-research/DPyVerification/issues/33
                #   Have a unit test with leadtimes that are incompatible with the available data.
                #   Additional thoughts on that, from earlier:
                #   Wait for adaptation of example input files, when smaller, can use large leadtime
                #   to be beyond end. To test what if any newtime values are not part of the input
                #   time dimension? -> Will give KeyError. What do we want to do in that case?
                #   Skip leadtime entirely? Or do create, but fully empty?
                #   Truncate newtime at min and max of time can be a first step,
                #   to only get valid time values. But what if newtime is then empty?

                # Parse the obs values
                select_at: dict[str, list[datetime64] | list[timedelta64]]
                select_at = {
                    DataModelCoords.time.name: selecttime,
                }
                vals = intermediate_dataset[pair.obs].sel(select_at)
                leadset[outnameobs] = vals.expand_dims(
                    dim={"leadtime": [leadtime]},
                    axis=len(vals.dims),
                )
                if "units" in intermediate_dataset[pair.obs].attrs:  # type: ignore[misc] # attrs is a dict[Any,Any]
                    leadset[outnameobs].attrs.update(  # type: ignore[misc] # attrs is a dict[Any,Any]
                        {"units": intermediate_dataset[pair.obs].attrs["units"]},  # type: ignore[misc] # attrs is a dict[Any,Any]
                    )

                # Parse the sim values
                select_at[DataModelCoords.leadtime.name] = [leadtime]
                vals = intermediate_dataset[pair.sim].sel(select_at)
                leadset[outnamesim] = vals
                if "units" in intermediate_dataset[pair.sim].attrs:  # type: ignore[misc] # attrs is a dict[Any,Any]
                    leadset[outnamesim].attrs.update(  # type: ignore[misc] # attrs is a dict[Any,Any]
                        {"units": intermediate_dataset[pair.sim].attrs["units"]},  # type: ignore[misc] # attrs is a dict[Any,Any]
                    )
            leadsets.append(leadset)
        # merge will expand time to cover all leadtimes
        return xarray.merge(leadsets)
