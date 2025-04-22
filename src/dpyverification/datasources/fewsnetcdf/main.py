"""Read and write netcdf files in a fews compatible format."""

from pathlib import Path
from typing import Self

import xarray as xr

from dpyverification.configuration import FileInputFewsnetcdfConfig
from dpyverification.constants import SimObsKind
from dpyverification.datasources.base import BaseDatasource

from .schema import FewsNetcdfFileInputSchema


class FewsNetcdfFileSource(BaseDatasource):
    """For reading data from, and writing data to, a fews netcdf file."""

    kind = "fewsnetcdf"
    config_class = FileInputFewsnetcdfConfig

    def __init__(self, config: FileInputFewsnetcdfConfig) -> None:
        self.config: FileInputFewsnetcdfConfig = config

    @staticmethod
    def _nc_to_xarray(path: Path, kind: str) -> xr.Dataset:
        """Read fews netcdf file and return xr.Dataset.

        Compatible with both observations and (ensemble) forecasts.

        Parameters
        ----------
        path : Path
            Path to the netcdf file
        kind : Literal["sim", "obs"]
            String indicating the kind. Should be either sim (for simulations)
             or obs (for observations).


        Returns
        -------
        xr.Dataset
            Dataset representation of the fews netcdf file.

        Raises
        ------
        TypeError
            Raised when pd.DataFrame.to_xarray() does not return xr.DataArray.
        """
        ds = xr.open_dataset(path)

        # Verify the structure of the dataset against known schema
        schema_like = ds.to_dict()  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the model
        # Assign to _, since the model will throw an error when not compliant
        _ = FewsNetcdfFileInputSchema(**schema_like)  # type: ignore[misc]

        if kind == SimObsKind.OBS and "ensemble_member" in ds.coords:
            # Can this happen? What to do? Squeeze it out like in pixml file?
            raise NotImplementedError

        raise NotImplementedError
        # From here on, may need to also adapt how datamodel uses an xarray
        # DataModel only tested for single parameter inputs for now, not yet multiple
        # Need to check what coordinates an obs, and a sim, can have, and if that matches the
        #  DataModel expectations on inputs
        return ds  # type: ignore[unreachable] # yes, for now this is unreachable, but do want to keep it

    def get_data(self) -> Self:
        """Retrieve fewsnetcdf content as an xarray DataArray."""
        if self.config.simobstype == SimObsKind.COMBINED:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        filepath = Path(self.config.directory) / self.config.filename
        self.xarray = self._nc_to_xarray(filepath, self.config.simobstype)
        return self
