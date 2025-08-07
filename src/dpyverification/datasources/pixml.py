"""PI-XML support module."""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import xarray as xr
from fewsio.pi import (  # type: ignore[import-untyped] # See comment below imports
    Timeseries,
    TimeseriesId,
)

from dpyverification.configuration import FileInputPixmlConfig
from dpyverification.constants import (
    SimObsKinds,
    StandardCoords,
    StandardDims,
)
from dpyverification.datasources.base import BaseDatasource

if TYPE_CHECKING:
    import datetime


# Ignore import untyped on fewsio since no type stub available. Unfortunately, this also means
#  that almost all locations where types from fewsio are used, need to have a type: ignore[misc],
#  because those types are seen as Any


class PiXmlFile(BaseDatasource):
    """For reading data from a pixml file."""

    kind = "pixml"
    config_class = FileInputPixmlConfig

    def __init__(self, config: FileInputPixmlConfig) -> None:
        self.config: FileInputPixmlConfig = config
        self.simobstype = config.simobstype

    @staticmethod
    def pi_xml_to_xarray(path: Path, simobstype: str) -> xr.Dataset:
        """Convert pi-xml to an :py:class:`~xarray.Dataset`.

        Parameters
        ----------
        path : Path
            Path to file.
        simobstype : SimObsType
            Indicator for what type of data is contained in the file.

        Returns
        -------
        xr.Dataset
            :py:class:`~xarray.Dataset` representation of the pi-xml file.
        """
        attrs: dict[str, str]

        # Load  pi-xml file
        pi_series: Timeseries = Timeseries(path, binary=False)  # type: ignore[misc] # Timeseries has no type hinting, so pi_series is Any
        times: list[datetime.datetime] = pi_series.times  # type: ignore[misc] # pi_series is Any
        variables: set[str] = {k.parameter_id for k, _ in pi_series.items()}  # type: ignore[misc]  # pi_series is Any, and the keys from the items are Any, we are assuming str
        if len(variables) != 1:
            msg = "More than one parameter found."
            raise NotImplementedError(msg)
        variable_name = variables.pop()
        data_arrays = []

        def get_location_info(
            pi_series: Timeseries,
            timeseries_id: TimeseriesId,
        ) -> tuple[str, float, float]:
            location_info = pi_series.get_location(timeseries_id.location_id)  # type: ignore[misc] # location_info is Any
            lat = float(location_info.lat)  # type: ignore[misc] # lat is Any, we are assuming float convertable
            lon = float(location_info.lon)  # type: ignore[misc] # lat is Any, we are assuming float convertable
            if not math.isfinite(lat) or not math.isfinite(lon):
                msg = (
                    f"Lat ({lat}) and lon ({lon}) must be finite, from file {pi_series.path.name}."  # type: ignore[misc] # pi_series is Any
                )
                raise ValueError(msg)
            return str(timeseries_id.location_id), lat, lon  # type: ignore[misc] # timeseries_id is Any

        if simobstype == SimObsKinds.SIM:
            simulation_starttime: datetime.datetime = pi_series.forecast_datetime  # type: ignore[misc]  # pi_series is Any
            ensemble_member: int
            for ensemble_member in range(pi_series.ensemble_size):  # type: ignore[misc]  # pi_series is Any
                timeseries_id: TimeseriesId
                for timeseries_id, data in pi_series.items(  # type: ignore[misc] # pi_series and data are Any
                    ensemble_member=ensemble_member,
                ):
                    location_id, lat, lon = get_location_info(pi_series, timeseries_id)  # type: ignore[misc]  # pi_series is Any
                    coords = {  # separate variable for readability and type hinting
                        StandardCoords.time.name: times,
                        StandardCoords.location.name: [location_id],
                        StandardCoords.realization.name: [ensemble_member],
                        StandardCoords.lat.name: ([StandardDims.stations], [lat]),
                        StandardCoords.lon.name: ([StandardDims.stations], [lon]),
                        StandardCoords.forecast_reference_time.name: [simulation_starttime],
                    }
                    attrs = {"units": pi_series.get_unit(timeseries_id)}  # type: ignore[misc]  # pi_series is Any
                    da = xr.DataArray(
                        data=np.expand_dims(data, axis=(1, 2, 3)),  # type: ignore[misc] # data and ndarray are Any
                        dims=[
                            StandardDims.time,
                            StandardDims.stations,
                            StandardDims.realization,
                            StandardDims.forecast_reference_time,
                        ],
                        coords=coords,
                        attrs=attrs,
                    )
                    da.name = variable_name
                    data_arrays.append(da)

        elif simobstype == SimObsKinds.OBS:
            for timeseries_id, data in pi_series.items():  # type: ignore[misc] # pi_series and data are Any
                location_id, lat, lon = get_location_info(pi_series, timeseries_id)  # type: ignore[misc]  # pi_series is Any
                coords = {
                    StandardCoords.time.name: times,
                    StandardCoords.location.name: [location_id],
                    StandardCoords.lat.name: ([StandardDims.stations], [lat]),
                    StandardCoords.lon.name: ([StandardDims.stations], [lon]),
                }
                attrs = {"units": pi_series.get_unit(timeseries_id)}  # type: ignore[misc]  # pi_series is Any
                da = xr.DataArray(
                    data=np.expand_dims(data, axis=(1)),  # type: ignore[misc] # data and ndarray are Any
                    dims=[
                        StandardDims.time,
                        StandardDims.stations,
                    ],
                    coords=coords,
                    attrs=attrs,
                )
                da.name = variable_name
                data_arrays.append(da)
        else:
            msg = f"{simobstype} not supported."
            raise NotImplementedError(msg)
        return xr.merge(data_arrays)

    def get_data(self) -> Self:
        """Retrieve pixml content as an xarray DataArray."""
        if self.simobstype == SimObsKinds.COMBINED:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        filepath = Path(self.config.directory) / self.config.filename
        self.xarray = self.pi_xml_to_xarray(filepath, self.config.simobstype)
        return self
