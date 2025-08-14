"""A module for default implementation of datasources."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.configuration.utils import FewsWebserviceAuthConfig, LocalFiles
from dpyverification.constants import DataSourceKind, SimObsKind


class FewsNetcdfKind(StrEnum):
    """List of kinds of FEWS NetCDFs."""

    observation = "observation"
    one_full_simulation = "one_full_simulation"
    simulation_for_one_forecast_period = "simulation_for_one_forecast_period"


class FewsWebserviceInputConfig(BaseDatasourceConfig):
    """A fews webservice input config element."""

    kind: Literal[DataSourceKind.FEWSWEBSERVICE]
    auth_config: FewsWebserviceAuthConfig
    location_ids: Annotated[list[str], Field(min_length=1)]
    parameter_ids: Annotated[list[str], Field(min_length=1)]
    module_instance_ids: Annotated[list[str], Field(min_length=1)]
    qualifier_ids: Annotated[list[str], Field(default=None)]


class FewsWebserviceInputObsConfig(FewsWebserviceInputConfig):
    """Fews webservice config for obs and sim."""

    simobskind: Literal[SimObsKind.obs]


class FewsWebserviceInputSimConfig(FewsWebserviceInputConfig):
    """A fews webservice input sim config element."""

    simobskind: Literal[SimObsKind.sim]
    ensemble_id: Annotated[list[str], Field(default=None)]
    ensemble_member_id: Annotated[list[int], Field(default=None)]


class FewsWebserviceOutputConfig(FewsWebserviceInputConfig):
    """A fews webservice output config element."""


class FileInputFewsnetcdfConfig(BaseDatasourceConfig, LocalFiles):
    """A file input fewsnetcdf config element."""

    kind: Literal[DataSourceKind.FEWSNETCDF]
    netcdf_kind: FewsNetcdfKind
    station_ids: Annotated[list[str], Field(min_length=1)] | None = None
