"""A module for default implementation of datasources."""

from typing import Annotated, Literal

from pydantic import Field

from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.configuration.utils import FewsWebserviceAuthConfig, LocalFile
from dpyverification.constants import DataSourceKinds, SimObsKinds


class FewsWebserviceInputConfig(BaseDatasourceConfig):
    """A fews webservice input config element."""

    kind: Literal[DataSourceKinds.FEWSWEBSERVICE]
    auth_config: FewsWebserviceAuthConfig
    location_ids: Annotated[list[str], Field(min_length=1)]
    parameter_ids: Annotated[list[str], Field(min_length=1)]
    module_instance_ids: Annotated[list[str], Field(min_length=1)]
    qualifier_ids: Annotated[list[str], Field(default=None)]


class FewsWebserviceInputObsConfig(FewsWebserviceInputConfig):
    """Fews webservice config for obs and sim."""

    simobstype: Literal[SimObsKinds.OBS]


class FewsWebserviceInputSimConfig(FewsWebserviceInputConfig):
    """A fews webservice input sim config element."""

    simobstype: Literal[SimObsKinds.SIM]
    ensemble_id: Annotated[list[str], Field(default=None)]
    ensemble_member_id: Annotated[list[int], Field(default=None)]


class FewsWebserviceOutputConfig(FewsWebserviceInputConfig):
    """A fews webservice output config element."""


class FileInputFewsnetcdfConfig(BaseDatasourceConfig, LocalFile):
    """A file input fewsnetcdf config element."""

    kind: Literal[DataSourceKinds.FEWSNETCDF]
