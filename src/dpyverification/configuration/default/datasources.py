"""A module for default implementation of datasources."""

from typing import Annotated, Literal

from pydantic import Field

from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.configuration.utils import LeadTimes, LocalFile
from dpyverification.constants import DataSourceKind, SimObsKind


class FewsWebserviceConfig(BaseDatasourceConfig):
    """A basic fews webservice config element."""

    kind: Literal[DataSourceKind.FEWSWEBSERVICE]
    url: str


class FewsWebserviceInputConfig(FewsWebserviceConfig):
    """A fews webservice input config element."""

    location_ids: Annotated[list[str], Field(min_length=1)]
    parameter_ids: Annotated[list[str], Field(min_length=1)]
    module_instance_ids: Annotated[list[str], Field(min_length=1)]
    qualifier_ids: Annotated[list[str], Field(default=[])]
    _document_format: Annotated[
        Literal["PI_XML"],
        Field(
            description=(
                "A private attribute with a default value:"
                " 1. A literal with a default value, as users are not expected to set it."
                " 2. Private, since the code only supports this one option for now, but it is"
                " likely that other values may be needed in the future, depending on what FEWS"
                " system this code interacts with. Then this value does need to be configurable,"
                " therefore do have it as a configuration setting already."
            ),
        ),
    ] = "PI_XML"
    _document_version: Annotated[
        Literal["1.32"],
        Field(
            description=(
                "A private attribute with a default value:"
                " 1. A literal with a default value, as users are not expected to set it."
                " 2. Private, since the code only supports this one option for now, but it is"
                " likely that other values may be needed in the future, depending on what FEWS"
                " system this code interacts with. Then this value does need to be configurable,"
                " therefore do have it as a configuration setting already."
            ),
        ),
    ] = "1.32"


class FewsWebserviceInputObsConfig(FewsWebserviceInputConfig):
    """A fews webservice input obs config element."""

    simobstype: Literal[SimObsKind.OBS]


class FewsWebserviceInputSimConfig(FewsWebserviceInputConfig):
    """A fews webservice input sim config element."""

    simobstype: Literal[SimObsKind.SIM]
    leadtimes: Annotated[
        LeadTimes | None,
        Field(
            description="Value from General leadtimes used if not set.",
        ),
    ] = None
    forecastcount: Annotated[
        int,
        Field(
            description=(
                "Number of forecast runs to retrieve."
                " When value is 0 (default), ALL matching forecast runs will be used."
            ),
        ),
    ] = 0
    ensemble_id: str | None = None


class FewsWebserviceOutputConfig(FewsWebserviceConfig):
    """A fews webservice output config element."""


class FileInputPixmlConfig(BaseDatasourceConfig, LocalFile):
    """A file input of type pixml config element."""

    kind: Literal[DataSourceKind.PIXML]


class FileInputFewsnetcdfConfig(BaseDatasourceConfig, LocalFile):
    """A file input fewsnetcdf config element."""

    kind: Literal[DataSourceKind.FEWSNETCDF]
