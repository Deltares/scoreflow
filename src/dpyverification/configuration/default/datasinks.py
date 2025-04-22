"""A module for default implementation of datasinks."""

from typing import Annotated, Literal

from pydantic import Field

from dpyverification.configuration.base import BaseDatasinkConfig
from dpyverification.configuration.utils import LocalFile
from dpyverification.constants import DataSinkKind


class FewsNetcdfOutputConfig(LocalFile, BaseDatasinkConfig):
    """A fews netcdf output config element."""

    kind: Literal[DataSinkKind.FEWSNETCDF]
    title: Annotated[
        str | None,
        Field(
            description=(
                "Value for the title attribute in the generated netcdf."
                " A title will be generated if not provided"
            ),
        ),
    ] = None
    institution: Annotated[
        str,
        Field(description="Value for the institution attribute in the generated netcdf."),
    ] = "Deltares"
