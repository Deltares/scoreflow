"""Implementations for threshold datasources."""

from typing import Annotated

from pydantic import Field

from dpyverification.configuration.config import BaseThresholdsDatasourceConfig
from dpyverification.configuration.utils import LocalFile


class CsvFileConfig(LocalFile, BaseThresholdsDatasourceConfig):
    """Configuration for parsing thresholds from csv file."""

    stations: Annotated[list[str], Field(min_length=1)]
    variables: Annotated[list[str], Field(min_length=1)]
    thresholds: Annotated[list[str], Field(min_length=1)]
