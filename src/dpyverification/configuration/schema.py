"""The schema definition for the configuration yaml file.

To generate a yaml / json file with the json representation of this schema:
    with FILEPATH.open() as myfile:
        yaml.dump(ConfigSchema.model_json_schema(), myfile)

To generate a pydantic schema from a yaml/json file, see datamodel_code_generator,
for example from https://docs.pydantic.dev/latest/integrations/datamodel_code_generator/
Note that this can generate a pydantic model that is not up-to-date with the latest
pydantic / python, and might need some modifications.
"""


# ruff: noqa: D101 Do not require class docstrings for the classes in this file

from datetime import datetime, timedelta
from typing import Annotated, Literal, TypeAlias

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from dpyverification.constants import CalculationTypeEnum, DataSourceTypeEnum, SimObsType, TimeUnits


class DateTime(BaseModel):
    format: str | None = "%Y-%m-%dT%H:%M:%S%z"
    value: str

    @property
    def datetime64(self) -> np.datetime64:
        """As numpy datetime64."""
        return pd.to_datetime(self.value, format=self.format).to_numpy()

    @property
    def datetime(self) -> datetime:
        """As datetime datetime."""
        return pd.to_datetime(self.value, format=self.format)


class LeadTimes(BaseModel):
    unit: TimeUnits
    values: list[int]

    @property
    def timedelta64(self) -> list[np.timedelta64]:
        """As numpy timedelta64."""
        return [np.timedelta64(v, self.unit) for v in self.values]

    @property
    def timedelta(self) -> list[timedelta]:
        """As datetime timedelta."""

        def convert_to_timedelta(value: int) -> timedelta:
            return np.timedelta64(value, self.unit).astype(timedelta)  # type: ignore[no-any-return, misc]

        return [convert_to_timedelta(v) for v in self.values]


class TimePeriod(BaseModel):
    start: DateTime
    end: DateTime


class FewsWebservice(BaseModel):
    datasourcetype: Literal[DataSourceTypeEnum.fewswebservice]
    url: str


class FewsWebserviceInput(FewsWebservice):
    simobstype: SimObsType


class FewsWebserviceOutput(FewsWebservice):
    pass


class LocalFile(BaseModel):
    directory: str
    filename: str


class FileInput(LocalFile):
    datasourcetype: Literal[DataSourceTypeEnum.pixml, DataSourceTypeEnum.fewsnetcdf]
    simobstype: SimObsType


class FewsNetcdfOutput(LocalFile):
    datasourcetype: Literal[DataSourceTypeEnum.fewsnetcdf]
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


DataSource: TypeAlias = (
    FewsWebserviceInput | FileInput
)  # A Type Alias for the combination of data source schema classes

Output: TypeAlias = (
    FewsWebserviceOutput | FewsNetcdfOutput
)  # A Type Alias for the combination of output schema classes


class SimObsPair(BaseModel):
    sim: str
    obs: str


class PinScore(BaseModel):
    calculationtype: Literal[CalculationTypeEnum.pinscore]


class SimObsPairs(BaseModel):
    calculationtype: Literal[CalculationTypeEnum.simobspairs]
    # One combination of list-of-leadtimes and list-of-variablepairs, use multiple SimObsPairs
    # to define more combinations
    leadtimes: LeadTimes | None = None  # Default from GeneralInfo
    variablepairs: list[SimObsPair]


Calculation: TypeAlias = (
    SimObsPairs | PinScore  # A Type Alias for the combination of calculation schema classes
)


class GeneralInfo(BaseModel):
    # Is this general info, or might it be different for different calculations?
    verificationperiod: TimePeriod
    leadtimes: LeadTimes | None = None


class ConfigSchema(BaseModel):
    output: Annotated[list[Output], Field(min_length=1)]
    calculations: Annotated[list[Calculation], Field(min_length=1)]
    datasources: Annotated[list[DataSource], Field(min_length=1)]
    general: GeneralInfo
    fileversion: str
