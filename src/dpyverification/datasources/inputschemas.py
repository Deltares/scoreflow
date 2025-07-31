"""A collection of schemas for input data."""

# mypy: ignore-errors
# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

AllowedDTypeInt = Literal["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
AllowedDTypeFloat = Literal["float16", "float32", "float64"]
AllowedDTypeDateTime = Literal["datetime64[ns]"]
AllowedDTypeTimeDelta = Literal["timedelta64[ns]"]


def tuple_contains_elements(
    required: set[str],
    optional: set[str] | None = None,
) -> callable:
    def validator(cls, v: tuple[str, ...]) -> tuple[str, ...]:  # noqa: ANN001, ARG001
        v_set = set(v)

        # Compute allowed elements
        allowed = required.union(optional) if optional is not None else required

        # Validate the provided elements are a subset of allowed elements
        if not v_set.issubset(allowed):
            invalid = v_set - allowed
            msg = f"contains invalid elements not in required or optional sets: {invalid}"
            raise ValueError(
                msg,
            )

        # Validate the prodived elements are a subset of the required elements
        if not required.issubset(v_set):
            missing = required - v_set
            msg = f"missing required elements: {missing}"
            raise ValueError(msg)

        return v

    return validator


class SharedDims(BaseModel):
    time: int
    stations: int


class ObsDims(SharedDims):
    pass


class TimeCoord(BaseModel):
    dims: tuple
    dtype: AllowedDTypeDateTime

    @field_validator("dims")
    @classmethod
    def check_tuple(cls, v: tuple) -> tuple:
        if v != ("time",):
            msg = f"value must be ('time',), got {v}"
            raise ValueError(msg)
        return v


class ForecastReferenceTimeCoord(BaseModel):
    dims: tuple
    dtype: AllowedDTypeDateTime

    @field_validator("dims")
    @classmethod
    def check_tuple(cls, v: tuple) -> tuple:
        if v != ("forecast_reference_time",):
            msg = f"value must be ('forecast_reference_time',), got {v}"
            raise ValueError(msg)
        return v


class StationsCoord(BaseModel):
    dims: tuple

    @field_validator("dims")
    @classmethod
    def check_tuple(cls, v: tuple) -> tuple:
        tuple_contains_elements(
            required={"stations"},
        )(cls, v)


class XYZCoord(BaseModel):
    dims: tuple
    dtype: AllowedDTypeFloat

    @field_validator("dims")
    @classmethod
    def check_tuple(cls, v: tuple) -> tuple:
        tuple_contains_elements(
            required={"stations"},
        )(cls, v)


class SharedCoords(BaseModel):
    time: TimeCoord
    stations: StationsCoord
    x: XYZCoord | None
    y: XYZCoord | None
    z: XYZCoord | None


class ObsCoords(SharedCoords):
    pass


class XarrayObservationsDataArray(BaseModel):
    dims: tuple
    dtype: AllowedDTypeFloat

    @field_validator("dims")
    @classmethod
    def check_tuple(cls, v: tuple) -> tuple:
        tuple_contains_elements(
            required={"time", "stations"},
        )(cls, v)


ValidVarName = Annotated[str, Field(pattern=r"[a-zA-Z_]*")]


class XarrayDatasetObservations(BaseModel):
    dims: ObsDims
    coords: ObsCoords
    data_vars: dict[
        ValidVarName,
        XarrayObservationsDataArray,
    ]


class RealizationCoord(BaseModel):
    dims: tuple
    dtype: AllowedDTypeInt

    @field_validator("dims")
    @classmethod
    def check_tuple(cls, v: tuple) -> tuple:
        tuple_contains_elements(
            required={"realization"},
        )(cls, v)


class Sim1Dims(SharedDims):
    forecast_reference_time: int
    realization: int | None


class Sim1Coords(SharedCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    realization: RealizationCoord | None  # Optional to handle ensemble and deterministic forecasts


class XarrayDatasetSimulationsByForecastReferenceTime(BaseModel):
    dims: Sim1Dims
    coords: Sim1Coords


class Sim2Dims(SharedDims):
    forecast_period: int
    realization: int | None


class ForecastPeriodCoord(BaseModel):
    dims: tuple
    dtype: AllowedDTypeTimeDelta

    @field_validator("dims")
    @classmethod
    def check_tuple(cls, v: tuple) -> tuple:
        tuple_contains_elements(
            required={"forecast_period"},
        )(cls, v)


class Sim2Coords(SharedCoords):
    forecast_period: ForecastPeriodCoord
    realization: RealizationCoord | None  # Optional to handle ensemble and deterministic forecasts


class XarraySimulationsByForecastPeriodDataArray(BaseModel):
    dims: tuple[str, ...]

    @field_validator("dims")
    @classmethod
    def check_tuple(cls, v: tuple) -> tuple:
        tuple_contains_elements(
            required={"forecast_period", "time", "stations"},
            optional={"realization"},
        )(cls, v)


class XarrayDatasetSimulationsByForecastPeriod(BaseModel):
    dims: Sim2Dims
    coords: Sim2Coords
    data_vars: dict[
        ValidVarName,
        XarraySimulationsByForecastPeriodDataArray,
    ]
