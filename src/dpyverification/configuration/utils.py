"""A module for frequently used config elements in the context of verification."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel

from dpyverification.constants import TimeUnits


class DateTime(BaseModel):
    """A datetime config element."""

    format: str = "%Y-%m-%dT%H:%M:%S%z"
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
    """A leadtimes config element."""

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
    """A time period config element."""

    start: DateTime
    end: DateTime


class SimObsVariables(BaseModel):
    """A simobs variables config element."""

    sim: str
    obs: str


class LocalFile(BaseModel):
    """A local file config element."""

    directory: str
    filename: str
