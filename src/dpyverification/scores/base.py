"""An abstract implementation of a calculation."""

from abc import abstractmethod
from typing import ClassVar

import xarray as xr

from dpyverification.base import Base
from dpyverification.configuration.base import BaseCategoricalScoreConfig, BaseScoreConfig
from dpyverification.constants import DataType


class BaseScore(Base):
    """An abstract calculation class."""

    kind = ""  # to be defined by subclasses
    config_class: type[BaseScoreConfig] = BaseScoreConfig  # to be defined by subclasses
    supported_data_types: ClassVar[set[DataType]] = set()

    def __init__(self, config: BaseScoreConfig) -> None:
        self.config: BaseScoreConfig = config

    @abstractmethod
    def compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.DataArray | xr.Dataset:
        """Abstract calculation."""

    def validate_and_compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.DataArray | xr.Dataset:
        """Validate and compute."""
        data_type: DataType = sim.verification.data_type  # type:ignore[misc]
        if data_type not in self.supported_data_types:
            msg = f"The data type '{data_type} is not supported by"
            f"{self.__class__.__name__}. Supported types: "
            f"{sorted(self.supported_data_types)}."
            raise ValueError(msg)
        result = self.compute(obs, sim)
        if isinstance(result, xr.DataArray) and result.name is None:  # type:ignore[misc]
            result.name = self.config.score_adapter
        return result


class BaseCategoricalScore(Base):
    """An abstract calculation class for categorical scores."""

    kind = ""  # to be defined by subclasses
    config_class: type[BaseCategoricalScoreConfig] = (
        BaseCategoricalScoreConfig  # to be defined by subclasses
    )
    supported_data_types: ClassVar[set[DataType]] = set()

    def __init__(self, config: BaseCategoricalScoreConfig) -> None:
        self.config: BaseCategoricalScoreConfig = config

    @abstractmethod
    def compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
        thresholds: xr.DataArray,
    ) -> xr.DataArray | xr.Dataset:
        """Abstract calculation."""

    def validate_and_compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
        thresholds: xr.DataArray,
    ) -> xr.DataArray | xr.Dataset:
        """Validate and compute."""
        data_type: DataType = sim.verification.data_type  # type:ignore[misc]
        if data_type not in self.supported_data_types:
            msg = f"The data type '{data_type} is not supported by"
            f"{self.__class__.__name__}. Supported types: "
            f"{sorted(self.supported_data_types)}."
            raise ValueError(msg)
        result = self.compute(obs, sim, thresholds=thresholds)
        if isinstance(result, xr.DataArray) and result.name is None:  # type:ignore[misc]
            result.name = self.config.score_adapter
        return result
