"""An abstract implementation of a calculation."""

from abc import abstractmethod

import xarray as xr

from dpyverification.base import Base
from dpyverification.configuration.base import BaseScoreConfig
from dpyverification.datamodel import DataModel


class BaseScore(Base):
    """An abstract calculation class."""

    kind = ""  # to be defined by subclasses
    config_class: type[BaseScoreConfig] = BaseScoreConfig  # to be defined by subclasses

    def __init__(self, config: BaseScoreConfig) -> None:
        self.config: BaseScoreConfig = config

    @abstractmethod
    def compute(
        self,
        data: DataModel,
    ) -> xr.Dataset | xr.DataArray:
        """Abstract calculation."""
