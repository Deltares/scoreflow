"""Module with the base class that all datasources should inherit from."""

from abc import abstractmethod
from typing import Self

import xarray

from dpyverification.base import Base
from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.constants import SimObsKind


class BaseDatasource(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind: str = ""
    config_class: type[BaseDatasourceConfig] = BaseDatasourceConfig

    def __init__(self, config: BaseDatasourceConfig) -> None:
        self.config: BaseDatasourceConfig = config
        self.simobstype = config.simobstype
        self.xarray = xarray.Dataset()

    @property
    def simobstype(self) -> str:
        """Whether the instance represents sim or obs data."""
        return self.config.simobstype

    @simobstype.setter
    def simobstype(self, new_simobstype: SimObsKind) -> None:
        if new_simobstype not in (SimObsKind.SIM, SimObsKind.OBS):
            # Even if the underlying file or service can contain combined data, the creation of the
            #  datasource objects should split those. This assumption can then be used in the
            #  creation of the data model.
            msg: str = (
                "The simpobstype of a " + self.__class__.__name__ + " can only be either sim or obs"
            )
            raise ValueError(msg)
        self._simobstype = new_simobstype

    @abstractmethod
    def get_data(self) -> Self:
        """Get the data from the datasource and set it to the self.xarray an the instance."""
