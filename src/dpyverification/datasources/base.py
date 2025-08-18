"""Module with the base class that all datasources should inherit from."""

import hashlib
from abc import abstractmethod
from typing import Self

import xarray
import xarray as xr

from dpyverification.base import Base
from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.constants import SimObsKind


class BaseDatasource(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind: str = ""
    config_class: type[BaseDatasourceConfig] = BaseDatasourceConfig

    def __init__(self, config: BaseDatasourceConfig) -> None:
        self.config: BaseDatasourceConfig = config
        self.simobskind = config.simobskind
        self.dataset = xarray.Dataset()

    @property
    def simobskind(self) -> str:
        """Whether the instance represents sim or obs data."""
        return self.config.simobskind

    @simobskind.setter
    def simobskind(self, new_simobskind: SimObsKind) -> None:
        if new_simobskind not in (SimObsKind.sim, SimObsKind.obs):
            # Even if the underlying file or service can contain combined data, the creation of the
            #  datasource objects should split those. This assumption can then be used in the
            #  creation of the data model.
            msg: str = (
                "The simobskind of a " + self.__class__.__name__ + " can only be either sim or obs"
            )
            raise ValueError(msg)
        self._simobskind = new_simobskind

    @abstractmethod
    def fetch_data(self) -> Self:
        """Fetch data from datasource and assign xr.Dataset to the self.dataset variable."""

    def get_data(self) -> Self:
        """Get cached data, or fetch and cache."""
        config_json = self.config.model_dump_json().encode("utf-8")
        config_hash = hashlib.sha256(config_json).hexdigest()

        cached_dataset_path = (
            self.config.general.cache_dir / f"{self.__class__.__name__}_{config_hash}.nc"
        )
        if cached_dataset_path.exists():
            self.dataset = xr.open_dataset(cached_dataset_path)
            return self
        # Go fetch and cache
        self.fetch_data()
        self.dataset.to_netcdf(cached_dataset_path)
        return self
