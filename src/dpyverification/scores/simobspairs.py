"""Create simobspair for specific lead times."""

import xarray

from dpyverification.configuration import SimObsPairsConfig
from dpyverification.datamodel import SimObsDataset
from dpyverification.scores.base import BaseScore


class SimObsPairs(BaseScore):
    """SimObsPairs implementation."""

    kind = "simobspairs"
    config_class = SimObsPairsConfig

    def __init__(self, config: SimObsPairsConfig) -> None:
        self.config: SimObsPairsConfig = config

    def compute(
        self,
        data: SimObsDataset,
    ) -> xarray.Dataset:
        """Create pairs of obs and sim values, for the given forecast periods (default forecast_period 0)."""
        return data.dataset
