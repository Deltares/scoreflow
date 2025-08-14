"""Module for reading from and writing to a fews webservice."""

from typing import Self

import xarray as xr

from dpyverification.api.fewswebservice import FewsWebserviceClient
from dpyverification.configuration import (
    FewsWebserviceInputConfig,
)
from dpyverification.constants import SimObsKind
from dpyverification.datasources.base import BaseDatasource


class FewsWebservice(BaseDatasource):
    """For downloading data using a Delft-FEWS webservice."""

    # TODO(AU): Fix and document timezone information in fewswebservice requests # noqa: FIX002
    #   A hardcoded Z is added at the end, that cannot be right? See the issue for details:
    #   https://github.com/Deltares-research/DPyVerification/issues/43
    #

    kind = "fewswebservice"
    config_class = FewsWebserviceInputConfig
    # Annotate the correct type, otherwise mypy will infer from baseclass
    config: FewsWebserviceInputConfig
    # The datetime format that is used to pass datetimes to the fewswebservice
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    timeout = 30

    def __init__(self, config: FewsWebserviceInputConfig) -> None:
        self.config = config
        self.simobskind = config.simobskind
        self.dataset = xr.Dataset()

        # Initialize the webservice client
        self.client = FewsWebserviceClient(
            url=self.config.auth_config.url.unicode_string(),
            username=self.config.auth_config.username.get_secret_value(),
            password=self.config.auth_config.password.get_secret_value(),
        )

    def get_data(self) -> Self:
        """Retrieve :py::class`~xarray.Dataset` from Delft-FEWS Webservice."""
        # Get observations
        if self.config.simobskind == SimObsKind.obs:
            msg = "Observations are not yet supported."
            raise NotImplementedError(msg)

        # Get simulations
        if self.config.simobskind == SimObsKind.sim:
            # Implement forecast retrieval, once Delft-FEWS development is completed.
            msg = "Simulations are not yet supported."
            raise NotImplementedError(msg)

        msg = f"Simobskind {self.simobskind} not implemented yet. Only sim and obs are supported."
        raise NotImplementedError(msg)
