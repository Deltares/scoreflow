"""Module for reading from and writing to a fews webservice."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Self

import requests

from dpyverification.configuration import FewsWebserviceInputConfig, FewsWebserviceInputSimConfig
from dpyverification.datasources.base import BaseDatasource
from dpyverification.datasources.pixml import PiXmlFile


class FewsWebService(BaseDatasource):
    """For downloading data using a Delft-FEWS webservice."""

    # TODO(AU): Fix and document timezone information in fewswebservice requests # noqa: FIX002
    #   A hardcoded Z is added at the end, that cannot be right? See the issue for details:
    #   https://github.com/Deltares-research/DPyVerification/issues/43
    #
    # The datetime format that is used to pass datetimes to the fewswebservice
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    timeout = 30

    def get_timeseries_xml_string(self) -> requests.Response:
        """Perform a REST GET to retrieve timeseries data as a pi-xml string."""
        if not isinstance(self.config, FewsWebserviceInputConfig):
            msg = "Provided config is not valid. Expected FewsWebserviceInput."
            raise TypeError(msg)

        ########## DOING: update de comments, per de PR comment
        endpoint = "timeseries"
        url = self.config.url + "/" + endpoint
        if self.config.verificationperiod is None:
            msg = "No verificationperiod specified."
            raise ValueError(msg)
        start = self.config.verificationperiod.start.datetime
        end = self.config.verificationperiod.end.datetime

        params = {
            "locationIds": self.config.location_ids,
            "parameterIds": self.config.parameter_ids,
            "moduleInstanceIds": self.config.module_instance_ids,
            "qualifierIds": self.config.qualifier_ids,
            "startTime": datetime.strftime(start, FewsWebService.datetime_format),
            "endTime": datetime.strftime(end, FewsWebService.datetime_format),
            "documentFormat": self.config._document_format,  # noqa: SLF001 # This config private member is meant to be used directly
            "documentVersion": self.config._document_version,  # noqa: SLF001 # This config private member is meant to be used directly
        }

        if isinstance(self.config, FewsWebserviceInputSimConfig):
            if self.config.leadtimes is None:
                msg = "No lead times specified for simulation."
                raise ValueError(msg)
            if self.config.forecastcount != 1:
                # TODO(AU): Issues 44 and 45 # noqa: FIX002
                #   See the more detailed split up in the get_data() for more info, and
                #   https://github.com/Deltares-research/DPyVerification/issues/44
                #   https://github.com/Deltares-research/DPyVerification/issues/45
                msg = (
                    "Retrieving ALL forecasts within a period not yet implemented,"
                    " specify a (very large) forecastcount value for now."
                )
                raise NotImplementedError(msg)

            # Work out the correct forecastStartTime and forecastEndTime
            # so that all forecasts overlapping with the verification period
            # defined by start_time and end_time will be requested from the web service.
            start_forecast_time = start - max(self.config.leadtimes.timedelta)
            end_forecast_time = end

            params.update(
                {
                    "startForecastTime": datetime.strftime(
                        start_forecast_time,
                        FewsWebService.datetime_format,
                    ),
                    "endForecastTime": datetime.strftime(
                        end_forecast_time,
                        FewsWebService.datetime_format,
                    ),
                    "forecastCount": str(self.config.forecastcount),
                },
            )

        response = requests.get(url=url, params=params, timeout=FewsWebService.timeout)
        response.raise_for_status()
        return response

    def get_data(self) -> Self:
        """Retrieve :py::class`~xarray.Dataset` from Delft-FEWS Webservice."""
        if not isinstance(self.config, FewsWebserviceInputConfig):
            msg = "Input dsconfig does not have kind FewsWebserviceInput"
            raise TypeError(msg)
        if isinstance(self.config, FewsWebserviceInputSimConfig) and self.config.forecastcount != 1:
            if self.config.forecastcount == 0:
                # TODO(AU): Implement ability to retrieve all forecastruns in period # noqa: FIX002
                #   First, issue 44 should be fixed. Then, see this issue for details and solution
                #   direction
                #   https://github.com/Deltares-research/DPyVerification/issues/45
                msg = (
                    "Retrieving ALL forecasts within a period not yet implemented,"
                    " specify a (very large) forecastcount value for now."
                )
            else:
                # TODO(AU): Implement ability to retrieve more than one forecastrun # noqa: FIX002
                #   See issue for details and solution direction
                #   https://github.com/Deltares-research/DPyVerification/issues/44
                msg = (
                    "Retrieving more than one forecast within a period not yet implemented,"
                    " due to fews-io package limitation in converting pixml files."
                )
            raise NotImplementedError(msg)

        response = self.get_timeseries_xml_string()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(response.content)
        self.xarray = PiXmlFile.pi_xml_to_xarray(Path(temp_file_path), self.config.simobstype)

        return self
