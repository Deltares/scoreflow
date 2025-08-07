"""Module with the dpyverification internal DataModel."""

from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

import numpy as np
import xarray as xr
from pydantic import BaseModel, ValidationError

from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.utils import TimePeriod
from dpyverification.datasources.inputschemas import (
    XarrayDatasetObservations,
    XarrayDatasetSimulationsByForecastPeriod,
    XarrayDatasetSimulationsByForecastReferenceTime,
)


class DatasetKind(Enum):
    """Enumeration of the supported types of input data."""

    SIM_BY_FORECAST_REFERENCE_TIME = "SIM_BY_FORECAST_REFERENCE_TIME"
    SIM_BY_FORECAST_PERIOD = "SIM_BY_FORECAST_PERIOD"
    OBSERVATION = "OBSERVATION"


def transform_dataset(
    dataset: xr.Dataset,
    kind: DatasetKind,
    general_config: GeneralInfoConfig,
) -> xr.Dataset:
    """Transform a datasource to be compatible with the internal DataModel."""

    def clip_time_to_verification_period(
        dataset: xr.Dataset,
        verification_period: TimePeriod,
    ) -> xr.Dataset:
        """Clip the dataset on time dimension to verification period."""
        return dataset.sel(time=slice(verification_period.start, verification_period.end))  # type:ignore[misc]

    # Make a copy of the original dataset
    dataset = dataset.copy()

    # Clip any input to be within the verification period
    dataset = clip_time_to_verification_period(
        dataset=dataset,
        verification_period=general_config.verificationperiod,
    )

    # Return observations, no further tranformation needed
    if kind == DatasetKind.OBSERVATION:
        return dataset

    # Return simulations with forecast period dimension
    #   after selecting only configured forecast periods
    if kind == DatasetKind.SIM_BY_FORECAST_PERIOD:
        try:
            # Config is dtype timedelta64, dataset may be timedelta64][ns]
            #   xarray will handle these dtype differenes automatically
            return dataset.sel(forecast_period=general_config.leadtimes.timedelta64)
        except KeyError as e:
            msg = "Not all configured lead times could be found in dataset."
            raise KeyError(msg) from e

    # Return simulations with forecast reference time dimension
    #   after transforming it to a forecast-period based dataset.
    def transform_forecast_reference_time_sim_to_forecast_period_sim(ds: xr.Dataset) -> xr.Dataset:
        """Transform an input simulation with forecast_reference_time.

        Transform an input simulation with forecast_reference_time dims/coords to
        an input simulation dataset with forecast_period based dims/coords. This
        allows easy forecast_period-based slicing in the main simobs datamodel, where
        simulations are paired with observations along the time dimension.
        """
        # Stack time and forecast_reference_time into one dimension ft_pair,
        #   representing each unique pair of time and forecast_reference_time
        #   along a 1d dimension.
        ds_stacked = ds.stack(ft_pair=("forecast_reference_time", "time"))  # noqa: PD013

        # For each unique pair of time and forecast_reference_time
        #   compute the forecast_period (time - forecast_reference_time)
        forecast_period = (
            ds_stacked["time"].to_numpy() - ds_stacked["forecast_reference_time"].to_numpy()  # type:ignore[misc]
        )

        # Assign the result as a coordinate on dimension ft_pair
        ds_stacked = ds_stacked.assign_coords(forecast_period=("ft_pair", forecast_period))  # type:ignore[misc]

        # Swap the dimension so that time and forecast_reference_time now lie
        #   on dimension forecast_pepriod
        ds_stacked = ds_stacked.swap_dims({"ft_pair": "forecast_period"})  # type:ignore[misc]

        # Set a new multi-index based (forecast_period, time)
        #   to re-introduce the original time dimension
        ds_stacked = ds_stacked.set_index(forecast_index=["forecast_period", "time"])

        # Unstack to get back to time(time)
        ds_out = ds_stacked.unstack("forecast_index")  # noqa: PD010

        # Drop forecast_reference_time and ft_pair
        ds_out = ds_out.drop_vars(["ft_pair", "forecast_reference_time"])

        # Drop forecast_period < 0
        ds_out = ds_out.where(ds_out["forecast_period"] >= np.timedelta64(0, "h"), drop=True)

        # Reorder dimensions and return
        return ds_out.transpose("time", "forecast_period", "stations", "realization")

    return transform_forecast_reference_time_sim_to_forecast_period_sim(dataset)


def validate_dataset(dataset: xr.Dataset) -> tuple[xr.Dataset, DatasetKind]:
    """Validate a datasource by validating the xr.Dataset to a Pydantic schema."""
    schemas = {
        XarrayDatasetObservations: DatasetKind.OBSERVATION,
        XarrayDatasetSimulationsByForecastPeriod: DatasetKind.SIM_BY_FORECAST_PERIOD,
        XarrayDatasetSimulationsByForecastReferenceTime: DatasetKind.SIM_BY_FORECAST_REFERENCE_TIME,
    }

    def attempt_validation(schema: type[BaseModel], dataset: xr.Dataset) -> bool:
        """Validate and return True when succesfull, else False."""
        try:
            schema.model_validate(dataset.to_dict(data=False))  # type:ignore[misc]
            return True  # noqa: TRY300
        except ValidationError:
            return False

    for schema, kind in schemas.items():
        if attempt_validation(schema=schema, dataset=dataset):
            return dataset, kind
    msg = f"Invalid dataset {dataset}."
    raise ValidationError(msg)


class OutputDataset:
    """The internal output dataset.

    Contains input data, results from verificaition scores and metadata.
    """

    def __init__(self, input_dataset: xr.Dataset) -> None:
        self.input_dataset: xr.Dataset = input_dataset
        self.scores: dict[str, xr.Dataset]

        # Metadata
        self.current_time = datetime.now(tz=timezone.utc).strftime("%d/%m/%Y, %H:%M:%S")

    def add_score(self, kind: str, score: xr.Dataset | xr.DataArray) -> None:
        """Add a score to the scores list."""
        if kind in self.scores:
            msg = f"Cannot add score to OutputDataset. Score ({score}) is already present."
            raise ValueError(msg)
        self.scores[kind] = score

    def _get_score(self, kind: str) -> xr.Dataset:
        try:
            return self.scores[kind]
        except KeyError as e:
            msg = (
                f"Score kind ({kind}) not added to OutputDataset.",
                f"Available scores: ({self.scores.keys()})",
            )
            raise KeyError(msg) from e

    def get_output_dataset(
        self,
        scores: list[str] | Literal["all"] = "all",
        *,
        include_simobs: bool = True,
    ) -> xr.Dataset:
        """Get the output dataset."""
        scores_selection = (
            list(self.scores.values())
            if scores == "all"
            else [self._get_score(kind) for kind in scores]
        )

        if include_simobs:
            scores_selection.append(self.input_dataset)

        return xr.merge(scores_selection)


class SimObsDataset:
    """
    The main verification dataset, containing merged observations and simulations.

    The SimObsDataset takes as input a sequence of data and general configuration.
    Based on the configured verification period and forecast periods, initializing
    the SimObsDataset will validate the structure of each input dataset using a Pydantic schema,
    transform datasets based on their derrived type and merge all input data into one dataset.

    The allowed input dataset are currently:
    - Xarray Observations
      (:class:`dpyverification.datasources.inputschemas.XarrayDatasetObservations`)
    - Xarray Simulations based on forecast reference time
      (:class:`dpyverification.datasources.inputschemas.XarrayDatasetSimulationsByForecastReferenceTime`)
    - Xarray Simulations based on forecast period
      (:class:`dpyverification.datasources.inputschemas.XarrayDatasetSimulationsByForecastPeriod`)

    The SimObsDataset contains pairs of simulations and observations along dimensions
    time, stations, forecast_period and, in case of ensemble forecasts, realization.
    In this way, the dataset allows easy slicing based on forecast periods which makes it suited
    for calculating of a wide range of verification metrics.
    """

    def __init__(
        self,
        data: Sequence[xr.Dataset],
        general_config: GeneralInfoConfig,
    ) -> None:
        # Validate input data
        validated_datasets = (validate_dataset(dataset) for dataset in data)

        # Transform datasets based on their type
        transformed_datasets = (
            transform_dataset(dataset, simulation_kind, general_config)
            for dataset, simulation_kind in validated_datasets
        )

        # Merge input data and assign to self
        self.dataset = xr.merge(transformed_datasets)
