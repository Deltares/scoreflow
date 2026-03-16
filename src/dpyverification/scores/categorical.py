"""Categorical scores, based on a 2x2 contingency table."""

from collections.abc import Callable
from enum import StrEnum
from typing import ClassVar

import xarray as xr
from scores.categorical import (  # type:ignore[import-untyped]
    BasicContingencyManager,
    BinaryContingencyManager,
)

from dpyverification.configuration.default.scores import CategoricalScoresConfig, EventOperator
from dpyverification.constants import DataType, SupportedCategoricalScores
from dpyverification.datamodel import InputDataset
from dpyverification.scores.base import BaseScore


def get_categorical_score(score_name: SupportedCategoricalScores) -> type:
    """Get a categorical score from the scores package."""
    return getattr(BasicContingencyManager, score_name.value)  # type:ignore[no-any-return, misc]


class CategoricalScoreDim(StrEnum):
    """Names of dimensions added when computing a categorical score."""

    EVENT_THRESHOLD = "event_threshold"
    EVENT_OPERATOR = "event_operator"


def create_binary_array(
    data: xr.DataArray,
    thresholds: xr.DataArray,
    operator: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
) -> xr.DataArray:
    """Given data and thresholds, compute the binary events."""
    # Align along dimension station
    data_aligned, thresholds_aligned = xr.align(data, thresholds, join="inner")
    result = operator(data_aligned, thresholds_aligned)
    if isinstance(result, xr.DataArray):  # type:ignore[misc]
        return result
    msg = "Failed to create a binary xr.DataArray based on data and thresholds."  # type:ignore[unreachable] # runtime check
    raise ValueError(msg)


def set_event_coordinates_on_result(
    data_array: xr.Dataset,
    threshold: str,
    operator: EventOperator,
) -> xr.Dataset:
    """Set coordinates on data array to represent the event for which a score was computed."""
    data_array = data_array.expand_dims(
        {CategoricalScoreDim.EVENT_THRESHOLD: 1, CategoricalScoreDim.EVENT_OPERATOR: 1},
    )
    return data_array.assign_coords(
        {  # type:ignore[misc]
            CategoricalScoreDim.EVENT_OPERATOR: [operator.name],  # type:ignore[misc]
            CategoricalScoreDim.EVENT_THRESHOLD: [threshold],  # type:ignore[misc]
        },
    )


class CategoricalScores(BaseScore):
    """Implementation for CRPS for probabilistic forecasts, expressed as cdf."""

    kind = "categorical_scores"
    config_class = CategoricalScoresConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.simulated_forecast_single,
    }

    def __init__(self, config: CategoricalScoresConfig) -> None:
        self.config: CategoricalScoresConfig = config

    def compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
        thresholds: xr.DataArray | None = None,  # optional, to satisfy contract with the ABC
    ) -> xr.Dataset | xr.DataArray:
        """Compute any number of categorical scores."""
        # Runtime check for thresholds, because they are always required
        if thresholds is None:
            msg = "Argument 'thresholds' is None, but required for computing categorical scores."
            raise ValueError(msg)

        results: list[xr.DataArray | xr.Dataset] = []
        obs_mapped = InputDataset.map_historical_into_forecast_space(obs, sim)
        results = []
        for event in self.config.events:
            obs_events = create_binary_array(
                obs_mapped,
                thresholds=thresholds,
                operator=event.operator.value,  # type:ignore[misc]
            )
            sim_events = create_binary_array(
                sim,
                thresholds=thresholds,
                operator=event.operator.value,  # type:ignore[misc]
            )
            binary_contingency_manager = BinaryContingencyManager(  # type:ignore[misc]
                fcst_events=sim_events,
                obs_events=obs_events,
            )
            basic_contingency_manager = binary_contingency_manager.transform(  # type:ignore[misc]
                preserve_dims=self.config.preserve_dims,
            )
            scores = []
            for score in self.config.scores:
                score_func = get_categorical_score(score)
                score_array = score_func(basic_contingency_manager)  # type:ignore[misc]
                score_array.name = str(score.value)  # type:ignore[misc]
                scores.append(score_array)  # type:ignore[misc]

            if self.config.return_contingency_table is True:
                table: xr.DataArray = basic_contingency_manager.get_table()  # type:ignore[misc]
                table.name = "contingency_table"
                scores.append(table)  # type:ignore[misc]

            merged_scores: xr.Dataset = xr.merge(scores)  # type:ignore[misc, assignment]
            merged_scores = set_event_coordinates_on_result(
                merged_scores,
                threshold=event.threshold,
                operator=event.operator,
            )
            results.append(merged_scores)

        return xr.combine_by_coords(results)
