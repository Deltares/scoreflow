"""A module for default implementation of scores."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel, model_validator

from veriflow.configuration.base import (
    BaseCategoricalScoreConfig,
    BaseEvent,
    BaseScoreConfig,
)
from veriflow.constants import (
    ScoreKind,
    StandardDim,
    SupportedCategoricalScores,
    SupportedContinuousScore,
)


class EventOperator(Enum):
    """Event operators."""

    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL_TO = "greater_than_or_equal_to"
    LESS_THAN_OR_EQUAL_TO = "less_than_or_equal_to"


class ReduceDims(BaseModel):
    """Unified dimensions configuration for all data types and scenarios.

    Supports:
    - Point forecast data: station, forecast_reference_time, lead_time
    - Gridded forecast data: forecast_reference_time, lead_time, x, y
    - Historical data: station, time
    - Mixed historical + forecast data: station, forecast_reference_time, lead_time, time

    Validation ensures x, y and station can never be configured together.
    The actual dimensions filtered to data are computed at runtime via
    compute_reduce_and_preserve_dims().
    """

    reduce_dims: Annotated[
        list[
            Literal[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                StandardDim.time,
                StandardDim.x,
                StandardDim.y,
            ]
        ],
        Field(
            default_factory=list,
            description="Dimensions to reduce over. Can include any combination of: "
            "station, forecast_reference_time, lead_time, time, x, y. "
            "Only dimensions present in the data will be used. "
            "The x, y (spatial) and station dimensions cannot be used together.",
        ),
    ]

    @model_validator(mode="after")
    def validate_spatial_dims(self) -> "ReduceDims":
        """Validate that x, y and station are never configured together."""
        has_station = StandardDim.station in self.reduce_dims
        has_spatial = StandardDim.x in self.reduce_dims or StandardDim.y in self.reduce_dims

        if has_station and has_spatial:
            msg = (
                "Cannot configure both spatial dimensions (x, y) and station together in "
                "reduce_dims. Use station for point data or x, y for gridded data, but not both."
            )
            raise ValueError(msg)

        # If x is present, y must also be present (and vice versa)
        has_x = StandardDim.x in self.reduce_dims
        has_y = StandardDim.y in self.reduce_dims
        if has_x != has_y:
            msg = (
                "Both x and y dimensions must be configured together. "
                "Cannot reduce over x without y or y without x."
            )
            raise ValueError(msg)

        return self


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def get_external_to_internal_mapping(self, data_source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        return {v[data_source]: k for k, v in self.root.items()}


class RankHistogramConfig(BaseScoreConfig, ReduceDims):
    """A rank histogram config element supporting both point and gridded ensemble data."""

    score_adapter: Literal[ScoreKind.rank_histogram]


class CrpsForEnsembleConfig(BaseScoreConfig, ReduceDims):
    """Configuration for CRPS for ensemble supporting both point and gridded data.

    For reference, see: See: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_for_ensemble
    """

    score_adapter: Literal[ScoreKind.crps_for_ensemble]
    method: Annotated[
        Literal["ecdf", "fair"],
        Field(
            description="Method to compute the cumulative distribution function from an ensemble.",
            default="ecdf",
        ),
    ]


class CrpsCDFConfig(BaseScoreConfig, ReduceDims):
    """Configuration for CRPS for CDF.

    For reference, see: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_cdf
    """

    score_adapter: Literal[ScoreKind.crps_cdf]
    integration_method: Annotated[
        Literal["exact", "trapz"],
        Field(
            description="The method of integration. 'exact' computes the exact integral, "
            "'trapz' uses a trapezoidal rule and is an approximation of the CRPS.",
        ),
    ] = "exact"


class ContinuousScoresConfig(BaseScoreConfig, ReduceDims):
    """Configure multiple continuous scores."""

    score_adapter: Literal[ScoreKind.continuous_scores]
    scores: list[SupportedContinuousScore]

    @model_validator(mode="after")
    def validate_nse(self) -> "ContinuousScoresConfig":
        """Validate that if nse in scores, reduce_dims is not empty."""
        if SupportedContinuousScore.nse in self.scores and len(self.reduce_dims) == 0:
            msg = (
                "NSE: need at least one dimension to be reduced. "
                "Please add at least one dimension to reduce_dims."
            )
            raise ValueError(msg)
        return self


class SALScoreConfig(BaseScoreConfig):
    """Configuration for the SAL (Structure-Amplitude-Location) spatial score.

    Applies to single deterministic gridded forecasts. For reference, see:
    https://pysteps.readthedocs.io/en/stable/generated/pysteps.verification.salscores.sal.html
    """

    score_adapter: Literal[ScoreKind.sal]
    thr_factor: Annotated[
        float,
        Field(
            description="Factor by which the threshold quantile is multiplied to obtain the "
            "threshold used for object identification. If None in pysteps, no threshold is "
            "applied; here a sensible default is used.",
        ),
    ] = 0.067
    thr_quantile: Annotated[
        float,
        Field(
            description="Quantile (in [0, 1]) used together with thr_factor to define the "
            "threshold for identifying precipitation objects.",
            ge=0.0,
            le=1.0,
        ),
    ] = 0.95


class ThresholdEvent(BaseEvent):
    """An event definition for a threshold."""

    threshold: Annotated[
        str,
        Field(description="Threshold id to use in event definition."),
    ]
    operator: Annotated[
        EventOperator,
        Field(description="The operator to use for creating the events."),
    ]


class CategoricalScoresConfig(BaseCategoricalScoreConfig, ReduceDims):
    """Config to compute categorical scores, based on an event definition."""

    score_adapter: Literal[ScoreKind.categorical_scores]
    scores: Annotated[
        list[SupportedCategoricalScores],
        Field(
            description="For reference, see: https://scores.readthedocs.io/en/stable/api.html#module-scores.categorical.",
        ),
    ]
    events: Annotated[
        list[ThresholdEvent],
        Field(
            description="A list of threshold event definitions. For each event, a categorical "
            "score will be computed. A threshold event is defined by a threshold and an operator. "
            "The threshold is a string that corresponds to a threshold id defined in the "
            "configuration. The operator defines how the threshold is applied to the data to "
            "create the event. For example, if the threshold is '10' and the operator is "
            "'greater_than', the event will be created by applying the operator to the data "
            "and the threshold, i.e. data > 10. ",
        ),
    ]
    return_contingency_table: Annotated[
        bool,
        Field(description="Whether to return the contingency table in the output."),
    ] = True
