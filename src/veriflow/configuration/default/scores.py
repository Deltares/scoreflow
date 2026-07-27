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


class ReduceDimsForecast(BaseModel):
    """The dimensions over which a forecast can be reduced."""

    reduce_dims: Annotated[
        list[
            Literal[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
            ]
        ],
        Field(default_factory=list),
    ]

    @property
    def preserve_dims(self) -> list[StandardDim]:
        """The dimensions to preserve."""
        return [
            k
            for k in [
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
            ]
            if k not in self.reduce_dims
        ]


class ReduceDimsHistoricalOrForecast(BaseModel):
    """The dimensions over which a historical data can be reduced."""

    reduce_dims: Annotated[
        list[
            Literal[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                StandardDim.time,
            ]
        ],
        Field(
            default_factory=list,
            description="The dimensions over which to reduce. Can be either forecast or historical "
            "dimensions, but not both. For historical verification, the reduce_dims can only "
            "contain 'station' and 'time'. For forecast verification, the reduce_dims can only "
            "contain 'station', 'forecast_reference_time' and 'lead_time'.",
        ),
    ]

    @property
    def preserve_dims(self) -> list[StandardDim]:
        """The dimensions to preserve."""
        if (
            StandardDim.forecast_reference_time in self.reduce_dims
            or StandardDim.lead_time in self.reduce_dims
        ):
            return [
                k
                for k in [
                    StandardDim.station,
                    StandardDim.time,
                    StandardDim.forecast_reference_time,
                    StandardDim.lead_time,
                ]
                if k not in self.reduce_dims
            ]
        if StandardDim.time in self.reduce_dims:
            return [
                k
                for k in [
                    StandardDim.station,
                    StandardDim.time,
                ]
                if k not in self.reduce_dims
            ]

        return [
            k
            for k in [
                StandardDim.station,
            ]
            if k not in self.reduce_dims
        ]

    @model_validator(mode="after")
    def validate_reduce_dims(
        self,
    ) -> "ReduceDimsHistoricalOrForecast":
        """Validate that reduce_dims only contains either forecast or historical dimensions."""
        if (
            StandardDim.forecast_reference_time in self.reduce_dims
            or StandardDim.lead_time in self.reduce_dims
        ) and StandardDim.time in self.reduce_dims:
            msg = (
                "reduce_dims cannot contain both forecast and historical dimensions. "  # noqa: ISC004
                "Please choose either 'time' for historical verification or "
                "'forecast_reference_time' and 'lead_time' for forecast verification.",
            )
            raise ValueError(msg)
        return self


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def get_external_to_internal_mapping(self, data_source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        return {v[data_source]: k for k, v in self.root.items()}


class RankHistogramConfig(BaseScoreConfig, ReduceDimsForecast):
    """A rank histogram config element."""

    score_adapter: Literal[ScoreKind.rank_histogram]


class CrpsForEnsembleConfig(BaseScoreConfig, ReduceDimsForecast):
    """Configuration for CRPS for ensemble.

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


class CrpsCDFConfig(BaseScoreConfig, ReduceDimsForecast):
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


class ContinuousScoresConfig(BaseScoreConfig, ReduceDimsHistoricalOrForecast):
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


class CategoricalScoresConfig(BaseCategoricalScoreConfig, ReduceDimsHistoricalOrForecast):
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
