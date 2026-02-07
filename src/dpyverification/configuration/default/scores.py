"""A module for default implementation of scores."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel

from dpyverification.configuration.config import BaseScoreConfig
from dpyverification.constants import ScoreKind, StandardDim, SupportedContinuousScore


class ReduceDimsForecast(BaseModel):
    """The dimensions over which a forecast can be reduced."""

    reduce_dims: Annotated[
        list[
            Literal[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.forecast_period,
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
                StandardDim.variable,
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.forecast_period,
            ]
            if k not in self.reduce_dims
        ]


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def get_external_to_internal_mapping(self, data_source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        return {v[data_source]: k for k, v in self.root.items()}


class RankHistogramConfig(BaseScoreConfig, ReduceDimsForecast):
    """A rank histogram config element."""

    kind: Literal[ScoreKind.rank_histogram]


class CrpsForEnsembleConfig(BaseScoreConfig, ReduceDimsForecast):
    """Configuration for CRPS for ensemble.

    For reference, see: See: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_for_ensemble
    """

    kind: Literal[ScoreKind.crps_for_ensemble]
    method: Annotated[
        Literal["ecdf", "fair"],
        Field(
            description=(
                "Method to compute the cumulative distribution function from an ensemble."
            ),
            default="ecdf",
        ),
    ]


class CrpsCDFConfig(BaseScoreConfig, ReduceDimsForecast):
    """Configuration for CRPS for CDF.

    For reference, see: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_cdf
    """

    kind: Literal[ScoreKind.crps_cdf]
    integration_method: Annotated[
        Literal["exact", "trapz"],
        Field(
            description="The method of integration. 'exact' computes the exact integral, "
            "'trapz' uses a trapezoidal rule and is an approximation of the CRPS.",
        ),
    ] = "exact"


class ContinuousScoresConfig(BaseScoreConfig, ReduceDimsForecast):
    """Configure multiple continuous scores."""

    kind: Literal[ScoreKind.continuous_scores]
    scores: list[SupportedContinuousScore]
