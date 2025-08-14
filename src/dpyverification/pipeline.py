"""
Specification of a pipeline that will collect data and run verification functions on the data.

Specification is expected to be in a configuration file.
Results can at least be written to netcdf file.
"""

from pathlib import Path
from typing import TypeVar

import xarray as xr

from dpyverification.configuration import Config, ConfigFile
from dpyverification.configuration.file import ConfigType
from dpyverification.datamodel import OutputDataset, SimObsDataset
from dpyverification.datasinks.base import BaseDatasink
from dpyverification.datasinks.cf_compliant_netdf import CFCompliantNetCDF
from dpyverification.datasources.base import BaseDatasource
from dpyverification.datasources.fewsnetcdf.main import FewsNetcdfFile
from dpyverification.scores.base import BaseScore
from dpyverification.scores.crps_for_ensemble import CrpsForEnsemble
from dpyverification.scores.rank_histogram import RankHistogram

TItem = TypeVar("TItem", bound=BaseDatasource | BaseDatasink | BaseScore)

DEFAULT_DATASOURCES: list[type[BaseDatasource]] = [FewsNetcdfFile]
DEFAULT_SCORES: list[type[BaseScore]] = [RankHistogram, CrpsForEnsemble]
DEFAULT_DATASINKS: list[type[BaseDatasink]] = [CFCompliantNetCDF]


def find_matching_kind_in_list(
    items: list[type[TItem]],
    kind: str,
) -> type[TItem]:
    """Return a datasource, calcuation or datasink of a given kind."""
    for item in items:
        if kind == item.kind:
            return item
    msg = f"No item with type {kind} exists."
    raise ValueError(msg)


def execute_pipeline(
    config: tuple[Path, ConfigType] | Config,
    user_datasources: list[type[BaseDatasource]] | None = None,
    user_scores: list[type[BaseScore]] | None = None,
    user_datasinks: list[type[BaseDatasink]] | None = None,
) -> xr.Dataset:
    """Execute a pipeline as defined in the configfile."""
    # TODO(AU): Implement parsing of a runinfo xml file into a valid config dict # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/8
    #   As part of that, add a unit test on what happens if a wrong conftype is passed, and make
    #   sure it gives a nice error message

    available_datasources = (
        user_datasources + DEFAULT_DATASOURCES
        if user_datasources is not None
        else DEFAULT_DATASOURCES
    )
    available_scores = user_scores + DEFAULT_SCORES if user_scores is not None else DEFAULT_SCORES
    available_datasinks = (
        user_datasinks + DEFAULT_DATASINKS if user_datasinks is not None else DEFAULT_DATASINKS
    )
    # Initialize the config instance from file when it's not directly provided
    if not isinstance(config, Config):
        config = ConfigFile(configfile=config[0], configtype=config[1]).content

    # Collect and initialize all datasources
    datasources: list[BaseDatasource] = []
    for datasource_config in config.datasources:
        source_kind = find_matching_kind_in_list(
            items=available_datasources,
            kind=datasource_config.kind,
        )
        datasource = source_kind.from_config(
            datasource_config.model_dump(),  # type: ignore[misc] # Allow Any
        )
        datasources.append(datasource)

    # Get data for each datasource
    for datasource in datasources:
        datasource.get_data()

    # Initialize the input dataset
    input_dataset = SimObsDataset(
        [datasource.dataset for datasource in datasources],
        config.general,
    )

    # Initialize output dataset
    output_dataset = OutputDataset(simobs_dataset=input_dataset.dataset)

    # Add score results to the datamodel
    for score_config in config.scores:
        score_kind = find_matching_kind_in_list(
            items=available_scores,
            kind=score_config.kind,
        )
        score = score_kind.from_config(score_config.model_dump())  # type: ignore[misc] # Allow Any
        result = score.compute(input_dataset)
        output_dataset.add_score(kind=score.kind, score=result)

    # Write data for each datasink
    for datasink_config in config.datasinks:
        sink_kind = find_matching_kind_in_list(items=available_datasinks, kind=datasink_config.kind)
        datasink = sink_kind.from_config(datasink_config.model_dump())  # type: ignore[misc] # Allow Any
        datasink.write_data(output_dataset.get_output_dataset(include_simobs=False))

    return output_dataset.get_output_dataset()
