"""Command-line interface for running the verification pipeline."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from dpyverification.configuration import Config
from dpyverification.configuration.file import ConfigFile
from dpyverification.constants import NAME, VERSION
from dpyverification.pipeline import run_pipeline

logger = logging.getLogger(__name__)


app = typer.Typer(  # The main app for the command-line interface
    help=(
        f"Welcome to the {NAME} command line interface. The interface allows you to run a "
        "verification pipeline."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},  # type:ignore[misc]
    no_args_is_help=True,
)
run_pipeline_subcommand = typer.Typer()  # Add subcommand group


def _version_callback(*, value: bool) -> None:
    """Show the version and exit when --version/-V is provided."""
    if value:
        typer.echo(f"{NAME} version {VERSION}")
        raise typer.Exit


@app.callback()  # type:ignore[misc]
def main(
    *,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    version: Annotated[  # noqa: ARG001
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Set the main callback for the app.

    We get the root logger and attach a RichHandler to it. In this way, log messages from the
    project (i.e. the pipeline) will be logged properly to the terminal.
    """
    logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    if not logger.hasHandlers():
        logger.addHandler(RichHandler())


@app.command("run-pipeline")  # type:ignore[misc]
def run_pipeline_cmd(  # noqa: C901, PLR0912
    path_to_yaml_config: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            readable=True,
            resolve_path=True,
            dir_okay=False,
            help="Path to the YAML configuration file.",
        ),
    ],
    set_verification_period_start: Annotated[
        datetime | None,
        typer.Option(
            help="Override verification period start (ISO format: YYYY-MM-DDTHH:MM:SS)",
        ),
    ] = None,
    set_verification_period_end: Annotated[
        datetime | None,
        typer.Option(
            help="Override verification period end (ISO format: YYYY-MM-DDTHH:MM:SS)",
        ),
    ] = None,
) -> None:
    """Run the verification pipeline from the command line."""
    # Load the YAML content from the provided path
    config = ConfigFile(config_file=path_to_yaml_config, config_type="yaml").content

    logger.info("Applying command-line overrides to the config content.")

    modified_config = config.model_copy(deep=True)

    if set_verification_period_start is not None:
        logger.info(
            f"Overriding verification period start to {set_verification_period_start.isoformat()} "  # noqa: G004
            f"based on command-line input.",
        )
        modified_config.general.verification_period.start = set_verification_period_start

        # Because datasources, scores, and datasinks will have a copy of the general configuration,
        # we also need to override the verification period start for each of them
        if modified_config.datasources is not None:
            for datasource in modified_config.datasources:
                datasource.general.verification_period.start = set_verification_period_start
        if modified_config.scores is not None:
            for score in modified_config.scores:
                score.general.verification_period.start = set_verification_period_start
        if modified_config.datasinks is not None:
            for datasink in modified_config.datasinks:
                datasink.general.verification_period.start = set_verification_period_start

    if set_verification_period_end is not None:
        logger.info(
            f"Overriding verification period end to {set_verification_period_end.isoformat()} "  # noqa: G004
            f"based on command-line input.",
        )
        modified_config.general.verification_period.end = set_verification_period_end

        # Because datasources, scores, and datasinks will have a copy of the general configuration,
        # we also need to override the verification period end for each of them
        if modified_config.datasources is not None:
            for datasource in modified_config.datasources:
                datasource.general.verification_period.end = set_verification_period_end
        if modified_config.scores is not None:
            for score in modified_config.scores:
                score.general.verification_period.end = set_verification_period_end
        if modified_config.datasinks is not None:
            for datasink in modified_config.datasinks:
                datasink.general.verification_period.end = set_verification_period_end

    # Load the configuration
    #   at this point, Typer has already validate path is an existing, readable file
    logger.info("Validating modified configuration.")
    config = Config.model_validate(modified_config)
    logger.info("Configuration loaded successfully. Starting the pipeline.")

    # Run the pipeline
    run_pipeline(config=config)


app.add_typer(run_pipeline_subcommand)


if __name__ == "__main__":
    app()
