"""Command-line interface for running the verification pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from dpyverification.configuration import Config, ConfigFile
from dpyverification.configuration.file import ConfigKind
from dpyverification.configuration.utils import VerificationPeriod
from dpyverification.pipeline import execute_pipeline

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DPyVerification pipeline from a YAML config, optionally overriding the "
            "verification period, and write the updated config to a new YAML file."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the updated YAML configuration.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Override verification_period.start (ISO datetime).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Override verification_period.end (ISO datetime).",
    )
    return parser


def _apply_verification_period_overrides(
    config: Config,
    start: str | None,
    end: str | None,
) -> Config:
    if start is None and end is None:
        return config

    current_period = config.general.verification_period
    new_period = VerificationPeriod(
        dimension=current_period.dimension,
        start=start if start is not None else current_period.start,
        end=end if end is not None else current_period.end,
    )
    new_general = config.general.model_copy(update={"verification_period": new_period})
    return config.model_copy(update={"general": new_general})


def _config_to_yaml_dict(config: Config) -> dict[str, Any]:
    data = config.model_dump(mode="json")

    for datasource in data.get("datasources", []):
        datasource.pop("general", None)
        datasource.pop("id_mapping", None)

    for score in data.get("scores", []):
        score.pop("general", None)

    for datasink in data.get("datasinks") or []:
        datasink.pop("general", None)

    if data.get("datasinks") is None:
        data.pop("datasinks", None)

    if data.get("id_mapping") is None:
        data.pop("id_mapping", None)

    return data


def _write_yaml(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = ConfigFile(
            config_file=args.config,
            config_type=ConfigKind.YAML,
        ).content
        config = _apply_verification_period_overrides(config, args.start, args.end)
        yaml_dict = _config_to_yaml_dict(config)
        _write_yaml(yaml_dict, args.output)
        _ = execute_pipeline(config)
    except Exception as exc:  # noqa: BLE001
        logger.error("CLI execution failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
