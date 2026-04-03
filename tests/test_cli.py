"""Tests for the CLI entry point."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from dpyverification import cli


def _write_example_config(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "example" / "config" / "config_test.yaml"
    destination = tmp_path / "config.yaml"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parse_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def test_cli_writes_output_with_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_example_config(tmp_path)
    output_path = tmp_path / "updated.yaml"

    def _fake_execute(_config: object) -> None:
        return None

    monkeypatch.setattr(cli, "execute_pipeline", _fake_execute)

    exit_code = cli.main(
        [
            str(config_path),
            "--output",
            str(output_path),
            "--start",
            "2026-01-03T00:00:00Z",
            "--end",
            "2026-01-04T00:00:00Z",
        ],
    )

    assert exit_code == 0
    assert output_path.exists()

    data = _read_yaml(output_path)
    verification = data["general"]["verification_period"]
    assert _parse_iso(verification["start"]) == datetime(2026, 1, 3)
    assert _parse_iso(verification["end"]) == datetime(2026, 1, 4)
    assert "general" not in data["datasources"][0]


def test_cli_writes_output_without_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_example_config(tmp_path)
    output_path = tmp_path / "updated.yaml"

    def _fake_execute(_config: object) -> None:
        return None

    monkeypatch.setattr(cli, "execute_pipeline", _fake_execute)

    exit_code = cli.main(
        [
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    output_data = _read_yaml(output_path)
    input_data = _read_yaml(config_path)

    output_verification = output_data["general"]["verification_period"]
    input_verification = input_data["general"]["verification_period"]
    assert _parse_iso(output_verification["start"]) == _parse_iso(input_verification["start"])
    assert _parse_iso(output_verification["end"]) == _parse_iso(input_verification["end"])


def test_cli_invalid_datetime_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_example_config(tmp_path)
    output_path = tmp_path / "updated.yaml"

    def _fake_execute(_config: object) -> None:
        return None

    monkeypatch.setattr(cli, "execute_pipeline", _fake_execute)

    exit_code = cli.main(
        [
            str(config_path),
            "--output",
            str(output_path),
            "--start",
            "not-a-datetime",
        ],
    )

    assert exit_code == 1
    assert not output_path.exists()
