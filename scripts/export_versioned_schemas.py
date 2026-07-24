"""Export config schemas for reference in config files."""

from pathlib import Path

from veriflow.configuration import Config
from veriflow.constants import SCHEMA_VERSION


def export_versioned_schemas(schema_path: Path) -> None:
    """Export the config schema for the current version to the schemas directory."""
    Config.write_schema(schema_path)


if __name__ == "__main__":
    schema_dir = Path(__file__).parent.parent / "schemas" / f"{SCHEMA_VERSION}"
    schema_dir.mkdir(exist_ok=True)
    schema_path = schema_dir / "config.schema.json"

    export_versioned_schemas(schema_path)
