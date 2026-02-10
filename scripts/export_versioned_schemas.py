"""Export config schemas for reference in config files."""

from pathlib import Path

from dpyverification.configuration import Config
from dpyverification.constants import VERSION

schema_dir = Path(__file__).parent.parent / "schemas" / f"{VERSION}"
schema_dir.mkdir()

schema_path = schema_dir / "config.schema.json"

if schema_path.exists():
    msg = f"Schema already exists. Please ensure no schema exists at: {schema_path}"
    raise ValueError(msg)
else:
    Config.write_schema(schema_path)
