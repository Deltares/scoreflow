"""Test the scripts."""

from pathlib import Path

from scripts.export_versioned_schemas import export_versioned_schemas


def test_export_versioned_schemas(tmp_path: Path) -> None:
    """Test that the export_versioned_schemas script runs without error and creates a schema."""
    schema_path = tmp_path / "config.schema.json"

    # Run the script
    export_versioned_schemas(schema_path)

    # Check that the schema file was created
    assert schema_path.exists()
