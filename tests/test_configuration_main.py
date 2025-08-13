"""Test the main module of the dpyverification.configuration package."""

from pathlib import Path

import yaml
from dpyverification.configuration import Config


def test_schema_jsonable(tmp_path: Path) -> None:
    """Check that the schema for our config is jsonable.

    This so we can be sure it will generate correctly for the documentation of our configuration.
    """
    tmpfile = tmp_path / "config.json"
    assert not tmpfile.exists()

    with tmpfile.open(mode="w") as myfile:
        yaml.dump(Config.model_json_schema(), myfile)  # type: ignore[misc] # model_json_schema output has Any

    assert tmpfile.exists()

    # TODO(AU): Additional tests on the configuration schema # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/37
    #   When adding documentation, can add the json schema in the doc. Then, also compare the
    #   version in the documentation with the current version as per this test.


# TODO(AU): Additional tests on the configuration schema # noqa: FIX002
#   https://github.com/Deltares-research/DPyVerification/issues/37
#   Do we want to test that all schema fields (recursive, and even private ones?) have a
#   description?
