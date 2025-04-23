"""The definition of the configuration settings.

This definition is used both as the schema for the configuration yaml file, and as the content of
the dpyverification configuration object.

To generate a yaml / json file with the json representation of this schema:
    import pathlib
    import yaml
    from dpyverification.configuration import Config
    FILEPATH = pathlib.Path("YOUR_PATH_HERE")
    with FILEPATH.open("w") as myfile:
        yaml.dump(Config.model_json_schema(), myfile)

Sidenote: It is also possible to go the other way around and generate a pydantic schema from a
yaml/json file, see datamodel_code_generator, for example from
https://docs.pydantic.dev/latest/integrations/datamodel_code_generator/ . Note that this can
generate a pydantic model that is not up-to-date with the latest pydantic / python, and might
need some modifications.
"""

# TODO(AU): Add pydantic Field with description, and maybe title, to all attributes. # noqa: FIX002
#   https://github.com/Deltares-research/DPyVerification/issues/9
#   Add pydantic Field with description, and maybe title, to approximately every attribute. To both
#   have a descriptive json schema when the json schema is generated from the pydantic objects, and
#   to document what the fields are for in the code. Maybe only for Literal attributes, the
#   description can be skipped. Do also add the description to private attributes, to document
#   their use.

# ruff: noqa: D101 Do not require class docstrings for the classes in this file

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .utils import LeadTimes, SimObsVariables, TimePeriod


class BaseConfig(BaseModel):
    """A base config.

    Each element in the pipeline (datasource, score, datasink)
    inherits from this BaseConfig, so that each config has a 'kind' attribute.

    Based on a user-input to the configuration field 'kind', the pipeline
    will find the correct user-provided class for either a Datasource, score
    or a Datasink.
    """

    kind: str

    # Accept additional fields.
    # This is a requirement to make sure that all fields are
    # available after initializing the config instance.
    model_config = ConfigDict(extra="allow")


class BaseDatasourceConfig(BaseConfig):
    """
    Base config for a datasource config.

    Specific config definitions should inherit from
    this base class.
    """

    simobstype: str
    verificationperiod: TimePeriod | None = Field(
        None,
        description="Value from General verificationperiod used if not set.",
    )
    leadtimes: Annotated[
        LeadTimes | None,
        Field(
            description="Value from General leadtimes used if not set.",
        ),
    ] = None


class BaseDatasinkConfig(BaseConfig):
    """
    Base config for a datasink config.

    Specific config definitions should inherit from
    this base class.
    """


class BaseScoreConfig(BaseConfig):
    """
    Base config for a score config.

    Specific config definitions should inherit from
    this base class.
    """

    leadtimes: Annotated[
        LeadTimes | None,
        Field(
            description="Value from General leadtimes used if not set.",
        ),
    ] = None
    variablepair: Annotated[
        SimObsVariables,
        Field(
            description="Variable pair to use for the computation.",
        ),
    ]


class GeneralInfoConfig(BaseModel):
    verificationperiod: TimePeriod
    leadtimes: LeadTimes


class Config(BaseModel):
    fileversion: str
    general: GeneralInfoConfig
    datasources: Annotated[list[BaseDatasourceConfig], Field(min_length=1)]
    scores: Annotated[list[BaseScoreConfig], Field(min_length=1)]
    datasinks: Annotated[list[BaseDatasinkConfig], Field(min_length=1)]
