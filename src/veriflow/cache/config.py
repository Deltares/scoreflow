"""Config for the cache."""

from enum import StrEnum
from os import R_OK, access
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator

from veriflow.configuration.utils import S3AuthConfig


class CacheBackEnd(StrEnum):
    """The veriflow cache backends."""

    zarr = "zarr"


class CacheType(StrEnum):
    """The veriflow cache types."""

    local = "local"
    remote = "remote"


class ReadWriteMode(StrEnum):
    """The veriflow cache read/write modes."""

    read = "r"
    read_write = "rw"


class ZarrCacheConfig(BaseModel):
    """Configuration for the veriflow cache.

    The cache will be materialized as a Zarr store on the local filesystem or remote object storage
    (e.g. S3). When configured, the cache will be used to store and retrieve datasets from any
    datasource. For example: when requesting forecast data from a datasource, and part of the data
    is already cached, the cache will be used to retrieve the cached data and only the missing data
    will be fetched from the datasource.
    """

    read_write_mode: ReadWriteMode = Field(
        ReadWriteMode.read,
        description="The read/write mode for the cache.",
    )

    path: Annotated[
        str,
        Field(
            min_length=1,
            description="Path to a single Zarr store. Local filesystem path (absolute or "
            "relative) or a remote URL such as 's3://bucket/key/store.zarr'.",
        ),
    ]
    auth_config: Annotated[
        S3AuthConfig | None,
        Field(
            default=None,
            description="Authentication configuration for remote stores. Only consulted "
            "when 'path' points to an 's3://' location. Credentials are loaded from "
            "S3_-prefixed environment variables; instantiate as 'auth_config: {}' in YAML "
            "to enable env-based loading.",
        ),
    ] = None
    storage_options: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description="Additional storage_options forwarded to xr.open_zarr. Merged on "
            "top of the options derived from 'auth_config'. Use this for advanced "
            "fsspec / s3fs settings not exposed by S3AuthConfig.",
        ),
    ] = None
    consolidated: Annotated[
        bool | None,
        Field(
            default=None,
            description="Whether to use consolidated metadata when opening the store. "
            "Forwarded to xr.open_zarr. Default ('None') lets xarray auto-detect.",
        ),
    ] = None

    def is_remote_path(self) -> bool:
        """Return True if ``path`` looks like a remote/fsspec URL (e.g. ``s3://``)."""
        return "://" in self.path

    @model_validator(mode="after")
    def validate_cache_path_accessible(self) -> Self:
        """Check that cache dir exists and is a directory."""
        if not self.is_remote_path():
            path = Path(self.path)
            if not path.exists():
                path.mkdir(parents=True)
            elif not path.is_dir() and access(path, R_OK):
                msg = "Cache directory is not an accessible directory."
                raise NotADirectoryError(msg)
        return self
