"""Integration tests for the cache against a real S3-compatible (MinIO) bucket.

These tests perform real network I/O against the endpoint configured via
environment variables and are intended to run only on a developer machine
that has a ``.env`` file in the repo root.

S3 credentials are loaded automatically by
:class:`veriflow.configuration.utils.S3AuthConfig` (a ``pydantic-settings``
``BaseSettings`` with ``env_prefix="S3_"``). Set any of:

- ``S3_ENDPOINT_URL`` -- MinIO endpoint, e.g. ``http://localhost:9000``.
- ``S3_ACCESS_KEY_ID`` / ``S3_SECRET_ACCESS_KEY``
- ``S3_REGION_NAME`` (optional)
- ``S3_ANON`` (optional)

Test-specific:

- ``VERIFLOW_TEST_S3_PATH`` -- ``s3://bucket/prefix`` the test may freely
  write to (e.g. ``s3://veriflow-test/cache``). A unique sub-path is appended
  per test to avoid collisions. If unset, the entire module is skipped.
"""

# mypy: ignore-errors

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import s3fs
import xarray as xr
from dotenv import load_dotenv

from veriflow.cache.cache import ZarrCache
from veriflow.cache.config import ReadWriteMode, ZarrCacheConfig
from veriflow.configuration.utils import S3AuthConfig

# Load .env from the repo root (tests/integration/test_remote_cache.py -> ../../).

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

RUN_MINIO_TESTS = os.environ.get("RUN_MINIO_TESTS")

pytestmark = pytest.mark.skipif(
    not RUN_MINIO_TESTS,
    reason=f"Skipping MinIO integration tests; ${RUN_MINIO_TESTS} not set",
)


def _build_unique_remote_path() -> str:
    """Return a unique ``s3://`` path under the integration test prefix."""
    return f"s3://deltares-verification-assets/veriflow/integration_tests/{uuid.uuid4().hex}.zarr"


def _make_remote_config(path: str, read_write_mode: ReadWriteMode) -> ZarrCacheConfig:
    """Build a ZarrCacheConfig pointed at ``path`` on the MinIO test bucket."""
    return ZarrCacheConfig(
        path=path,
        read_write_mode=read_write_mode,
        # S3AuthConfig() auto-loads S3_* env vars via pydantic-settings.
        auth_config=S3AuthConfig(),
        consolidated=True,
    )


@pytest.fixture
def remote_zarr_path() -> Iterator[str]:
    """Yield a unique ``s3://`` zarr path and recursively delete it after the test."""
    path = _build_unique_remote_path()
    yield path
    fs = s3fs.S3FileSystem(**S3AuthConfig().to_storage_options())
    # Strip the ``s3://`` scheme; s3fs.rm expects ``bucket/key``.
    fs_path = path.removeprefix("s3://")
    try:
        if fs.exists(fs_path):
            fs.rm(fs_path, recursive=True)
    except FileNotFoundError:
        # Test never wrote anything; nothing to clean.
        pass


def _ds_one_var(name: str = "v1") -> xr.Dataset:
    """Build a small xarray dataset for round-trip tests."""
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    stations = ["s1", "s2"]
    arr = np.arange(4, dtype="float32").reshape(2, 2)
    return xr.Dataset(
        {name: (("time", "station"), arr)},
        coords={"time": times, "station": stations},
    )


def test_remote_minio_is_remote(remote_zarr_path: str) -> None:
    """Verify a MinIO ``s3://`` path is reported as remote with auth wired through."""
    cfg = _make_remote_config(remote_zarr_path, ReadWriteMode.read)
    cache = ZarrCache(cfg)
    assert cache.is_remote
    assert cache.storage_options is not None


def test_remote_minio_get_dataset_missing_returns_empty(remote_zarr_path: str) -> None:
    """Verify ``get_dataset`` against a fresh remote prefix returns an empty dataset."""
    cfg = _make_remote_config(remote_zarr_path, ReadWriteMode.read)
    cache = ZarrCache(cfg)
    result = cache.get_dataset(source="does_not_exist")
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 0


def test_remote_minio_append_then_get_round_trip(remote_zarr_path: str) -> None:
    """Verify ``append`` followed by ``get_dataset`` round-trips through MinIO."""
    cfg = _make_remote_config(remote_zarr_path, ReadWriteMode.write)
    cache = ZarrCache(cfg)
    ds = _ds_one_var("v1")

    cache.append(ds, source="src1", dim="variable")
    result = cache.get_dataset(source="src1")

    assert "v1" in result.data_vars
    assert result["v1"].shape == ds["v1"].shape
    np.testing.assert_array_equal(result["v1"].to_numpy(), ds["v1"].to_numpy())
