"""Type aliases for the veriflow package."""

from typing import TypeAlias

from veriflow.constants import DataType, SpatialType

DataSpec: TypeAlias = tuple[DataType, SpatialType]
