"""Module for creating and managing the output dataset for veriflow."""

from typing import TYPE_CHECKING, cast

import xarray as xr

from veriflow.configuration.utils import VerificationPair


@xr.register_datatree_accessor("veriflow")  # type: ignore[no-untyped-call, misc]
class VeriflowAccessor:
    """Accessor for managing the output dataset for veriflow.

    The lay-out of the DataTree is as follows:
    - Root node: "veriflow-output"
    - Child nodes: Each child node corresponds to a verification pair, identified by its unique ID.
    - Each verification pair node contains two sub-nodes:
        - "input": Contains the input dataset for the verification pair.
        - "output": Contains the output dataset with computed scores for the verification pair.
    - Each output node contains one or more sub-nodes, each corresponding to a specific score
        (e.g., "crps", "rank_histogram"), which contain the computed score results as xarray
        Datasets.

    Schematic representation of the DataTree structure:

    veriflow-output
    ├── verification_pair_1
    │   └── input
    │         ├── observed_source_id
    │         └── simulated_forecast_ensemble_source_id
    │   └── output
    │       ├── crps
    │           ├── crps
    │           ├── overforecast_penalty
    │           ├── underforecast_penalty
    │           └── forecast_spread_term
    │       └── rank_histogram
    │           └── rank_histogram
    └── verification_pair_2
        ├── input
        │   └── ...
        └── output
            └── ...
    """

    def __init__(self, dt: xr.DataTree) -> None:
        self._dt = dt

    @property
    def dt(self) -> xr.DataTree:
        """Get the underlying DataTree."""
        return self._dt

    @property
    def verification_pairs(self) -> list[str]:
        """Get a list of verification pair IDs in the DataTree."""
        return list(self.dt.children.keys())

    def input(self, verification_pair_id: str) -> xr.Dataset:
        """Get the input dataset for a specific verification pair."""
        pair_node = cast("xr.DataTree", self.dt[verification_pair_id])
        return cast("xr.DataTree", pair_node["input"]).dataset

    def output(self, verification_pair_id: str) -> xr.Dataset:
        """Get the output dataset for a specific verification pair."""
        pair_node = cast("xr.DataTree", self.dt[verification_pair_id])
        return cast("xr.DataTree", pair_node["output"]).dataset

    def list_scores(self, verification_pair_id: str) -> list[str]:
        """Get a list of available scores for a specific verification pair."""
        pair_node = cast("xr.DataTree", self.dt[verification_pair_id])
        output_node = cast("xr.DataTree", pair_node["output"])
        return list(output_node.children.keys())

    def path_exists_in_dt(self, path_in_dt: str) -> bool:
        """Validate if a given path does not exist in the DataTree."""
        try:
            self.dt[path_in_dt]
        except KeyError:
            return False
        else:
            return True

    def validate_path_does_not_exist(self, path_in_dt: str) -> None:
        """Validate the path does not exist in the DataTree. Raise ValueError if it does."""
        if self.path_exists_in_dt(path_in_dt):
            msg = f"Path '{path_in_dt}' already exists in the DataTree."
            raise ValueError(msg)

    def add_score(
        self,
        verification_pair: VerificationPair,
        result: xr.DataArray | xr.Dataset,
        name: str,
    ) -> None:
        """Add a score results to the datatree."""
        # We always convert the result to a Dataset for consistency, even if it's a DataArray.
        # This results in a clean tree structure that is easier to understand and navigate.
        if isinstance(result, xr.DataArray):  # type:ignore[misc]
            result = result.to_dataset()

        path_in_dt = f"{verification_pair.id}/output/{name}"
        self.validate_path_does_not_exist(path_in_dt)
        self.dt[path_in_dt] = result

    def add_input_data(
        self,
        verification_pair: VerificationPair,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> None:
        """Add input data to the datastore."""
        path_in_dt = f"{verification_pair.id}/input"
        self.validate_path_does_not_exist(path_in_dt)
        self.dt[path_in_dt] = xr.merge([obs, sim], compat="override", join="outer")  # type:ignore[misc, call-overload]


if TYPE_CHECKING:

    class VeriflowDataTree(xr.DataTree):  # type: ignore[no-untyped-call]
        """A DataTree with the runtime-registered `veriflow` accessor, for static typing.

        `xr.DataTree` instances are never actually instances of this class at runtime; use
        `cast("VeriflowDataTree", dt)` once after creating/receiving a DataTree to get typed
        access to `dt.veriflow` for the rest of that scope.
        """

        veriflow: VeriflowAccessor
else:
    VeriflowDataTree = xr.DataTree
