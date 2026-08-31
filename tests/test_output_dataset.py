"""Test the veriflow.datamodel package."""

import xarray as xr

from veriflow.configuration.utils import VerificationPair
from veriflow.datamodel.output import VeriflowDataTree

# mypy: disable-error-code="misc"


def test_output_datatree_accessor(output_datatree_without_scores: VeriflowDataTree) -> None:
    """Test the output_dataset initializes successfully with lead time (fp) input."""
    assert output_datatree_without_scores is not None
    assert isinstance(output_datatree_without_scores, xr.DataTree)
    assert hasattr(output_datatree_without_scores, "veriflow")
    accessor = output_datatree_without_scores.veriflow
    assert hasattr(accessor, "verification_pairs")
    assert isinstance(accessor.verification_pairs, list)
    assert hasattr(accessor, "add_input_data")
    assert hasattr(accessor, "add_score")


def test_add_score_to_output_dataset(
    output_datatree_without_scores: VeriflowDataTree,
    xarray_fake_score_result: xr.DataArray,
    fake_verification_pair: VerificationPair,
) -> None:
    """Test adding a score to the output dataset."""
    dt = output_datatree_without_scores

    # Add score
    dt.veriflow.add_score(
        fake_verification_pair,
        xarray_fake_score_result,
        name=str(xarray_fake_score_result.name),
    )
    assert isinstance(dt, xr.DataTree)

    # Verify that the score was correctly added to the DataTree at the expected path.
    path_in_tree = f"{fake_verification_pair.id}/output/{xarray_fake_score_result.name}"
    assert dt[path_in_tree] == xarray_fake_score_result
