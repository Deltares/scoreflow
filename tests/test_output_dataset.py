"""Test the veriflow.datamodel package."""

import pytest
import xarray as xr

from veriflow.configuration.utils import VerificationPair
from veriflow.datamodel.input import InputDataset
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


def test_add_score_dataset_result(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
) -> None:
    """Test adding a score whose result is already a Dataset (not a DataArray)."""
    dt = output_datatree_without_scores
    result = xr.Dataset({"fake_score": ("station", [1, 2, 3])})

    dt.veriflow.add_score(fake_verification_pair, result, name="fake_score")

    path_in_tree = f"{fake_verification_pair.id}/output/fake_score"
    assert result.identical(dt[path_in_tree].dataset)


def test_input_and_output(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    xarray_fake_score_result: xr.DataArray,
) -> None:
    """Test the input() and output() accessor methods."""
    dt = output_datatree_without_scores
    dt.veriflow.add_score(
        fake_verification_pair,
        xarray_fake_score_result,
        name=str(xarray_fake_score_result.name),
    )

    input_dataset = dt.veriflow.input(fake_verification_pair.id)
    assert isinstance(input_dataset, xr.Dataset)
    assert fake_verification_pair.variable in input_dataset.data_vars

    output_dataset = dt.veriflow.output(fake_verification_pair.id)
    assert isinstance(output_dataset, xr.Dataset)
    assert "fake_score" in dt.veriflow.list_scores(fake_verification_pair.id)


def test_list_scores(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    xarray_fake_score_result: xr.DataArray,
) -> None:
    """Test listing the available scores for a verification pair."""
    dt = output_datatree_without_scores
    dt.veriflow.add_score(
        fake_verification_pair,
        xarray_fake_score_result,
        name="fake_score",
    )

    assert dt.veriflow.list_scores(fake_verification_pair.id) == ["fake_score"]


def test_path_exists_in_dt(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
) -> None:
    """Test path_exists_in_dt for both existing and non-existing paths."""
    dt = output_datatree_without_scores
    assert dt.veriflow.path_exists_in_dt(f"{fake_verification_pair.id}/input") is True
    assert dt.veriflow.path_exists_in_dt("does_not_exist") is False


def test_validate_path_does_not_exist_raises(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    xarray_input_dataset: InputDataset,
) -> None:
    """Test that adding input data twice for the same pair raises ValueError."""
    dt = output_datatree_without_scores
    obs, sim = xarray_input_dataset.get_pair(fake_verification_pair)

    with pytest.raises(ValueError, match="already exists"):
        dt.veriflow.add_input_data(
            verification_pair=fake_verification_pair,
            obs=obs,
            sim=sim,
        )
