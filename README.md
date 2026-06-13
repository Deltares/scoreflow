[![PyPI](https://img.shields.io/pypi/v/veriflow.svg?style=flat&color=6B8E23)](https://pypi.org/project/veriflow/)
[![Docs latest](https://img.shields.io/badge/docs-latest-8B4513.svg)](https://deltares.github.io/veriflow/index.html)
[![codecov](https://codecov.io/gh/Deltares/veriflow//branch/main/graph/badge.svg)](https://codecov.io/gh/Deltares/veriflow)
[![CI](https://img.shields.io/github/actions/workflow/status/Deltares/veriflow/ci.yml?color=white&label=CI
)](https://github.com/Deltares/veriflow/actions/workflows/ci.yml)
[![Docs build](https://img.shields.io/github/actions/workflow/status/Deltares/veriflow/docs.yml?color=white&label=docs%20build)](https://github.com/Deltares/veriflow/actions/workflows/docs.yml)


# Veriflow

A reproducible verification pipeline for evaluating model outputs and forecasts.
- 📥 Fetching data
- 🧮 Computing scores
- 📝 Writing results

<br>
<img src="_static/pipeline.svg" alt="Schematic overview of veriflow pipeline" width="80%">

## Key features
- ✅ Full control over the verification pipeline via configuration
- ✅ Native integration with [Delft-FEWS](https://oss.deltares.nl/web/delft-fews) 
- ✅ Native integration with local and remote (S3) [Zarr](https://zarr.dev/) 
- ✅ Builds on [scores](https://scores.readthedocs.io/en/stable/) for computation of scores. This package has extensive functionality, and it's documentation is world-class.
- ✅ Extensible with your own (private) datasources, scores and datasinks
- ✅ Optimized internal datamodel for efficient computation

## Installation

Install from PyPI:

```bash
pip install veriflow
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install veriflow
```

See [CONTRIBUTING.md](https://github.com/Deltares/veriflow/blob/main/CONTRIBUTING.md) for development setup.

## Running a verification pipeline with _veriflow_
The below example shows you the minimal code to run a verification pipeline.

```python
from veriflow import run_pipeline
from pathlib import Path

# Create your config and point to the file
path_to_config = Path("./config.yml")

# Run your verification pipeline in just one line of code
run_pipeline((path_to_config, "yaml"))
```

If it's your wish to further analyze the input data and verification results in an interactive Python session, assign the returned `OutputDataset` from `run_pipeline` to a Python variable. For each each configured `verification_pair`*, the `output_dataset` contains an `xarray.Dataset` with the results. By default, the input data will be included in the output.

*_a verification pair is definition of two datasources (e.g. observed and simulated) for one pysical variable (e.g. discharge or temperature)_
```python
# Alternatively, assign the returned `output_dataset` to a Python variable.
output_dataset = run_pipeline((path_to_config, "yaml"))

# List the verification_pairs in the output_dataset
verification_pairs = output_dataset.verification_pairs

# Retrieve the verification results for a verification_pair from the output_dataset
first_verification_pair = verification_pairs[0]
dataset = output_dataset.get(first_verification_pair) # An instance of xarray.Dataset
```

For more advanced documentation, please refer to our [user guide](https://deltares.github.io/veriflow/user_guide.html).

## 👥 Who Is This For?

This project is aimed at anyone who's interested in assessing model and forecast quality in an easy and reproducible way, like:
- operational forecasters
- model developers
- researchers and data-scientists

## Why this package?

Verification pipelines are complex: metrics require specialized computation, data volumes may exceed memory, and sources/destinations can vary widely. This package simplifies verification by handling the entire pipeline via a single configuration file. It's reliable (tested and versioned), transparent (fully documented), reproducible, and flexible—extensible with custom datasources, scores, and datasinks. Any pipeline created is immediately transferable to other users and systems.

So wether you're working on model development or operational forecasting: this tool can help you build robust and reproducible verification pipelines.


## Technical features
- Builds on [Xarray](https://docs.xarray.dev/en/stable/#) for handling multidimensional data. 
- Supports [Zarr](https://zarr.dev/) for cloud-friendly data storage
- Supports [Dask](https://www.dask.org/) for parallel and lazy computation

