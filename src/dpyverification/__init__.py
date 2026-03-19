"""
DPyVerification: a package to run a selection of verification functions on a dataset.

The data the functions apply to are observations and (a combination of multiple) forecasts.

"""

from dpyverification.datasources.inputschemas import validate_input_data
from dpyverification.pipeline import execute_pipeline
