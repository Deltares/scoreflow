from dpyverification.configuration.base import Config
from pathlib import Path
from dpyverification.pipeline import execute_pipeline

# For validation, we write the schema
schema_path = Path("config/config_schema.json")
Config.write_yaml_schema(schema_path)

import xarray as xr
ds = xr.open_dataarray("data/test_EMOS_obs.nc")
ds.attrs["timeseries_kind"] = "observed_historical"
ds.to_netcdf("data/test_EMOS_obs2.nc")

path_to_config = Path("config/config.yaml")
verification_dataset = execute_pipeline((path_to_config, "yaml"))

verification_dataset.isel(verification_pair=0, variable=0, station=1)["reliability_for_ensemble"]

print(test)