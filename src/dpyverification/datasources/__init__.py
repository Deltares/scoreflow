"""The various datasources that can be used as input data."""

from .base import BaseDatasource, BaseDatasourceConfig
from .csv import Csv
from .fewsnetcdf import FewsNetCDF, FewsNetCDFConfig
from .fewswebservice import FewsWebservice, FewsWebserviceConfig
from .inputschemas import validate_input_data
from .netcdf import NetCDF
from .thresholds import ThresholdCsv, ThresholdCsvConfig

DEFAULT_DATASOURCES: list[type[BaseDatasource]] = [
    FewsNetCDF,
    FewsWebservice,
    NetCDF,
    Csv,
]
