from feast import FileSource
from feast.data_format import ParquetFormat

eurusd_source = FileSource(
    path="delta/silver/silver_eurusd_1h_features",
    file_format=ParquetFormat(),
    timestamp_field="ts",
)
