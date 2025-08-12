from pyspark.sql import SparkSession

def get_spark(app_name: str, force_parquet: bool = False):
    if force_parquet:
        spark = SparkSession.builder.appName(app_name).getOrCreate()
        return spark, False
    try:
        from delta import configure_spark_with_delta_pip
        builder = (SparkSession.builder.appName(app_name)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        return spark, True
    except Exception:
        spark = SparkSession.builder.appName(app_name).getOrCreate()
        return spark, False
