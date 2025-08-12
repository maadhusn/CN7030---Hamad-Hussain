#!/usr/bin/env python3
"""
FRED Data Ingestion - PySpark job for FRED series and releases
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_timestamp, year, month, dayofmonth
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table
from code_spark.conf import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_fred_data():
    """Ingest FRED data from landing zone to Delta bronze tables"""
    config = load_config()
    spark, is_delta = get_spark("ingest-fred")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        fred_config = config.get('data_ranges', {}).get('fred', {})
        series_list = fred_config.get('series', ['CPIAUCSL', 'PCEPI'])
        
        logger.info(f"Ingesting FRED data for series: {series_list}")
        
        fred_series_path = "landing/fred_series"
        fred_releases_path = "landing/fred_releases"
        
        if Path(fred_series_path).exists():
            series_df = spark.read.parquet(f"{fred_series_path}/*.parquet")
            series_df = series_df.withColumn("_ingest_ts", lit(datetime.utcnow().isoformat()))
            
            if "date" in series_df.columns:
                series_df = series_df.withColumn("ts", to_timestamp(col("date"), "yyyy-MM-dd"))
            
            series_df = series_df.withColumn("year", year(col("ts")))
            series_df = series_df.withColumn("month", month(col("ts")))
            series_df = series_df.withColumn("day", dayofmonth(col("ts")))
            
            output_path = config.get('storage', {}).get('bronze', 'delta/bronze') + "/bronze_fred_series"
            
            writer = series_df.write.mode("append")
            if fmt == "delta":
                writer = writer.partitionBy("year", "month", "day")
            
            write_table(writer, output_path, fmt)
            logger.info(f"Wrote {series_df.count()} FRED series records to {output_path}")
        
        if Path(fred_releases_path).exists():
            releases_df = spark.read.parquet(f"{fred_releases_path}/**/*.parquet")
            releases_df = releases_df.withColumn("_ingest_ts", lit(datetime.utcnow().isoformat()))
            
            if "date_utc" in releases_df.columns:
                releases_df = releases_df.withColumn("ts", to_timestamp(col("date_utc")))
            
            releases_df = releases_df.withColumn("year", year(col("ts")))
            releases_df = releases_df.withColumn("month", month(col("ts")))
            releases_df = releases_df.withColumn("day", dayofmonth(col("ts")))
            
            output_path = config.get('storage', {}).get('bronze', 'delta/bronze') + "/bronze_fred_releases"
            
            writer = releases_df.write.mode("append")
            if fmt == "delta":
                writer = writer.partitionBy("year", "month", "day")
            
            write_table(writer, output_path, fmt)
            logger.info(f"Wrote {releases_df.count()} FRED releases records to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting FRED data: {e}")
        return False
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Ingest FRED data to bronze layer')
    args = parser.parse_args()
    
    success = ingest_fred_data()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
