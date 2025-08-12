#!/usr/bin/env python3
"""
GDELT GKG Data Ingestion - PySpark job for GDELT Global Knowledge Graph
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

def ingest_gdelt_data():
    """Ingest GDELT GKG data from landing zone to Delta bronze tables"""
    config = load_config()
    spark, is_delta = get_spark("ingest-gdelt")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        gdelt_config = config.get('data_ranges', {}).get('gdelt', {})
        start_date = gdelt_config.get('start', '2015-02-18T00:00:00Z')
        
        logger.info(f"Ingesting GDELT GKG data from {start_date}")
        
        gdelt_path = "landing/gdelt"
        
        if not Path(gdelt_path).exists():
            logger.warning("No GDELT data found in landing zone")
            return False
        
        gkg_df = spark.read.option("multiline", "true").csv(f"{gdelt_path}/**/*.csv", header=True, inferSchema=True)
        gkg_df = gkg_df.withColumn("_ingest_ts", lit(datetime.utcnow().isoformat()))
        gkg_df = gkg_df.withColumn("source", lit("gdelt_gkg"))
        
        if "DATE" in gkg_df.columns:
            gkg_df = gkg_df.withColumn("ts", to_timestamp(col("DATE"), "yyyyMMddHHmmss"))
        
        gkg_df = gkg_df.withColumn("year", year(col("ts")))
        gkg_df = gkg_df.withColumn("month", month(col("ts")))
        gkg_df = gkg_df.withColumn("day", dayofmonth(col("ts")))
        
        output_path = config.get('storage', {}).get('bronze', 'delta/bronze') + "/bronze_gkg"
        
        writer = gkg_df.write.mode("append")
        if fmt == "delta":
            writer = writer.partitionBy("year", "month", "day")
        
        write_table(writer, output_path, fmt)
        
        logger.info(f"Wrote {gkg_df.count()} GDELT GKG records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting GDELT data: {e}")
        return False
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Ingest GDELT GKG data to bronze layer')
    args = parser.parse_args()
    
    success = ingest_gdelt_data()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
