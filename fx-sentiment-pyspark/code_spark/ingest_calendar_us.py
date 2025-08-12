#!/usr/bin/env python3
"""
US Calendar Data Ingestion - PySpark job for US economic calendar events
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

def ingest_calendar_data():
    """Ingest US calendar data from landing zone to Delta bronze tables"""
    config = load_config()
    spark, is_delta = get_spark("ingest-calendar")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        calendar_config = config.get('data_ranges', {}).get('calendar_us', {})
        
        logger.info("Ingesting US calendar data")
        
        calendar_path = "landing/calendar_us"
        
        if not Path(calendar_path).exists():
            logger.warning("No US calendar data found in landing zone")
            return False
        
        calendar_df = spark.read.parquet(f"{calendar_path}/**/*.parquet")
        calendar_df = calendar_df.withColumn("_ingest_ts", lit(datetime.utcnow().isoformat()))
        calendar_df = calendar_df.withColumn("source", lit("us_calendar"))
        
        if "date" in calendar_df.columns:
            calendar_df = calendar_df.withColumn("ts", to_timestamp(col("date"), "yyyy-MM-dd"))
        
        calendar_df = calendar_df.withColumn("year", year(col("ts")))
        calendar_df = calendar_df.withColumn("month", month(col("ts")))
        calendar_df = calendar_df.withColumn("day", dayofmonth(col("ts")))
        
        output_path = config.get('storage', {}).get('bronze', 'delta/bronze') + "/bronze_calendar_us"
        
        writer = calendar_df.write.mode("append")
        if fmt == "delta":
            writer = writer.partitionBy("year", "month", "day")
        
        write_table(writer, output_path, fmt)
        
        logger.info(f"Wrote {calendar_df.count()} US calendar records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting US calendar data: {e}")
        return False
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Ingest US calendar data to bronze layer')
    args = parser.parse_args()
    
    success = ingest_calendar_data()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
