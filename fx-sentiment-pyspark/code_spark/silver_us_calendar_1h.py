#!/usr/bin/env python3
"""
Silver US Calendar 1h - Build hourly US economic calendar features
"""
import os
import sys
import argparse
import logging
from datetime import datetime

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, lit, when, max as spark_max, year, month, dayofmonth
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table
from code_spark.conf import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_silver_calendar():
    """Build silver layer US calendar features with unlimited data processing"""
    config = load_config()
    spark, is_delta = get_spark("silver-calendar")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        bronze_path = config.get('storage', {}).get('bronze', 'delta/bronze')
        silver_path = config.get('storage', {}).get('silver', 'delta/silver')
        
        calendar_df = read_table(spark, f"{bronze_path}/bronze_fred_releases", fmt)
        logger.info(f"Loaded {calendar_df.count()} calendar records")
        
        calendar_df = calendar_df.withColumn("event_day", lit(1))
        calendar_df = calendar_df.withColumn("event_0h", lit(0))
        calendar_df = calendar_df.withColumn("pre_event_2h", lit(0))
        calendar_df = calendar_df.withColumn("post_event_2h", lit(0))
        
        calendar_df = calendar_df.withColumn("year", year(col("ts")))
        calendar_df = calendar_df.withColumn("month", month(col("ts")))
        calendar_df = calendar_df.withColumn("day", dayofmonth(col("ts")))
        
        output_path = f"{silver_path}/silver_us_calendar_1h"
        
        writer = calendar_df.write.mode("overwrite")
        if fmt == "delta":
            writer = writer.partitionBy("year", "month", "day")
        
        write_table(writer, output_path, fmt)
        
        logger.info(f"Wrote {calendar_df.count()} silver calendar records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error building silver calendar: {e}")
        return False
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Build silver US calendar features')
    args = parser.parse_args()
    
    success = build_silver_calendar()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
