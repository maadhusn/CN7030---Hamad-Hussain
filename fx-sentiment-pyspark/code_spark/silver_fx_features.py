#!/usr/bin/env python3
"""
Silver FX Features - Build 1h FX features with rolling windows and macro series
"""
import os
import sys
import argparse
import logging
from datetime import datetime

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, lit, lag, lead, avg, stddev, min as spark_min, max as spark_max,
    sum as spark_sum, count, abs as spark_abs, sqrt, pow as spark_pow,
    year, month, dayofmonth, coalesce
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table
from code_spark.conf import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_silver_fx_features():
    """Build silver layer FX features with unlimited data processing"""
    config = load_config()
    spark, is_delta = get_spark("silver-fx-features")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        bronze_path = config.get('storage', {}).get('bronze', 'delta/bronze')
        silver_path = config.get('storage', {}).get('silver', 'delta/silver')
        
        fx_df = read_table(spark, f"{bronze_path}/bronze_fx", fmt)
        logger.info(f"Loaded {fx_df.count()} FX records")
        
        window_spec = Window.orderBy("ts")
        
        fx_df = fx_df.withColumn("ret_1", (col("close") / lag("close", 1).over(window_spec)) - 1)
        fx_df = fx_df.withColumn("ret_3", (col("close") / lag("close", 3).over(window_spec)) - 1)
        fx_df = fx_df.withColumn("ret_6", (col("close") / lag("close", 6).over(window_spec)) - 1)
        
        fx_df = fx_df.withColumn("rv_6", stddev("ret_1").over(window_spec.rowsBetween(-5, 0)))
        fx_df = fx_df.withColumn("rv_24", stddev("ret_1").over(window_spec.rowsBetween(-23, 0)))
        
        fx_df = fx_df.withColumn("ema_6", avg("close").over(window_spec.rowsBetween(-5, 0)))
        fx_df = fx_df.withColumn("ema_24", avg("close").over(window_spec.rowsBetween(-23, 0)))
        
        fx_df = fx_df.withColumn("year", year(col("ts")))
        fx_df = fx_df.withColumn("month", month(col("ts")))
        fx_df = fx_df.withColumn("day", dayofmonth(col("ts")))
        
        output_path = f"{silver_path}/silver_fx_features"
        
        writer = fx_df.write.mode("overwrite")
        if fmt == "delta":
            writer = writer.partitionBy("year", "month", "day")
        
        write_table(writer, output_path, fmt)
        
        logger.info(f"Wrote {fx_df.count()} silver FX feature records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error building silver FX features: {e}")
        return False
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Build silver FX features')
    args = parser.parse_args()
    
    success = build_silver_fx_features()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
