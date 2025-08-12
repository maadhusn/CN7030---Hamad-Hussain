#!/usr/bin/env python3
"""
Gold Training Matrix - Build labeled 1h rows with chronological train/valid/test tags
"""
import os
import sys
import argparse
import logging
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import col, when, row_number, year, month, dayofmonth

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table
from code_spark.conf import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_training_dataframe() -> DataFrame:
    """Load gold matrix from delta/gold, return Spark DataFrame with ['ts','symbol',features...,label,split]."""
    config = load_config()
    spark, is_delta = get_spark("gold-training-matrix")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        silver_path = config.get('storage', {}).get('silver', 'delta/silver')
        
        features_df = read_table(spark, f"{silver_path}/silver_eurusd_1h_features", fmt)
        logger.info(f"Loaded {features_df.count()} feature records")
        
        total_rows = features_df.count()
        train_end = int(total_rows * 0.70)
        valid_end = int(total_rows * 0.85)
        
        window_spec = Window.orderBy("ts")
        result_df = features_df.withColumn("row_num", row_number().over(window_spec))
        
        result_df = result_df.withColumn(
            "split",
            when(col("row_num") <= train_end, "train")
            .when(col("row_num") <= valid_end, "valid")
            .otherwise("test")
        ).drop("row_num")
        
        result_df = result_df.withColumn("symbol", col("symbol").cast("string"))
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error building training dataframe: {e}")
        return None

def build_gold_training_matrix():
    """Build and save gold training matrix"""
    config = load_config()
    spark, is_delta = get_spark("gold-training-matrix")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        training_df = build_training_dataframe()
        
        if training_df is None:
            logger.error("Failed to build training dataframe")
            return False
        
        training_df = training_df.withColumn("year", year(col("ts")))
        training_df = training_df.withColumn("month", month(col("ts")))
        training_df = training_df.withColumn("day", dayofmonth(col("ts")))
        
        gold_path = config.get('storage', {}).get('gold', 'delta/gold')
        output_path = f"{gold_path}/gold_training_matrix"
        
        writer = training_df.write.mode("overwrite")
        if fmt == "delta":
            writer = writer.partitionBy("year", "month", "day")
        
        write_table(writer, output_path, fmt)
        
        logger.info(f"Wrote {training_df.count()} gold training matrix records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error building gold training matrix: {e}")
        return False
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Build gold training matrix')
    args = parser.parse_args()
    
    success = build_gold_training_matrix()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
