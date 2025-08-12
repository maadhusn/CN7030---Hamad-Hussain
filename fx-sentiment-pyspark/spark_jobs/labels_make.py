#!/usr/bin/env python3
"""
Generate 3-class labels with dead-zone for FX data
"""
import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, lit, lead, lag, log, exp, when, percentile_approx, abs as spark_abs
)

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/features.yaml") -> dict:
    """Load features configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def generate_labels(
    spark: SparkSession, 
    silver_df: DataFrame, 
    label_mode: str,
    fixed_eps_bps: float,
    adaptive_pctile: float
) -> DataFrame:
    """Generate 3-class labels with dead-zone"""
    try:
        window_spec = Window.orderBy("ts")
        
        labeled_df = silver_df.withColumn(
            "future_close", 
            lead("close", 1).over(window_spec)
        ).withColumn(
            "log_return", 
            log(col("future_close") / col("close"))
        )
        
        if label_mode == "fixed":
            eps = fixed_eps_bps / 10000.0
            
            labeled_df = labeled_df.withColumn(
                "label",
                when(col("log_return") > eps, 1)
                .when(col("log_return") < -eps, -1)
                .otherwise(0)
            )
            
            logger.info(f"Applied fixed dead-zone with epsilon = {eps} (log-return)")
            
        else:  # adaptive mode
            abs_returns = labeled_df.select(
                spark_abs(col("log_return")).alias("abs_return")
            ).filter(
                col("abs_return").isNotNull()
            )
            
            threshold_df = abs_returns.agg(
                percentile_approx("abs_return", adaptive_pctile, 10000).alias("threshold")
            )
            
            threshold = threshold_df.collect()[0]["threshold"]
            
            labeled_df = labeled_df.withColumn(
                "label",
                when(col("log_return") > threshold, 1)
                .when(col("log_return") < -threshold, -1)
                .otherwise(0)
            )
            
            logger.info(f"Applied adaptive dead-zone with {adaptive_pctile} percentile = {threshold} (log-return)")
            
        return labeled_df
        
    except Exception as e:
        logger.error(f"Error generating labels: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate 3-class labels with dead-zone for FX data')
    parser.add_argument('--horizon', default='1h', help='Time horizon (default: 1h)')
    parser.add_argument('--label-mode', choices=['fixed', 'adaptive'], default='adaptive',
                       help='Label mode (default: adaptive)')
    parser.add_argument('--silver-path', default='delta/silver', help='Silver path (default: delta/silver)')
    
    args = parser.parse_args()
    
    features_config = load_config()
    
    label_config = features_config.get('label', {})
    fixed_eps_bps = label_config.get('fixed_eps_bps', 3)
    adaptive_pctile = label_config.get('adaptive_pctile', 0.40)
    
    spark, is_delta = get_spark("labels-make")
    fmt = "delta" if is_delta else "parquet"
    
    logger.info(f"Using storage format: {fmt}")
    logger.info(f"Label mode: {args.label_mode}")
    
    try:
        silver_path = f"{args.silver_path}/silver_eurusd_1h"
        
        try:
            silver_df = read_table(spark, silver_path, fmt)
            logger.info(f"Read {silver_df.count()} records from silver_eurusd_1h")
        except Exception as e:
            logger.error(f"Error reading silver_eurusd_1h: {e}")
            return False
            
        labeled_df = generate_labels(
            spark, 
            silver_df, 
            args.label_mode,
            fixed_eps_bps,
            adaptive_pctile
        )
        
        if labeled_df is None:
            logger.error("Failed to generate labels")
            return False
            
        labeled_path = f"{args.silver_path}/silver_eurusd_1h_labeled"
        writer = labeled_df.write.mode("overwrite")
        write_table(writer, labeled_path, fmt)
        
        label_counts = labeled_df.groupBy("label").count().collect()
        
        logger.info("Label distribution:")
        total_count = sum(row["count"] for row in label_counts)
        
        for row in label_counts:
            label = row["label"]
            count = row["count"]
            pct = (count / total_count) * 100 if total_count > 0 else 0
            
            label_name = "DOWN" if label == -1 else "FLAT" if label == 0 else "UP"
            logger.info(f"  {label_name} ({label}): {count} ({pct:.2f}%)")
            
        logger.info(f"Wrote {labeled_df.count()} records to silver_eurusd_1h_labeled")
        return True
        
    except Exception as e:
        logger.error(f"Error generating labels: {e}")
        return False
        
    finally:
        spark.stop()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
