#!/usr/bin/env python3
"""
Batch Prediction - Apply latest saved model over time range with no row caps
"""
import os
import sys
import argparse
import logging
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import mlflow.spark

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table
from code_spark.conf import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_batch(output_path: str,
                  start_ts: str,
                  end_ts: str,
                  symbol: str = "EURUSD") -> str:
    """Apply latest saved model over [start_ts,end_ts]; no row caps; write Parquet and return its path."""
    config = load_config()
    spark, is_delta = get_spark("predict-batch")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        silver_path = config.get('storage', {}).get('silver', 'delta/silver')
        
        features_df = read_table(spark, f"{silver_path}/silver_eurusd_1h_features", fmt)
        logger.info(f"Loaded {features_df.count()} feature records")
        
        filtered_df = features_df.filter(
            (col("ts") >= start_ts) & (col("ts") <= end_ts) & (col("symbol") == symbol)
        )
        
        logger.info(f"Filtered to {filtered_df.count()} records for prediction")
        
        try:
            model = mlflow.spark.load_model("artifacts/models/latest/model")
            logger.info("Loaded model from artifacts/models/latest/model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return ""
        
        predictions_df = model.transform(filtered_df)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_table(predictions_df.write.mode("overwrite"), output_path, "parquet")
        
        logger.info(f"Saved {predictions_df.count()} predictions to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return ""
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Batch prediction with no row caps')
    parser.add_argument('--output', required=True, help='Output path for predictions')
    parser.add_argument('--start', required=True, help='Start timestamp (ISO format)')
    parser.add_argument('--end', required=True, help='End timestamp (ISO format)')
    parser.add_argument('--symbol', default='EURUSD', help='Symbol to predict')
    
    args = parser.parse_args()
    
    result_path = predict_batch(
        output_path=args.output,
        start_ts=args.start,
        end_ts=args.end,
        symbol=args.symbol
    )
    
    if result_path:
        logger.info(f"Batch prediction completed: {result_path}")
        return True
    else:
        logger.error("Batch prediction failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
