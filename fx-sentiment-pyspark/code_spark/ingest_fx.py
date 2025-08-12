#!/usr/bin/env python3
"""
FX Data Ingestion - PySpark job for Alpha Vantage and TwelveData
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

def ingest_fx_data():
    """Ingest FX data from landing zone to Delta bronze tables"""
    config = load_config()
    spark, is_delta = get_spark("ingest-fx")
    fmt = "delta" if is_delta else "parquet"
    
    try:
        fx_config = config.get('data_ranges', {}).get('fx', {})
        symbol = fx_config.get('symbol', 'EURUSD')
        
        logger.info(f"Ingesting FX data for {symbol}")
        
        alpha_path = "landing/alpha_vantage/type=daily"
        twelve_path = "landing/twelvedata/interval=1h"
        
        dfs = []
        
        if Path(alpha_path).exists():
            alpha_df = spark.read.parquet(alpha_path)
            alpha_df = alpha_df.withColumn("source", lit("alpha_vantage"))
            alpha_df = alpha_df.withColumn("interval", lit("daily"))
            dfs.append(alpha_df)
            logger.info(f"Loaded Alpha Vantage data: {alpha_df.count()} rows")
        
        if Path(twelve_path).exists():
            twelve_df = spark.read.parquet(twelve_path)
            twelve_df = twelve_df.withColumn("source", lit("twelvedata"))
            twelve_df = twelve_df.withColumn("interval", lit("1h"))
            dfs.append(twelve_df)
            logger.info(f"Loaded TwelveData data: {twelve_df.count()} rows")
        
        if not dfs:
            logger.warning("No FX data found in landing zone")
            return False
        
        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = combined_df.unionByName(df, allowMissingColumns=True)
        
        combined_df = combined_df.withColumn("_ingest_ts", lit(datetime.utcnow().isoformat()))
        
        if "date" in combined_df.columns:
            combined_df = combined_df.withColumn("ts", to_timestamp(col("date"), "yyyy-MM-dd"))
        elif "datetime" in combined_df.columns:
            combined_df = combined_df.withColumn("ts", to_timestamp(col("datetime")))
        
        combined_df = combined_df.withColumn("year", year(col("ts")))
        combined_df = combined_df.withColumn("month", month(col("ts")))
        combined_df = combined_df.withColumn("day", dayofmonth(col("ts")))
        
        output_path = config.get('storage', {}).get('bronze', 'delta/bronze') + "/bronze_fx"
        
        writer = combined_df.write.mode("append")
        if fmt == "delta":
            writer = writer.partitionBy("year", "month", "day")
        
        write_table(writer, output_path, fmt)
        
        logger.info(f"Wrote {combined_df.count()} FX records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting FX data: {e}")
        return False
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Ingest FX data to bronze layer')
    args = parser.parse_args()
    
    success = ingest_fx_data()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
