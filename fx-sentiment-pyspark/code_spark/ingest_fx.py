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
            logger.warning("No FX data found in landing zone, generating mock data for Colab testing")
            
            from datetime import datetime, timedelta
            import os
            
            start_date_str = os.getenv('START_DATE')
            end_date_str = os.getenv('END_DATE')
            
            if start_date_str and end_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            else:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=90)
            
            logger.info(f"Generating mock FX data from {start_date.date()} to {end_date.date()}")
            
            mock_data = []
            base_price = 1.1450
            current_date = start_date
            
            while current_date <= end_date:
                days_elapsed = (current_date - start_date).days
                price_change = (days_elapsed % 20 - 10) * 0.0001
                close_price = base_price + price_change
                
                mock_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'open': close_price - 0.0002,
                    'high': close_price + 0.0005,
                    'low': close_price - 0.0005,
                    'close': close_price,
                    'volume': None,
                    'source': 'mock_alpha_vantage',
                    'collected_at': datetime.utcnow().isoformat()
                })
                
                current_date += timedelta(days=1)
            
            mock_df = spark.createDataFrame(mock_data)
            mock_df = mock_df.withColumn("source", lit("mock_alpha_vantage"))
            mock_df = mock_df.withColumn("interval", lit("daily"))
            
            combined_df = mock_df
            logger.info(f"Generated {len(mock_data)} mock FX records")
        else:
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
