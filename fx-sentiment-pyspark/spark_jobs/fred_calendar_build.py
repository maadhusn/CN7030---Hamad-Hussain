#!/usr/bin/env python3
"""
Build hourly US event calendar from FRED release dates
"""
import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, explode, sequence, to_timestamp, hour, date_format,
    when, expr, max as spark_max, array, struct, from_unixtime
)
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, BooleanType, IntegerType

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/calendar.yaml") -> dict:
    """Load calendar configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('calendar', {})
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def load_date_range(config_path: str = "configs/data.yaml") -> tuple:
    """Load date range from data config"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        date_range = config.get('date_range', {})
        start_date = date_range.get('start', '2019-01-01')
        end_date = date_range.get('end', '2024-12-31')
        return start_date, end_date
    except Exception as e:
        logger.error(f"Error loading date range: {e}")
        return '2019-01-01', '2024-12-31'

def create_hourly_timeline(spark: SparkSession, start_date: str, end_date: str) -> DataFrame:
    """Create a continuous hourly timeline"""
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() + 86400)  # Add a day to include end date
    
    df = spark.sql(f"""
    SELECT explode(sequence(
        to_timestamp({start_ts}), 
        to_timestamp({end_ts}), 
        interval 1 hour
    )) as ts
    """)
    
    return df

def build_event_flags(
    spark: SparkSession, 
    timeline_df: DataFrame, 
    releases_df: DataFrame, 
    calendar_mode: str,
    approx_times: dict
) -> DataFrame:
    """Build event flags based on calendar mode"""
    
    timeline_df.createOrReplaceTempView("timeline")
    
    if calendar_mode == "day":
        releases_df = releases_df.withColumn(
            "date_utc", 
            to_timestamp(col("date"), "yyyy-MM-dd")
        )
        
        result_df = timeline_df.withColumn(
            "date", date_format(col("ts"), "yyyy-MM-dd")
        ).join(
            releases_df.select(
                date_format(col("date_utc"), "yyyy-MM-dd").alias("date"),
                col("release_name")
            ),
            on="date",
            how="left"
        )
        
        result_df = result_df.withColumn(
            "event_day", when(col("release_name").isNotNull(), 1).otherwise(0)
        ).withColumn(
            "pre_event_2h", lit(0)  # No specific hour in day mode
        ).withColumn(
            "event_0h", lit(0)      # No specific hour in day mode
        ).withColumn(
            "post_event_2h", lit(0) # No specific hour in day mode
        )
        
    else:  # approx_time mode
        releases_df = releases_df.withColumn(
            "date_utc", 
            to_timestamp(col("date"), "yyyy-MM-dd")
        )
        
        approx_time_mapping = []
        for release_name, time_str in approx_times.items():
            hour_val = int(time_str.split(':')[0])
            approx_time_mapping.append((release_name, hour_val))
            
        approx_time_df = spark.createDataFrame(
            approx_time_mapping, 
            ["release_name", "event_hour"]
        )
        
        releases_with_times = releases_df.join(
            approx_time_df,
            on="release_name",
            how="inner"
        )
        
        releases_with_times = releases_with_times.withColumn(
            "event_ts", 
            expr("date_utc + make_interval(0, 0, 0, 0, event_hour, 0, 0)")
        ).withColumn(
            "pre_event_ts_1", expr("event_ts - interval 1 hour")
        ).withColumn(
            "pre_event_ts_2", expr("event_ts - interval 2 hours")
        ).withColumn(
            "post_event_ts_1", expr("event_ts + interval 1 hour")
        ).withColumn(
            "post_event_ts_2", expr("event_ts + interval 2 hours")
        )
        
        result_df = timeline_df.join(
            releases_with_times.select(
                col("event_ts"),
                col("pre_event_ts_1"),
                col("pre_event_ts_2"),
                col("post_event_ts_1"),
                col("post_event_ts_2"),
                col("release_name"),
                date_format(col("date_utc"), "yyyy-MM-dd").alias("event_date")
            ),
            timeline_df.ts.between(
                releases_with_times.date_utc,
                expr("date_utc + interval 23 hours + interval 59 minutes")
            ),
            how="left_outer"
        )
        
        result_df = result_df.withColumn(
            "event_day", 
            when(col("event_date").isNotNull(), 1).otherwise(0)
        ).withColumn(
            "event_0h", 
            when(col("ts") == col("event_ts"), 1).otherwise(0)
        ).withColumn(
            "pre_event_2h", 
            when(
                (col("ts") == col("pre_event_ts_1")) | 
                (col("ts") == col("pre_event_ts_2")), 
                1
            ).otherwise(0)
        ).withColumn(
            "post_event_2h", 
            when(
                (col("ts") == col("post_event_ts_1")) | 
                (col("ts") == col("post_event_ts_2")), 
                1
            ).otherwise(0)
        )
    
    result_df = result_df.groupBy("ts").agg(
        spark_max("event_day").alias("event_day_any"),
        spark_max("pre_event_2h").alias("pre_event_2h_any"),
        spark_max("event_0h").alias("event_0h_any"),
        spark_max("post_event_2h").alias("post_event_2h_any")
    )
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Build US event calendar from FRED release dates')
    parser.add_argument('--horizon', default='1h', help='Time horizon (default: 1h)')
    parser.add_argument('--delta-path', default='delta', help='Delta lake path (default: delta)')
    
    args = parser.parse_args()
    
    calendar_config = load_config()
    start_date, end_date = load_date_range()
    
    calendar_source = calendar_config.get('source', 'fred')
    calendar_mode = calendar_config.get('mode', 'day')
    approx_times = calendar_config.get('approx_times_utc', {})
    
    spark, is_delta = get_spark("fred-calendar-build")
    fmt = "delta" if is_delta else "parquet"
    
    logger.info(f"Using storage format: {fmt}")
    logger.info(f"Calendar source: {calendar_source}, mode: {calendar_mode}")
    
    try:
        bronze_path = f"{args.delta_path}/bronze/bronze_fred_releases"
        
        try:
            releases_df = read_table(spark, bronze_path, fmt)
            logger.info(f"Read {releases_df.count()} records from bronze_fred_releases")
        except Exception as e:
            logger.error(f"Error reading bronze_fred_releases: {e}")
            return False
        
        timeline_df = create_hourly_timeline(spark, start_date, end_date)
        logger.info(f"Created hourly timeline from {start_date} to {end_date}")
        
        calendar_df = build_event_flags(
            spark, 
            timeline_df, 
            releases_df, 
            calendar_mode,
            approx_times
        )
        
        silver_path = f"{args.delta_path}/silver/silver_us_calendar_1h"
        writer = calendar_df.write.mode("overwrite")
        write_table(writer, silver_path, fmt)
        
        logger.info(f"Wrote {calendar_df.count()} records to silver_us_calendar_1h")
        return True
        
    except Exception as e:
        logger.error(f"Error building US event calendar: {e}")
        return False
        
    finally:
        spark.stop()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
