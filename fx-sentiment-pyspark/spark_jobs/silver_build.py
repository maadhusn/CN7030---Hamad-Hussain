#!/usr/bin/env python3
"""
Build silver layer by aligning data to 1h and joining everything
"""
import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, lit, hour, date_format, to_timestamp, from_unixtime, 
    window, count, sum as spark_sum, avg, lower, when, regexp_extract,
    explode, array_contains, expr, max as spark_max, min as spark_min,
    first, last, lag, lead, row_number, rank, dense_rank, ntile,
    year, month, dayofmonth, dayofweek, weekofyear, quarter,
    unix_timestamp, datediff, months_between, add_months, date_add, date_sub,
    concat, concat_ws, coalesce, isnull, isnan, nullif, nvl, substring,
    trim, ltrim, rtrim, lpad, rpad, regexp_replace, translate, initcap,
    soundex, levenshtein, format_number, format_string, printf, 
    base64, unbase64, md5, sha1, sha2, crc32,
    to_json, from_json,
    struct, array, map_keys, map_values,
    size, sort_array, array_distinct, array_intersect, array_except, array_union,
    array_position, array_sort, array_min, array_max,
    collect_list, collect_set, explode_outer, posexplode, posexplode_outer,
    get_json_object, json_tuple, schema_of_json,
    current_timestamp, current_date, to_date,
    session_window, inline, inline_outer,
    udf, pandas_udf, transform, exists, forall, filter, aggregate
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, 
    FloatType, DoubleType, BooleanType, TimestampType, DateType,
    ArrayType, MapType
)
import pyspark.sql.functions as F

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

def build_fx_1h(spark: SparkSession, bronze_path: str, fmt: str) -> DataFrame:
    """Build 1h FX data from bronze_fx table"""
    try:
        try:
            fx_df = read_table(spark, f"{bronze_path}/bronze_fx", fmt)
        except Exception as e:
            logger.warning(f"bronze_fx table not found: {e}")
            logger.info("Creating mock FX data for testing")
            
            from datetime import datetime, timedelta
            import pandas as pd
            
            start_date = datetime(2019, 1, 1)
            end_date = datetime(2019, 1, 10)
            
            timestamps = []
            current = start_date
            while current <= end_date:
                timestamps.append(current)
                current += timedelta(hours=1)
            
            mock_data = []
            base_price = 1.1450
            
            for i, ts in enumerate(timestamps):
                price_change = (i % 10 - 5) * 0.001
                close_price = base_price + price_change
                
                mock_data.append({
                    'ts': ts,
                    'symbol': 'EURUSD',
                    'open': close_price - 0.0005,
                    'high': close_price + 0.001,
                    'low': close_price - 0.001,
                    'close': close_price,
                    '_source': 'mock'
                })
            
            # Convert to Spark DataFrame
            mock_df = spark.createDataFrame(mock_data)
            return mock_df
        
        twelve_df = fx_df.filter(col("_source") == "twelvedata")
        
        if twelve_df.count() > 0:
            logger.info("Using TwelveData for 1h FX data")
            
            result_df = twelve_df.withColumn(
                "ts", 
                to_timestamp(col("date"))
            ).select(
                col("ts"),
                col("symbol"),
                col("open"),
                col("high"),
                col("low"),
                col("close"),
                col("_source")
            )
            
        else:
            logger.info("Using Alpha Vantage for 1h FX data (upsampling daily)")
            
            alpha_df = fx_df.filter(col("_source") == "alpha_vantage")
            
            if alpha_df.count() == 0:
                logger.warning("No FX data found")
                return None
                
            alpha_df = alpha_df.withColumn(
                "date_ts", 
                to_timestamp(col("date"))
            )
            
            hours_df = spark.sql("SELECT explode(sequence(0, 23)) as hour")
            
            result_df = alpha_df.crossJoin(hours_df).withColumn(
                "ts",
                expr("date_ts + make_interval(0, 0, 0, 0, hour, 0, 0)")
            ).select(
                col("ts"),
                col("symbol"),
                col("open"),
                col("high"),
                col("low"),
                col("close"),
                col("_source")
            )
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error building FX 1h data: {e}")
        return None

def build_gdelt_1h(spark: SparkSession, bronze_path: str, fmt: str, keywords: list) -> DataFrame:
    """Build 1h GDELT data with keyword counts, rates, and z-scores"""
    try:
        gkg_df = read_table(spark, f"{bronze_path}/bronze_gkg_norm", fmt)
        
        if gkg_df.count() == 0:
            logger.warning("No GDELT data found")
            return None
            
        tone_df = gkg_df.filter(
            col("tone").isNotNull()
        ).groupBy(
            window(col("ts"), "1 hour").alias("window")
        ).agg(
            avg(col("tone")).alias("tone_mean_1h"),
            count("*").alias("doc_count_1h")
        ).withColumn(
            "ts",
            col("window").getField("start")
        ).drop("window")
        
        keyword_dfs = []
        
        for keyword in keywords:
            keyword_slug = keyword.lower().replace(" ", "_")
            
            keyword_df = gkg_df.withColumn(
                "has_keyword",
                lower(col("themes_raw")).contains(keyword.lower()) |
                lower(col("docid")).contains(keyword.lower())
            ).groupBy(
                window(col("ts"), "1 hour").alias("window")
            ).agg(
                sum(when(col("has_keyword"), 1).otherwise(0)).alias(f"g_event_{keyword_slug}_count_1h")
            ).withColumn(
                "ts",
                col("window").getField("start")
            ).drop("window")
            
            keyword_dfs.append(keyword_df)
        
        result_df = tone_df
        
        for keyword_df in keyword_dfs:
            result_df = result_df.join(keyword_df, on="ts", how="outer")
            
        event_cols = [c for c in result_df.columns if c.startswith("g_event_") and c.endswith("_count_1h")]
        
        if event_cols:
            result_df = result_df.withColumn(
                "g_event_total_count_1h",
                sum([col(c) for c in event_cols])
            )
            
        for col_name in result_df.columns:
            if col_name.endswith("_count_1h"):
                result_df = result_df.withColumn(
                    col_name,
                    coalesce(col(col_name), lit(0))
                )
        
        result_df = result_df.withColumn(
            "doc_count_1h",
            coalesce(col("doc_count_1h"), lit(1))  # Avoid division by zero
        )
        
        for col_name in event_cols + ["g_event_total_count_1h"]:
            rate_col = col_name.replace("_count_1h", "_rate")
            result_df = result_df.withColumn(
                rate_col,
                col(col_name).cast("double") / col("doc_count_1h")
            )
        
        window_30d = Window.orderBy("ts").rowsBetween(-720, 0)  # 30 days * 24 hours
        
        rate_cols = [c for c in result_df.columns if c.endswith("_rate")]
        
        for rate_col in rate_cols:
            z_col = rate_col + "_z30"
            
            result_df = result_df.withColumn(
                f"{rate_col}_mean30",
                F.avg(rate_col).over(window_30d)
            ).withColumn(
                f"{rate_col}_std30",
                F.stddev(rate_col).over(window_30d)
            )
            
            result_df = result_df.withColumn(
                z_col,
                F.when(
                    col(f"{rate_col}_std30") > 0,
                    (col(rate_col) - col(f"{rate_col}_mean30")) / col(f"{rate_col}_std30")
                ).otherwise(lit(0.0))
            )
            
            result_df = result_df.drop(f"{rate_col}_mean30", f"{rate_col}_std30")
                
        return result_df
        
    except Exception as e:
        logger.error(f"Error building GDELT 1h data: {e}")
        return None

def build_fred_series_1h(spark: SparkSession, bronze_path: str, fmt: str) -> DataFrame:
    """Build 1h FRED series data with forward-fill"""
    try:
        fred_df = read_table(spark, f"{bronze_path}/bronze_fred_series", fmt)
        
        if fred_df.count() == 0:
            logger.warning("No FRED series data found")
            return None
            
        fred_df = fred_df.withColumn(
            "ts",
            to_timestamp(col("date"), "yyyy-MM-dd")
        )
        
        pivot_df = fred_df.groupBy("ts").pivot("series_id").agg(
            first("value").alias("value")
        )
        
        min_date = fred_df.agg(min("date")).collect()[0][0]
        max_date = fred_df.agg(max("date")).collect()[0][0]
        
        timeline_df = spark.sql(f"""
        SELECT explode(sequence(
            to_timestamp('{min_date}'), 
            to_timestamp('{max_date}') + interval 1 day, 
            interval 1 hour
        )) as ts
        """)
        
        result_df = timeline_df.join(
            pivot_df,
            on="ts",
            how="left"
        )
        
        window_spec = Window.orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
        
        for col_name in result_df.columns:
            if col_name != "ts":
                result_df = result_df.withColumn(
                    col_name,
                    last(col_name, True).over(window_spec)
                )
                
        return result_df
        
    except Exception as e:
        logger.error(f"Error building FRED series 1h data: {e}")
        return None

def build_wiki_1h(spark: SparkSession, bronze_path: str, fmt: str) -> DataFrame:
    """Build 1h Wikipedia pageviews data"""
    try:
        wiki_df = read_table(spark, f"{bronze_path}/bronze_wiki", fmt)
        
        if wiki_df.count() == 0:
            logger.warning("No Wikipedia data found")
            return None
            
        wiki_df = wiki_df.withColumn(
            "date_ts",
            to_timestamp(col("date"), "yyyy-MM-dd")
        )
        
        hours_df = spark.sql("SELECT explode(sequence(0, 23)) as hour")
        
        wiki_df = wiki_df.crossJoin(hours_df).withColumn(
            "ts",
            expr("date_ts + make_interval(0, 0, 0, 0, hour, 0, 0)")
        )
        
        result_df = wiki_df.groupBy("ts").pivot("article").agg(
            first("views").alias("views")  # Use first since all hours have same daily value
        )
        
        for col_name in result_df.columns:
            if col_name != "ts":
                article_slug = col_name.lower().replace(" ", "_")
                result_df = result_df.withColumnRenamed(
                    col_name,
                    f"wiki_{article_slug}_views"
                )
                
        view_cols = [c for c in result_df.columns if c.endswith("_views")]
        
        if view_cols:
            result_df = result_df.withColumn(
                "wiki_total_views",
                sum([col(c) for c in view_cols])
            )
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error building Wikipedia 1h data: {e}")
        return None

def join_silver_data(
    spark: SparkSession, 
    fx_df: DataFrame, 
    gdelt_df: DataFrame, 
    fred_df: DataFrame, 
    wiki_df: DataFrame,
    calendar_df: DataFrame
) -> DataFrame:
    """Join all silver data on timestamp"""
    try:
        if fx_df is None:
            logger.error("FX data is required for silver layer")
            return None
            
        result_df = fx_df
        
        if gdelt_df is not None:
            result_df = result_df.join(
                gdelt_df,
                on="ts",
                how="left"
            )
            
        if fred_df is not None:
            result_df = result_df.join(
                fred_df,
                on="ts",
                how="left"
            )
            
        if wiki_df is not None:
            result_df = result_df.join(
                wiki_df,
                on="ts",
                how="left"
            )
            
        if calendar_df is not None:
            result_df = result_df.join(
                calendar_df,
                on="ts",
                how="left"
            )
            
        for col_name in result_df.columns:
            if col_name.startswith("event_") or col_name.endswith("_count_1h"):
                result_df = result_df.withColumn(
                    col_name,
                    coalesce(col(col_name), lit(0))
                )
                
        return result_df
        
    except Exception as e:
        logger.error(f"Error joining silver data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Build silver layer by aligning data to 1h and joining everything')
    parser.add_argument('--horizon', default='1h', help='Time horizon (default: 1h)')
    parser.add_argument('--bronze-path', default='delta/bronze', help='Bronze path (default: delta/bronze)')
    parser.add_argument('--silver-path', default='delta/silver', help='Silver path (default: delta/silver)')
    
    args = parser.parse_args()
    
    calendar_config = load_config()
    keywords = calendar_config.get('gdelt_event_keywords', [])
    
    spark, is_delta = get_spark("silver-build")
    fmt = "delta" if is_delta else "parquet"
    
    logger.info(f"Using storage format: {fmt}")
    
    try:
        fx_df = build_fx_1h(spark, args.bronze_path, fmt)
        gdelt_df = build_gdelt_1h(spark, args.bronze_path, fmt, keywords)
        fred_df = build_fred_series_1h(spark, args.bronze_path, fmt)
        wiki_df = build_wiki_1h(spark, args.bronze_path, fmt)
        
        try:
            calendar_df = read_table(spark, f"{args.silver_path}/silver_us_calendar_1h", fmt)
            logger.info(f"Read {calendar_df.count()} records from silver_us_calendar_1h")
        except Exception as e:
            logger.warning(f"Error reading silver_us_calendar_1h: {e}")
            calendar_df = None
            
        silver_df = join_silver_data(
            spark, 
            fx_df, 
            gdelt_df, 
            fred_df, 
            wiki_df,
            calendar_df
        )
        
        if silver_df is None:
            logger.error("Failed to build silver layer")
            return False
            
        silver_path = f"{args.silver_path}/silver_eurusd_1h"
        writer = silver_df.write.mode("overwrite")
        if fmt == "delta":
            writer = writer.option("overwriteSchema", "true")
        write_table(writer, silver_path, fmt)
        
        logger.info(f"Wrote {silver_df.count()} records to silver_eurusd_1h")
        return True
        
    except Exception as e:
        logger.error(f"Error building silver layer: {e}")
        return False
        
    finally:
        spark.stop()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
