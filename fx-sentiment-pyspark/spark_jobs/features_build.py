#!/usr/bin/env python3
"""
Build features including rolling windows and surprise proxy
"""
import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, lit, lag, lead, when, avg, stddev, min as spark_min, max as spark_max,
    sum as spark_sum, count, abs as spark_abs, sqrt, pow as spark_pow,
    expr, row_number, rank, dense_rank, percent_rank, ntile,
    year, month, dayofmonth, dayofweek, weekofyear, quarter,
    unix_timestamp, datediff, months_between, add_months, date_add, date_sub,
    concat, concat_ws, coalesce, isnull, isnan, nullif, nvl, substring,
    trim, ltrim, rtrim, lpad, rpad, regexp_replace, translate, initcap,
    soundex, levenshtein, format_number, format_string, printf
)
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def build_price_features(df: DataFrame, ret_windows: list, vol_windows: list, ema_windows: list, spark) -> DataFrame:
    """Build price-based features: returns, realized volatility, EMAs"""
    try:
        result_df = df
        
        window_spec = Window.orderBy("ts")
        result_df = result_df.withColumn(
            "log_return", 
            F.log(col("close") / lag("close", 1).over(window_spec))
        )
        
        for w in ret_windows:
            window_spec_lag = Window.orderBy("ts")
            
            result_df = result_df.withColumn(
                f"ret_{w}",
                F.when(
                    F.lag("close", w).over(window_spec_lag).isNull(),
                    None
                ).otherwise(
                    F.log(col("close")) - F.log(F.lag("close", w).over(window_spec_lag))
                )
            )
            
        for w in vol_windows:
            window_spec_w = Window.orderBy("ts").rowsBetween(-w, 0)
            
            result_df = result_df.withColumn(
                f"rv_{w}",
                F.stddev("log_return").over(window_spec_w)
            )
            
        for w in ema_windows:
            alpha = 2.0 / (w + 1)
            
            ordered_data = result_df.select("ts", "close").orderBy("ts").collect()
            
            ema_values = []
            ema_prev = None
            close_prev = None
            
            for row in ordered_data:
                close_val = row["close"]
                if ema_prev is None:
                    ema_val = close_val  # Initialize with first close
                else:
                    ema_val = alpha * close_prev + (1 - alpha) * ema_prev
                ema_values.append((row["ts"], ema_val))
                ema_prev = ema_val
                close_prev = close_val  # Update for next iteration
            
            ema_df = spark.createDataFrame(ema_values, ["ts", f"ema_{w}"])
            result_df = result_df.join(ema_df, "ts", "left")
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error building price features: {e}")
        return df

def build_tone_features(df: DataFrame) -> DataFrame:
    """Build GDELT tone features"""
    try:
        result_df = df
        
        if "tone_mean_1h" not in df.columns:
            logger.warning("Tone column not found, skipping tone features")
            return df
            
        window_spec_24h = Window.orderBy("ts").rowsBetween(-23, 0)
        
        result_df = result_df.withColumn(
            "tone_mean_24",
            F.avg("tone_mean_1h").over(window_spec_24h)
        )
        
        alpha = 2.0 / (24 + 1)
        
        window_spec_lag = Window.orderBy("ts")
        result_df = result_df.withColumn(
            "tone_ema_24_init",
            col("tone_mean_24")
        ).withColumn(
            "tone_ema_24",
            F.when(
                F.lag("tone_ema_24_init", 1).over(window_spec_lag).isNull(),
                col("tone_ema_24_init")
            ).otherwise(
                F.lag("tone_mean_1h", 1).over(window_spec_lag) * lit(alpha) + 
                F.lag("tone_ema_24", 1).over(window_spec_lag) * (1 - alpha)
            )
        ).drop("tone_ema_24_init")
        
        window_spec_long = Window.orderBy("ts").rowsBetween(-168, 0)  # 7 days
        
        result_df = result_df.withColumn(
            "tone_mean_long",
            F.avg("tone_mean_1h").over(window_spec_long)
        ).withColumn(
            "tone_std_long",
            F.stddev("tone_mean_1h").over(window_spec_long)
        ).withColumn(
            "tone_z",
            (col("tone_mean_1h") - col("tone_mean_long")) / 
            F.when(col("tone_std_long") > 0, col("tone_std_long")).otherwise(1.0)
        ).drop("tone_mean_long", "tone_std_long")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error building tone features: {e}")
        return df

def build_attention_features(df: DataFrame) -> DataFrame:
    """Build Wikipedia attention features"""
    try:
        result_df = df
        
        wiki_cols = [c for c in df.columns if c.startswith("wiki_") and c.endswith("_views")]
        
        if not wiki_cols:
            logger.warning("Wiki columns not found, skipping attention features")
            return df
            
        window_spec_24h = Window.orderBy("ts").rowsBetween(-23, 0)
        
        for col_name in wiki_cols:
            result_df = result_df.withColumn(
                f"{col_name}_sum_24",
                F.sum(col_name).over(window_spec_24h)
            )
            
            alpha = 2.0 / (24 + 1)
            
            window_spec_lag = Window.orderBy("ts")
            result_df = result_df.withColumn(
                f"{col_name}_ema_24_init",
                col(f"{col_name}_sum_24")
            ).withColumn(
                f"{col_name}_ema_24",
                F.when(
                    F.lag(f"{col_name}_ema_24_init", 1).over(window_spec_lag).isNull(),
                    col(f"{col_name}_ema_24_init")
                ).otherwise(
                    F.lag(col_name, 1).over(window_spec_lag) * lit(alpha) + 
                    F.lag(f"{col_name}_ema_24", 1).over(window_spec_lag) * (1 - alpha)
                )
            ).drop(f"{col_name}_ema_24_init")
            
            window_spec_long = Window.orderBy("ts").rowsBetween(-168, 0)  # 7 days
            
            result_df = result_df.withColumn(
                f"{col_name}_mean_long",
                F.avg(col_name).over(window_spec_long)
            ).withColumn(
                f"{col_name}_std_long",
                F.stddev(col_name).over(window_spec_long)
            ).withColumn(
                f"{col_name}_z",
                (col(col_name) - col(f"{col_name}_mean_long")) / 
                F.when(col(f"{col_name}_std_long") > 0, col(f"{col_name}_std_long")).otherwise(1.0)
            ).drop(f"{col_name}_mean_long", f"{col_name}_std_long")
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error building attention features: {e}")
        return df

def build_macro_features(df: DataFrame) -> DataFrame:
    """Build macro features from FRED series"""
    try:
        result_df = df
        
        monthly_series = ["CPIAUCSL", "PCEPI", "UNRATE"]
        daily_series = ["FEDFUNDS", "DTWEXBGS"]
        
        for series in monthly_series:
            if series not in df.columns:
                logger.warning(f"{series} column not found, skipping")
                continue
                
            window_spec_lag = Window.orderBy("ts")
            
            result_df = result_df.withColumn(
                f"{series.lower()}_lag1m",
                F.lag(series, 720).over(window_spec_lag)  # 30 days * 24 hours
            ).withColumn(
                f"{series.lower()}_lag12m",
                F.lag(series, 8640).over(window_spec_lag)  # 360 days * 24 hours
            ).withColumn(
                f"{series.lower()}_mom",
                (col(series) / col(f"{series.lower()}_lag1m") - 1) * 100
            ).withColumn(
                f"{series.lower()}_yoy",
                (col(series) / col(f"{series.lower()}_lag12m") - 1) * 100
            ).drop(f"{series.lower()}_lag1m", f"{series.lower()}_lag12m")
            
        for series in daily_series:
            if series not in df.columns:
                logger.warning(f"{series} column not found, skipping")
                continue
                
            for window in [5, 20]:
                alpha = 2.0 / (window * 24 + 1)  # Convert days to hours
                
                window_spec_lag = Window.orderBy("ts")
                result_df = result_df.withColumn(
                    f"{series.lower()}_ema{window}_init",
                    col(series)
                ).withColumn(
                    f"{series.lower()}_ema{window}",
                    F.when(
                        F.lag(f"{series.lower()}_ema{window}_init", 1).over(window_spec_lag).isNull(),
                        col(f"{series.lower()}_ema{window}_init")
                    ).otherwise(
                        F.lag(series, 1).over(window_spec_lag) * lit(alpha) + 
                        F.lag(f"{series.lower()}_ema{window}", 1).over(window_spec_lag) * (1 - alpha)
                    )
                ).drop(f"{series.lower()}_ema{window}_init")
                
        return result_df
        
    except Exception as e:
        logger.error(f"Error building macro features: {e}")
        return df

def build_surprise_proxy(df: DataFrame) -> DataFrame:
    """Build surprise proxy at event_0h"""
    try:
        result_df = df
        
        if "event_0h_any" not in df.columns:
            logger.warning("Event columns not found, skipping surprise proxy")
            return df
            
        for series in ["cpiaucsl", "pcepi"]:
            yoy_col = f"{series}_yoy"
            
            if yoy_col not in df.columns:
                logger.warning(f"{yoy_col} column not found, skipping surprise proxy")
                continue
                
            window_spec_12m = Window.orderBy("ts").rowsBetween(-8640, 0)  # 360 days * 24 hours
            
            result_df = result_df.withColumn(
                f"{yoy_col}_mean12m",
                F.avg(yoy_col).over(window_spec_12m)
            ).withColumn(
                f"{yoy_col}_median12m",
                F.expr(f"percentile_approx({yoy_col}, 0.5, 10000)").over(window_spec_12m)
            )
            
            result_df = result_df.withColumn(
                f"{yoy_col}_dev12m_abs",
                F.when(
                    col("event_0h_any") == 1,
                    F.abs(col(yoy_col) - col(f"{yoy_col}_mean12m"))
                ).otherwise(None)
            ).withColumn(
                f"{yoy_col}_dev12m_median_abs",
                F.when(
                    col("event_0h_any") == 1,
                    F.abs(col(yoy_col) - col(f"{yoy_col}_median12m"))
                ).otherwise(None)
            )
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error building surprise proxy: {e}")
        return df

def clean_features(df: DataFrame) -> DataFrame:
    """Clean features: replace inf/NaN with null, drop rows with all nulls"""
    try:
        result_df = df
        
        for col_name in df.columns:
            if df.schema[col_name].dataType in [DoubleType()]:
                result_df = result_df.withColumn(
                    col_name,
                    F.when(
                        F.isnan(col_name) | F.isinf(col_name),
                        None
                    ).otherwise(col(col_name))
                )
                
        max_window = 8640  # 360 days * 24 hours (max lookback for YoY)
        total_rows = result_df.count()
        
        if total_rows > max_window:
            result_df = result_df.withColumn(
                "row_num", 
                F.row_number().over(Window.orderBy("ts"))
            ).filter(
                col("row_num") > max_window
            ).drop("row_num")
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error cleaning features: {e}")
        return df

def main():
    parser = argparse.ArgumentParser(description='Build features including rolling windows and surprise proxy')
    parser.add_argument('--horizon', default='1h', help='Time horizon (default: 1h)')
    parser.add_argument('--silver-path', default='delta/silver', help='Silver path (default: delta/silver)')
    
    args = parser.parse_args()
    
    features_config = load_config()
    
    windows_config = features_config.get('windows', {})
    ret_windows = windows_config.get('ret', [1, 3, 6, 12, 24])
    vol_windows = windows_config.get('vol', [6, 24, 72])
    ema_windows = windows_config.get('ema', [6, 24, 72])
    
    spark, is_delta = get_spark("features-build", force_parquet=True)
    fmt = "parquet"
    
    logger.info(f"Using storage format: {fmt}")
    
    try:
        labeled_path = f"{args.silver_path}/silver_eurusd_1h_labeled"
        
        try:
            labeled_df = read_table(spark, labeled_path, fmt)
            logger.info(f"Read {labeled_df.count()} records from silver_eurusd_1h_labeled")
        except Exception as e:
            logger.error(f"Error reading silver_eurusd_1h_labeled: {e}")
            return False
            
        features_df = labeled_df
        
        features_df = build_price_features(features_df, ret_windows, vol_windows, ema_windows, spark)
        logger.info("Built price features")
        
        features_df = build_tone_features(features_df)
        logger.info("Built tone features")
        
        features_df = build_attention_features(features_df)
        logger.info("Built attention features")
        
        features_df = build_macro_features(features_df)
        logger.info("Built macro features")
        
        features_df = build_surprise_proxy(features_df)
        logger.info("Built surprise proxy")
        
        features_df = clean_features(features_df)
        logger.info("Cleaned features")
        
        features_path = f"{args.silver_path}/silver_eurusd_1h_features"
        writer = features_df.write.mode("overwrite")
        write_table(writer, features_path, fmt)
        
        logger.info(f"Wrote {features_df.count()} records to silver_eurusd_1h_features")
        
        required_columns = [
            "open", "high", "low", "close",
            "tone_mean_24", "wiki_total_views",
            "CPIAUCSL", "PCEPI", "UNRATE", "FEDFUNDS", "DTWEXBGS",
            "g_event_total_count_1h", "event_day_any", "event_0h_any",
            "ret_1", "rv_6", "ema_6", "label"
        ]
        
        logger.info("Column presence:")
        for col_name in required_columns:
            present = col_name in features_df.columns
            logger.info(f"  {col_name}: {'Present' if present else 'Missing'}")
            
        label_counts = features_df.groupBy("label").count().collect()
        
        logger.info("Label distribution:")
        total_count = sum(row["count"] for row in label_counts)
        
        for row in label_counts:
            label = row["label"]
            count = row["count"]
            pct = (count / total_count) * 100 if total_count > 0 else 0
            
            label_name = "DOWN" if label == -1 else "FLAT" if label == 0 else "UP"
            logger.info(f"  {label_name} ({label}): {count} ({pct:.2f}%)")
            
        return True
        
    except Exception as e:
        logger.error(f"Error building features: {e}")
        return False
        
    finally:
        spark.stop()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
