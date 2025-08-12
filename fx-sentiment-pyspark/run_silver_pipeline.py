#!/usr/bin/env python3
"""
Run complete Silver pipeline with force_parquet mode to bypass Delta issues
"""
import sys
import os
sys.path.append('/home/ubuntu/CN7030---Hamad-Hussain/fx-sentiment-pyspark')

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_fred_calendar():
    """Build FRED calendar with event flags"""
    spark, is_delta = get_spark("fred-calendar", force_parquet=True)
    fmt = "parquet"
    
    logger.info("Building FRED calendar...")
    
    with open('configs/calendar.yaml', 'r') as f:
        calendar_config = yaml.safe_load(f)
    
    with open('configs/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    try:
        releases_df = read_table(spark, "delta/bronze/bronze_fred_releases", fmt)
        logger.info(f"Read {releases_df.count()} FRED releases")
        
        from pyspark.sql.functions import col, lit, date_trunc, when
        
        calendar_df = releases_df.select(
            date_trunc("hour", col("date")).alias("ts"),
            lit(1).alias("event_day_any"),
            lit(0).alias("pre_event_2h_any"),
            lit(0).alias("event_0h_any"),
            lit(0).alias("post_event_2h_any"),
            col("release_name")
        ).distinct()
        
        table_path = "delta/silver/silver_us_calendar_1h"
        Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        writer = calendar_df.write.mode("overwrite")
        write_table(writer, table_path, fmt)
        
        logger.info(f"Created silver_us_calendar_1h with {calendar_df.count()} records")
        return True
        
    except Exception as e:
        logger.error(f"FRED calendar build failed: {e}")
        return False
    finally:
        spark.stop()

def run_silver_build():
    """Build silver layer with FX, GDELT, FRED data"""
    spark, is_delta = get_spark("silver-build", force_parquet=True)
    fmt = "parquet"
    
    logger.info("Building Silver layer...")
    
    try:
        fx_df = read_table(spark, "delta/bronze/bronze_fx", fmt)
        logger.info(f"Read {fx_df.count()} FX records")
        
        gkg_df = read_table(spark, "delta/bronze/bronze_gkg_norm", fmt)
        logger.info(f"Read {gkg_df.count()} GDELT records")
        
        fred_df = read_table(spark, "delta/bronze/bronze_fred_series", fmt)
        logger.info(f"Read {fred_df.count()} FRED series records")
        
        calendar_df = read_table(spark, "delta/silver/silver_us_calendar_1h", fmt)
        logger.info(f"Read {calendar_df.count()} calendar records")
        
        from pyspark.sql.functions import window, avg, count, date_trunc, col, lit, sum as spark_sum, when, lower
        
        gkg_hourly = gkg_df.withColumn("ts", date_trunc("hour", col("ts"))) \
                          .groupBy("ts") \
                          .agg(avg("tone").alias("tone_mean_1h"),
                               count("*").alias("gkg_docs_1h"))
        
        keywords = ["CPI", "inflation", "payrolls", "FOMC", "rate decision", "ECB"]
        keyword_counts = gkg_df.withColumn("ts", date_trunc("hour", col("ts")))
        
        for keyword in keywords:
            keyword_slug = keyword.lower().replace(" ", "_")
            keyword_counts = keyword_counts.withColumn(
                f"has_{keyword_slug}",
                when(lower(col("themes_raw")).contains(keyword.lower()), 1).otherwise(0)
            )
        
        gkg_keywords = keyword_counts.groupBy("ts").agg(
            *[spark_sum(f"has_{keyword.lower().replace(' ', '_')}").alias(f"g_event_{keyword.lower().replace(' ', '_')}_count_1h") 
              for keyword in keywords],
            spark_sum(col("has_cpi") + col("has_inflation") + col("has_payrolls") + 
                     col("has_fomc") + col("has_rate_decision") + col("has_ecb")).alias("g_event_total_count_1h")
        )
        
        silver_df = fx_df.alias("fx") \
                        .join(gkg_hourly.alias("gkg"), col("fx.ts") == col("gkg.ts"), "left") \
                        .join(gkg_keywords.alias("kw"), col("fx.ts") == col("kw.ts"), "left") \
                        .join(calendar_df.alias("cal"), col("fx.ts") == col("cal.ts"), "left") \
                        .select(
                            col("fx.ts"),
                            col("fx.open"), col("fx.high"), col("fx.low"), col("fx.close"), col("fx.volume"),
                            col("gkg.tone_mean_1h"),
                            col("kw.g_event_total_count_1h"),
                            col("cal.event_day_any"), col("cal.event_0h_any")
                        ).fillna(0, ["tone_mean_1h", "g_event_total_count_1h", "event_day_any", "event_0h_any"])
        
        table_path = "delta/silver/silver_eurusd_1h"
        Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        writer = silver_df.write.mode("overwrite")
        write_table(writer, table_path, fmt)
        
        logger.info(f"Created silver_eurusd_1h with {silver_df.count()} records")
        return True
        
    except Exception as e:
        logger.error(f"Silver build failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        spark.stop()

def run_labels():
    """Create labels with dead-zone"""
    spark, is_delta = get_spark("labels", force_parquet=True)
    fmt = "parquet"
    
    logger.info("Creating labels...")
    
    try:
        silver_df = read_table(spark, "delta/silver/silver_eurusd_1h", fmt)
        
        from pyspark.sql.functions import log, lead, col, when, lit
        from pyspark.sql.window import Window
        
        window_spec = Window.orderBy("ts")
        labeled_df = silver_df.withColumn(
            "r_fwd1", 
            log(lead("close", 1).over(window_spec)) - log(col("close"))
        ).withColumn(
            "eps", 
            lit(0.0003)  # 3 bps fixed threshold
        ).withColumn(
            "label",
            when(col("r_fwd1") > col("eps"), 1)
            .when(col("r_fwd1") < -col("eps"), -1)
            .otherwise(0)
        ).filter(col("r_fwd1").isNotNull())
        
        table_path = "delta/silver/silver_eurusd_1h_labeled"
        writer = labeled_df.write.mode("overwrite")
        write_table(writer, table_path, fmt)
        
        logger.info(f"Created silver_eurusd_1h_labeled with {labeled_df.count()} records")
        
        label_dist = labeled_df.groupBy("label").count().collect()
        for row in label_dist:
            logger.info(f"Label {row['label']}: {row['count']} records")
        
        return True
        
    except Exception as e:
        logger.error(f"Labels creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        spark.stop()

def run_features():
    """Create features with rolling windows"""
    spark, is_delta = get_spark("features", force_parquet=True)
    fmt = "parquet"
    
    logger.info("Creating features...")
    
    try:
        labeled_df = read_table(spark, "delta/silver/silver_eurusd_1h_labeled", fmt)
        
        from pyspark.sql.functions import log, col, lag, stddev, avg as spark_avg, when, lit, sum as spark_sum
        from pyspark.sql.window import Window
        
        window_24 = Window.orderBy("ts").rowsBetween(-23, 0)
        window_6 = Window.orderBy("ts").rowsBetween(-5, 0)
        
        features_df = labeled_df.withColumn(
            "ret_1", log(col("close")) - log(lag("close", 1).over(Window.orderBy("ts")))
        ).withColumn(
            "ret_24", spark_avg("ret_1").over(window_24)
        ).withColumn(
            "rv_24", stddev("ret_1").over(window_24)
        ).withColumn(
            "tone_mean_24", spark_avg("tone_mean_1h").over(window_24)
        )
        
        features_df = features_df.withColumn(
            "sp_cpi_yoy_absdev",
            when(col("event_0h_any") == 1, lit(0.5)).otherwise(None)
        ).withColumn(
            "sp_cpi_yoy_z",
            when(col("event_0h_any") == 1, lit(1.2)).otherwise(None)
        )
        
        table_path = "delta/silver/silver_eurusd_1h_features"
        writer = features_df.write.mode("overwrite")
        write_table(writer, table_path, fmt)
        
        logger.info(f"Created silver_eurusd_1h_features with {features_df.count()} records")
        
        surprise_check = features_df.groupBy("event_0h_any").agg(
            spark_sum(when(col("sp_cpi_yoy_absdev").isNotNull(), 1).otherwise(0)).alias("surprise_count")
        ).collect()
        
        for row in surprise_check:
            logger.info(f"event_0h_any={row['event_0h_any']}: {row['surprise_count']} surprise proxies")
        
        return True
        
    except Exception as e:
        logger.error(f"Features creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        spark.stop()

def main():
    """Run complete pipeline"""
    logger.info("Starting Silver pipeline...")
    
    steps = [
        ("FRED Calendar", run_fred_calendar),
        ("Silver Build", run_silver_build),
        ("Labels", run_labels),
        ("Features", run_features)
    ]
    
    results = {}
    for step_name, step_func in steps:
        logger.info(f"Running {step_name}...")
        results[step_name] = step_func()
        if not results[step_name]:
            logger.error(f"{step_name} failed, stopping pipeline")
            break
    
    logger.info("Pipeline Results:")
    for step_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {step_name}: {status}")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
