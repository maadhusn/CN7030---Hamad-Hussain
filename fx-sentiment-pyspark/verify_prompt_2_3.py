#!/usr/bin/env python3
"""
Verification script for Prompt 2.3 acceptance criteria
"""
import sys
sys.path.append('/home/ubuntu/CN7030---Hamad-Hussain/fx-sentiment-pyspark')

from spark_utils.session import get_spark
from spark_utils.io import read_table
from pyspark.sql.functions import col, count, sum as spark_sum, min as spark_min, max as spark_max, when, isnan, isnull, lit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run all acceptance checks for Prompt 2.3"""
    spark, is_delta = get_spark("verify-prompt-2-3", force_parquet=True)
    fmt = "parquet"
    
    print(f"Storage mode: {'DELTA_OK' if is_delta else 'PARQUET_FALLBACK'}")
    print("=" * 60)
    
    print("1. BRONZE_FX TABLE VERIFICATION")
    print("-" * 40)
    
    try:
        fx_df = read_table(spark, "delta/bronze/bronze_fx", fmt)
        total_rows = fx_df.count()
        print(f"Total bronze_fx rows: {total_rows}")
        
        if total_rows > 0:
            ts_stats = fx_df.agg(
                spark_min("ts").alias("earliest"),
                spark_max("ts").alias("latest")
            ).collect()[0]
            print(f"Earliest timestamp: {ts_stats['earliest']}")
            print(f"Latest timestamp: {ts_stats['latest']}")
            
            source_counts = fx_df.groupBy("_source").count().collect()
            for row in source_counts:
                print(f"{row['_source']} rows: {row['count']}")
        else:
            print("ERROR: bronze_fx table is empty!")
            
    except Exception as e:
        print(f"ERROR reading bronze_fx: {e}")
    
    print()
    
    print("2. GDELT TONE PROCESSING VERIFICATION")
    print("-" * 40)
    
    try:
        gkg_norm_df = read_table(spark, "delta/bronze/bronze_gkg_norm", fmt)
        tone_count = gkg_norm_df.filter(col("tone").isNotNull()).count()
        print(f"bronze_gkg_norm records with valid tone: {tone_count}")
        
        silver_df = read_table(spark, "delta/silver/silver_eurusd_1h", fmt)
        tone_mean_count = silver_df.filter(col("tone_mean_1h").isNotNull()).count()
        print(f"silver_eurusd_1h records with tone_mean_1h: {tone_mean_count}")
        
        if tone_mean_count > 0:
            print("Sample tone_mean_1h values:")
            silver_df.select("ts", "tone_mean_1h").filter(col("tone_mean_1h").isNotNull()).show(5)
        
    except Exception as e:
        print(f"ERROR checking GDELT tone: {e}")
    
    print()
    
    print("3. SILVER FEATURES TABLE VERIFICATION")
    print("-" * 40)
    
    try:
        features_df = read_table(spark, "delta/silver/silver_eurusd_1h_features", fmt)
        total_features = features_df.count()
        print(f"Total silver_eurusd_1h_features rows: {total_features}")
        
        print("\nColumn presence check:")
        columns = features_df.columns
        
        required_groups = {
            "FX": ["open", "high", "low", "close"],
            "Returns": ["ret_1", "ret_24"],
            "Volatility": ["rv_24"],
            "GDELT": ["tone_mean_1h", "tone_mean_24", "g_event_total_count_1h"],
            "Events": ["event_day_any", "event_0h_any"],
            "Surprise": ["sp_cpi_yoy_absdev", "sp_cpi_yoy_z"],
            "Labels": ["label", "r_fwd1"]
        }
        
        for group, cols in required_groups.items():
            present = [col for col in cols if col in columns]
            missing = [col for col in cols if col not in columns]
            print(f"  {group}: {len(present)}/{len(cols)} present")
            if missing:
                print(f"    Missing: {missing}")
        
        print(f"\nAll columns ({len(columns)}): {columns}")
        
    except Exception as e:
        print(f"ERROR checking features table: {e}")
    
    print()
    
    print("4. SURPRISE PROXY DISTRIBUTION CHECK")
    print("-" * 40)
    
    try:
        features_df = read_table(spark, "delta/silver/silver_eurusd_1h_features", fmt)
        
        surprise_check = features_df.groupBy("event_0h_any").agg(
            spark_sum(when(col("sp_cpi_yoy_absdev").isNotNull(), 1).otherwise(0)).alias("absdev_count"),
            spark_sum(when(col("sp_cpi_yoy_z").isNotNull(), 1).otherwise(0)).alias("z_count"),
            spark_sum(lit(1)).alias("total_rows")
        ).collect()
        
        print("Surprise proxy distribution by event_0h_any:")
        for row in surprise_check:
            event_flag = row['event_0h_any']
            absdev_count = row['absdev_count']
            z_count = row['z_count']
            total = row['total_rows']
            print(f"  event_0h_any={event_flag}: {absdev_count} absdev, {z_count} z-score (out of {total} rows)")
        
        leaks = features_df.filter(
            (col("event_0h_any") == 0) & 
            (col("sp_cpi_yoy_absdev").isNotNull())
        ).count()
        
        hits = features_df.filter(
            (col("event_0h_any") == 1) & 
            (col("sp_cpi_yoy_absdev").isNotNull())
        ).count()
        
        print(f"\nSurprise proxy leakage check:")
        print(f"  Hits (event_0h_any=1 with surprise): {hits}")
        print(f"  Leaks (event_0h_any=0 with surprise): {leaks}")
        
        if leaks == 0:
            print("  ✓ No leakage detected")
        else:
            print("  ✗ Leakage detected!")
        
    except Exception as e:
        print(f"ERROR checking surprise proxies: {e}")
    
    print()
    
    print("5. LABEL CLASS DISTRIBUTION")
    print("-" * 40)
    
    try:
        labeled_df = read_table(spark, "delta/silver/silver_eurusd_1h_labeled", fmt)
        
        label_dist = labeled_df.groupBy("label").count().orderBy("label").collect()
        total_labeled = sum(row['count'] for row in label_dist)
        
        print("Label distribution:")
        for row in label_dist:
            label = row['label']
            count = row['count']
            pct = (count / total_labeled) * 100 if total_labeled > 0 else 0
            label_name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(label, f"UNKNOWN({label})")
            print(f"  {label_name} ({label}): {count} ({pct:.1f}%)")
        
    except Exception as e:
        print(f"ERROR checking labels: {e}")
    
    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    
    spark.stop()

if __name__ == "__main__":
    main()
