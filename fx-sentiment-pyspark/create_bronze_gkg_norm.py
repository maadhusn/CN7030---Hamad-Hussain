#!/usr/bin/env python3
"""
Script to manually create bronze_gkg_norm table
"""
from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table
from utils.gkg import normalize_gkg
from pyspark.sql.functions import lit, current_timestamp

def main():
    spark, is_delta = get_spark("create-gkg-norm")
    fmt = "parquet"  # Force parquet since Delta has issues
    
    print(f"Storage mode: {'DELTA_OK' if is_delta else 'PARQUET_FALLBACK'}")
    
    try:
        gkg_df = read_table(spark, "delta/bronze/bronze_gkg", fmt)
        print(f"Read {gkg_df.count()} records from bronze_gkg")
        
        print("Columns:", gkg_df.columns)
        gkg_df.show(2, truncate=False)
        
        normalized_df = normalize_gkg(gkg_df)
        normalized_df = normalized_df.withColumn("_source", lit("gdelt_gkg_norm")) \
                                   .withColumn("_ingest_ts", current_timestamp())
        
        print(f"Normalized to {normalized_df.count()} records")
        normalized_df.show(5)
        
        table_path = "delta/bronze/bronze_gkg_norm"
        writer = normalized_df.write.mode("overwrite")
        write_table(writer, table_path, fmt)
        
        print(f"Successfully wrote bronze_gkg_norm table")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    spark.stop()

if __name__ == "__main__":
    main()
