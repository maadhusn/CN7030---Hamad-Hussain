#!/usr/bin/env python3
"""
GDELT GKG utilities for normalizing data
"""
from pyspark.sql import functions as F

def normalize_gkg(df):
    """
    Normalize GDELT GKG data with proper timestamp and tone parsing
    """
    columns = df.columns
    if "V2DATE" in columns:
        ts_raw = F.col("V2DATE")
    else:
        ts_raw = F.col("date")
    
    if "V2Tone" in columns:
        tone_raw = F.col("V2Tone")
    else:
        tone_raw = F.col("tone")
        
    if "V2Themes" in columns:
        themes = F.col("V2Themes")
    else:
        themes = F.col("themes")
    
    return (df
        .withColumn("ts", F.to_timestamp(ts_raw.cast("string"), "yyyyMMddHHmmss"))
        .withColumn("tone", F.split(tone_raw.cast("string"), ",").getItem(0).cast("double"))  # Extract first value from comma-separated string
        .withColumn("themes_raw", themes.cast("string"))
        .withColumn("docid", F.col("DocumentIdentifier") if "DocumentIdentifier" in columns else F.col("document_id"))
        .select("ts", "tone", "themes_raw", "docid")
        .filter(F.col("ts").isNotNull())
    )
