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
    
    ts_parsed = F.when(
        ts_raw.cast("string").rlike("^[0-9]{14}$"),
        F.to_timestamp(ts_raw.cast("string"), "yyyyMMddHHmmss")
    ).when(
        ts_raw.cast("string").rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$"),
        F.to_timestamp(ts_raw.cast("string"), "yyyy-MM-dd HH:mm:ss")
    ).when(
        ts_raw.cast("string").rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2}$"),
        F.to_timestamp(ts_raw.cast("string"), "yyyy-MM-dd")
    ).otherwise(F.lit(None).cast("timestamp"))
    
    return (df
        .withColumn("ts", ts_parsed)
        .withColumn("tone", F.split(tone_raw.cast("string"), ",").getItem(0).cast("double"))
        .withColumn("themes_raw", themes.cast("string"))
        .withColumn("docid", F.col("DocumentIdentifier") if "DocumentIdentifier" in columns else F.col("document_id"))
        .select("ts", "tone", "themes_raw", "docid")
    )
