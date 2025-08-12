from typing import Tuple, List, Dict, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, isnan, isnull, count, when, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, LongType
import os

MAX_TRAIN_ROWS = int(os.getenv('MAX_TRAIN_ROWS', '200'))
MAX_VALID_ROWS = int(os.getenv('MAX_VALID_ROWS', '40'))
MAX_TEST_ROWS = int(os.getenv('MAX_TEST_ROWS', '40'))
PLOT_ROWS = int(os.getenv('PLOT_ROWS', '254'))

def build_spark(app_name: str = "eurusd-ml") -> SparkSession:
    from spark_utils.session import get_spark
    spark, _ = get_spark(app_name)
    return spark

def load_first_available(spark: SparkSession, paths: List[str]) -> DataFrame:
    from spark_utils.io import read_table
    
    for path in paths:
        try:
            df = read_table(spark, path, "parquet")
            print(f"Successfully loaded data from: {path}")
            return df
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
    
    raise FileNotFoundError(f"None of the paths could be loaded: {paths}")

def sanitize_schema(df: DataFrame) -> DataFrame:
    for field in df.schema.fields:
        if field.dataType in [IntegerType(), LongType()] and field.name != "label":
            df = df.withColumn(field.name, col(field.name).cast(DoubleType()))
    
    return df

def get_feature_columns(df: DataFrame,
                        label_col: str = "label",
                        drop_cols: Optional[List[str]] = None,
                        max_missing_frac: float = 0.20) -> List[str]:
    if drop_cols is None:
        drop_cols = ["ts", "label", "r_fwd1", "r_fwd3", "r_fwd6", "eps"]
    
    candidate_cols = [c for c in df.columns if c not in drop_cols]
    
    total_rows = df.count()
    feature_cols = []
    
    for col_name in candidate_cols:
        missing_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
        missing_frac = missing_count / total_rows if total_rows > 0 else 0
        
        if missing_frac > max_missing_frac:
            print(f"Dropping {col_name}: {missing_frac:.2%} missing")
            continue
            
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count <= 1:
            print(f"Dropping {col_name}: constant column")
            continue
            
        feature_cols.append(col_name)
    
    print(f"Selected {len(feature_cols)} feature columns: {feature_cols}")
    return feature_cols

def split_timewise(df: DataFrame,
                   ts_col: str = "ts",
                   splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
                   caps: Optional[Dict[str, int]] = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
    if caps is None:
        caps = {"train": MAX_TRAIN_ROWS, "valid": MAX_VALID_ROWS, "test": MAX_TEST_ROWS}
    
    df_sorted = df.orderBy(ts_col)
    total_rows = df_sorted.count()
    
    train_end = int(total_rows * splits[0])
    valid_end = int(total_rows * (splits[0] + splits[1]))
    
    window = Window.orderBy(ts_col)
    df_with_row_num = df_sorted.withColumn("row_num", row_number().over(window))
    
    train_df = df_with_row_num.filter(col("row_num") <= train_end).drop("row_num")
    valid_df = df_with_row_num.filter((col("row_num") > train_end) & (col("row_num") <= valid_end)).drop("row_num")
    test_df = df_with_row_num.filter(col("row_num") > valid_end).drop("row_num")
    
    if train_df.count() > caps["train"]:
        train_df = train_df.limit(caps["train"])
    if valid_df.count() > caps["valid"]:
        valid_df = valid_df.limit(caps["valid"])
    if test_df.count() > caps["test"]:
        test_df = test_df.limit(caps["test"])
    
    from pyspark.sql.functions import min as spark_min, max as spark_max
    
    train_max_ts = train_df.agg(spark_max(ts_col).alias("max_ts")).collect()[0]["max_ts"]
    valid_min_ts = valid_df.agg(spark_min(ts_col).alias("min_ts")).collect()[0]["min_ts"]
    valid_max_ts = valid_df.agg(spark_max(ts_col).alias("max_ts")).collect()[0]["max_ts"]
    test_min_ts = test_df.agg(spark_min(ts_col).alias("min_ts")).collect()[0]["min_ts"]
    
    assert train_max_ts < valid_min_ts, f"Chronological order violated: train_max({train_max_ts}) >= valid_min({valid_min_ts})"
    assert valid_max_ts < test_min_ts, f"Chronological order violated: valid_max({valid_max_ts}) >= test_min({test_min_ts})"
    
    return train_df, valid_df, test_df

def print_split_summary(train_df: DataFrame, valid_df: DataFrame, test_df: DataFrame,
                        ts_col: str = "ts") -> None:
    from pyspark.sql.functions import min as spark_min, max as spark_max
    
    for name, df in [("TRAIN", train_df), ("VALID", valid_df), ("TEST", test_df)]:
        count = df.count()
        if count > 0:
            ts_stats = df.agg(spark_min(ts_col).alias("min_ts"), spark_max(ts_col).alias("max_ts")).collect()[0]
            min_ts = ts_stats["min_ts"]
            max_ts = ts_stats["max_ts"]
            print(f"{name}: {count} rows, {min_ts} to {max_ts}")
        else:
            print(f"{name}: {count} rows (empty)")
