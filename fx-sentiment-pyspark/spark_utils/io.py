def write_table(writer, path, fmt):
    """
    Write a DataFrame to a table using the specified format
    
    Args:
        writer: DataFrame writer (df.write.mode(...))
        path: Path to write the table to
        fmt: Format to use (delta or parquet)
    """
    writer.format(fmt).save(path)

def read_table(spark, path, fmt):
    """
    Read a table using the specified format
    
    Args:
        spark: SparkSession
        path: Path to read the table from
        fmt: Format to use (delta or parquet)
        
    Returns:
        DataFrame: The loaded table
    """
    return spark.read.format(fmt).load(path)
