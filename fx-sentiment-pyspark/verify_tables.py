#!/usr/bin/env python3
"""
Script to verify bronze table counts and generate results for PR documentation
"""
from pyspark.sql import SparkSession

def main():
    print('=== FX Sentiment PySpark - Bronze Table Verification ===')
    print()

    spark = SparkSession.builder.appName('TableCounts').getOrCreate()

    tables = ['bronze_gkg', 'bronze_wiki']
    print('Bronze Table Counts:')
    print('-' * 40)

    for table in tables:
        try:
            df = spark.read.parquet(f'delta/bronze/{table}')
            count = df.count()
            print(f'{table}: {count} rows')
        except Exception as e:
            print(f'{table}: ERROR - {e}')

    print()
    print('Table Details:')
    print('-' * 40)

    for table in tables:
        try:
            df = spark.read.parquet(f'delta/bronze/{table}')
            print(f'\n{table.upper()}:')
            print(f'  Rows: {df.count()}')
            print(f'  Columns: {len(df.columns)}')
            print(f'  Schema: {df.columns}')
            print('  Sample data:')
            df.show(3, truncate=True)
        except Exception as e:
            print(f'{table}: ERROR - {e}')

    spark.stop()
    print('\n=== Verification Complete ===')

if __name__ == "__main__":
    main()
