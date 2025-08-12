#!/usr/bin/env python3
"""
Spark job to load raw data from landing zone into Delta Bronze tables
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp, input_file_name
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BronzeDeltaLoader:
    def __init__(self, spark: SparkSession, landing_path: str, delta_path: str, fmt: str = "parquet"):
        self.spark = spark
        self.landing_path = landing_path
        self.delta_path = delta_path
        self.fmt = fmt
        
    def create_bronze_fx_table(self, mode: str = 'batch') -> bool:
        """Create bronze_fx table from Alpha Vantage and TwelveData"""
        try:
            logger.info("Processing FX data for bronze_fx table...")
            
            alpha_path = f"{self.landing_path}/alpha_vantage/*/*/*/*"
            twelvedata_path = f"{self.landing_path}/twelvedata/*/*/*/*"
            
            dfs_to_union = []
            
            try:
                if self._path_exists(f"{self.landing_path}/alpha_vantage"):
                    alpha_df = self.spark.read.option("header", "true").csv(alpha_path)
                    if alpha_df.count() > 0:
                        alpha_df = alpha_df.select(
                            col("date").cast("date"),
                            col("symbol"),
                            col("open").cast("double"),
                            col("high").cast("double"),
                            col("low").cast("double"),
                            col("close").cast("double"),
                            lit("alpha_vantage").alias("_source"),
                            current_timestamp().alias("_ingest_ts"),
                            input_file_name().alias("_source_file")
                        )
                        dfs_to_union.append(alpha_df)
                        logger.info(f"Alpha Vantage: {alpha_df.count()} records")
            except Exception as e:
                logger.warning(f"Error processing Alpha Vantage data: {e}")
                
            try:
                if self._path_exists(f"{self.landing_path}/twelvedata"):
                    twelve_df = self.spark.read.option("header", "true").csv(twelvedata_path)
                    if twelve_df.count() > 0:
                        twelve_df = twelve_df.select(
                            col("date").cast("date"),
                            col("symbol"),
                            col("open").cast("double"),
                            col("high").cast("double"),
                            col("low").cast("double"),
                            col("close").cast("double"),
                            lit("twelvedata").alias("_source"),
                            current_timestamp().alias("_ingest_ts"),
                            input_file_name().alias("_source_file")
                        )
                        dfs_to_union.append(twelve_df)
                        logger.info(f"TwelveData: {twelve_df.count()} records")
            except Exception as e:
                logger.warning(f"Error processing TwelveData data: {e}")
                
            if not dfs_to_union:
                logger.warning("No FX data found to process")
                return False
                
            fx_df = dfs_to_union[0]
            for df in dfs_to_union[1:]:
                fx_df = fx_df.union(df)
                
            table_path = f"{self.delta_path}/bronze/bronze_fx"
            
            if mode == 'batch':
                writer = fx_df.write.mode("overwrite")
                write_table(writer, table_path, self.fmt)
                logger.info(f"Wrote {fx_df.count()} records to bronze_fx table")
            else:
                logger.info("Streaming mode not yet implemented for FX data")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating bronze_fx table: {e}")
            return False
            
    def create_bronze_gkg_table(self, mode: str = 'batch') -> bool:
        """Create bronze_gkg table from GDELT data"""
        try:
            logger.info("Processing GDELT GKG data for bronze_gkg table...")
            
            gkg_path = f"{self.landing_path}/gdelt/*/*/*/*"
            
            if not self._path_exists(f"{self.landing_path}/gdelt"):
                logger.warning("No GDELT data found")
                return False
                
            gkg_df = self.spark.read.option("header", "true").csv(gkg_path)
            
            if gkg_df.count() == 0:
                logger.warning("No GDELT records found")
                return False
                
            gkg_df = gkg_df.withColumn("_source", lit("gdelt_gkg")) \
                          .withColumn("_ingest_ts", current_timestamp()) \
                          .withColumn("_source_file", input_file_name())
                          
            table_path = f"{self.delta_path}/bronze/bronze_gkg"
            
            if mode == 'batch':
                writer = gkg_df.write.mode("overwrite")
                write_table(writer, table_path, self.fmt)
                logger.info(f"Wrote {gkg_df.count()} records to bronze_gkg table")
            else:
                logger.info("Streaming mode not yet implemented for GKG data")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating bronze_gkg table: {e}")
            return False
            
    def create_bronze_econ_table(self, mode: str = 'batch') -> bool:
        """Create bronze_econ table from Trading Economics data"""
        try:
            logger.info("Processing Trading Economics data for bronze_econ table...")
            
            econ_path = f"{self.landing_path}/tradingeconomics/*/*/*/*"
            
            if not self._path_exists(f"{self.landing_path}/tradingeconomics"):
                logger.warning("No Trading Economics data found")
                return False
                
            econ_df = self.spark.read.option("header", "true").csv(econ_path)
            
            if econ_df.count() == 0:
                logger.warning("No Trading Economics records found")
                return False
                
            econ_df = econ_df.withColumn("_source", lit("tradingeconomics")) \
                            .withColumn("_ingest_ts", current_timestamp()) \
                            .withColumn("_source_file", input_file_name())
                            
            table_path = f"{self.delta_path}/bronze/bronze_econ"
            
            if mode == 'batch':
                writer = econ_df.write.mode("overwrite")
                write_table(writer, table_path, self.fmt)
                logger.info(f"Wrote {econ_df.count()} records to bronze_econ table")
            else:
                logger.info("Streaming mode not yet implemented for Economics data")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating bronze_econ table: {e}")
            return False
            
    def create_bronze_fred_table(self, mode: str = 'batch') -> bool:
        """Create bronze_fred table from FRED data"""
        try:
            logger.info("Processing FRED data for bronze_fred table...")
            
            fred_path = f"{self.landing_path}/fred/*/*/*/*"
            
            if not self._path_exists(f"{self.landing_path}/fred"):
                logger.warning("No FRED data found")
                return False
                
            fred_df = self.spark.read.option("header", "true").csv(fred_path)
            
            if fred_df.count() == 0:
                logger.warning("No FRED records found")
                return False
                
            fred_df = fred_df.withColumn("_source", lit("fred")) \
                            .withColumn("_ingest_ts", current_timestamp()) \
                            .withColumn("_source_file", input_file_name())
                            
            table_path = f"{self.delta_path}/bronze/bronze_fred"
            
            if mode == 'batch':
                writer = fred_df.write.mode("overwrite")
                write_table(writer, table_path, self.fmt)
                logger.info(f"Wrote {fred_df.count()} records to bronze_fred table")
            else:
                logger.info("Streaming mode not yet implemented for FRED data")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating bronze_fred table: {e}")
            return False
            
    def create_bronze_wiki_table(self, mode: str = 'batch') -> bool:
        """Create bronze_wiki table from Wikipedia data"""
        try:
            logger.info("Processing Wikipedia data for bronze_wiki table...")
            
            wiki_path = f"{self.landing_path}/wikipedia/*/*/*/*"
            
            if not self._path_exists(f"{self.landing_path}/wikipedia"):
                logger.warning("No Wikipedia data found")
                return False
                
            wiki_df = self.spark.read.option("header", "true").csv(wiki_path)
            
            if wiki_df.count() == 0:
                logger.warning("No Wikipedia records found")
                return False
                
            wiki_df = wiki_df.withColumn("_source", lit("wikipedia")) \
                            .withColumn("_ingest_ts", current_timestamp()) \
                            .withColumn("_source_file", input_file_name())
                            
            table_path = f"{self.delta_path}/bronze/bronze_wiki"
            
            if mode == 'batch':
                writer = wiki_df.write.mode("overwrite")
                write_table(writer, table_path, self.fmt)
                logger.info(f"Wrote {wiki_df.count()} records to bronze_wiki table")
            else:
                logger.info("Streaming mode not yet implemented for Wikipedia data")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating bronze_wiki table: {e}")
            return False

    def create_bronze_fred_releases_table(self, mode: str = 'batch') -> bool:
        """Create bronze_fred_releases table from FRED releases data"""
        try:
            logger.info("Processing FRED releases data for bronze_fred_releases table...")
            
            fred_releases_path = f"{self.landing_path}/fred_releases/*/*/*/*"
            
            if not self._path_exists(f"{self.landing_path}/fred_releases"):
                logger.warning("No FRED releases data found")
                return False
                
            fred_releases_df = self.spark.read.option("header", "true").csv(fred_releases_path)
            
            if fred_releases_df.count() == 0:
                logger.warning("No FRED releases records found")
                return False
                
            fred_releases_df = fred_releases_df.withColumn("_source", lit("fred_releases")) \
                                             .withColumn("_ingest_ts", current_timestamp()) \
                                             .withColumn("_source_file", input_file_name())
                                             
            table_path = f"{self.delta_path}/bronze/bronze_fred_releases"
            
            if mode == 'batch':
                writer = fred_releases_df.write.mode("overwrite")
                write_table(writer, table_path, self.fmt)
                logger.info(f"Wrote {fred_releases_df.count()} records to bronze_fred_releases table")
            else:
                logger.info("Streaming mode not yet implemented for FRED releases data")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating bronze_fred_releases table: {e}")
            return False

    def create_bronze_fred_series_table(self, mode: str = 'batch') -> bool:
        """Create bronze_fred_series table from FRED series data"""
        try:
            logger.info("Processing FRED series data for bronze_fred_series table...")
            
            fred_series_path = f"{self.landing_path}/fred_series/*/*/*/*"
            
            if not self._path_exists(f"{self.landing_path}/fred_series"):
                logger.warning("No FRED series data found")
                return False
                
            fred_series_df = self.spark.read.option("header", "true").csv(fred_series_path)
            
            if fred_series_df.count() == 0:
                logger.warning("No FRED series records found")
                return False
                
            fred_series_df = fred_series_df.withColumn("_source", lit("fred_series")) \
                                         .withColumn("_ingest_ts", current_timestamp()) \
                                         .withColumn("_source_file", input_file_name())
                                         
            table_path = f"{self.delta_path}/bronze/bronze_fred_series"
            
            if mode == 'batch':
                writer = fred_series_df.write.mode("overwrite")
                write_table(writer, table_path, self.fmt)
                logger.info(f"Wrote {fred_series_df.count()} records to bronze_fred_series table")
            else:
                logger.info("Streaming mode not yet implemented for FRED series data")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating bronze_fred_series table: {e}")
            return False
            
    def _path_exists(self, path: str) -> bool:
        """Check if path exists in the file system"""
        try:
            return Path(path).exists()
        except Exception:
            return False
            
    def run_all_bronze_tables(self, mode: str = 'batch') -> dict:
        """Run all bronze table creation jobs"""
        results = {}
        
        logger.info(f"Starting bronze table creation in {mode} mode...")
        
        results['bronze_fx'] = self.create_bronze_fx_table(mode)
        results['bronze_gkg'] = self.create_bronze_gkg_table(mode)
        results['bronze_econ'] = self.create_bronze_econ_table(mode)
        results['bronze_fred'] = self.create_bronze_fred_table(mode)
        results['bronze_fred_releases'] = self.create_bronze_fred_releases_table(mode)
        results['bronze_fred_series'] = self.create_bronze_fred_series_table(mode)
        results['bronze_wiki'] = self.create_bronze_wiki_table(mode)
        
        successful = sum(results.values())
        total = len(results)
        
        logger.info("Bronze table creation results:")
        for table, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {table}: {status}")
            
        logger.info(f"Successfully created {successful}/{total} bronze tables")
        
        return results

def create_spark_session(app_name: str = "BronzeDeltaLoader") -> tuple:
    """Create Spark session with Delta-first approach"""
    return get_spark(app_name)

def main():
    parser = argparse.ArgumentParser(description='Load raw data into Delta Bronze tables')
    parser.add_argument('--mode', choices=['batch', 'stream'], default='batch',
                       help='Processing mode (default: batch)')
    parser.add_argument('--landing-path', default='landing',
                       help='Landing zone path (default: landing)')
    parser.add_argument('--delta-path', default='delta',
                       help='Delta lake path (default: delta)')
    
    args = parser.parse_args()
    
    spark, is_delta = create_spark_session()
    fmt = "delta" if is_delta else "parquet"
    
    logger.info(f"Using storage format: {fmt}")
    
    try:
        loader = BronzeDeltaLoader(
            spark=spark,
            landing_path=args.landing_path,
            delta_path=args.delta_path,
            fmt=fmt
        )
        
        results = loader.run_all_bronze_tables(mode=args.mode)
        
        success = any(results.values())
        
        if success:
            logger.info("Bronze table creation completed successfully")
        else:
            logger.error("All bronze table creation jobs failed")
            
        return success
        
    except Exception as e:
        logger.error(f"Error in bronze table creation: {e}")
        return False
        
    finally:
        spark.stop()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
