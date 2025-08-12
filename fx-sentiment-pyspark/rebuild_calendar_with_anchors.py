#!/usr/bin/env python3
"""
Rebuild FRED calendar with anchor hours using approx_time mode
"""
import sys
sys.path.append('/home/ubuntu/CN7030---Hamad-Hussain/fx-sentiment-pyspark')

from spark_utils.session import get_spark
from spark_utils.io import write_table, read_table
import yaml
from pathlib import Path
import logging
from pyspark.sql.functions import col, lit, when, expr, hour, date_format, max as spark_max
from pyspark.sql.window import Window
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Rebuild FRED calendar with anchor hours"""
    
    with open('configs/calendar.yaml', 'r') as f:
        calendar_config = yaml.safe_load(f)['calendar']
    
    with open('configs/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Calendar mode: {calendar_config['mode']}")
    print(f"Approx times: {calendar_config['approx_times_utc']}")
    
    spark, is_delta = get_spark('rebuild-calendar', force_parquet=True)
    fmt = 'parquet'
    
    try:
        releases_df = read_table(spark, 'delta/bronze/bronze_fred_releases', fmt)
        print(f"Read {releases_df.count()} FRED releases")
        
        print("Sample FRED releases:")
        releases_df.show(5, truncate=False)
        
        start_ts = int(datetime.strptime('2019-01-01', '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime('2024-12-31', '%Y-%m-%d').timestamp() + 86400)
        
        timeline_df = spark.sql(f"""
        SELECT explode(sequence(
            to_timestamp({start_ts}), 
            to_timestamp({end_ts}), 
            interval 1 hour
        )) as ts
        """)
        
        print(f"Created timeline with {timeline_df.count()} hours")
        
        releases_df = releases_df.withColumn('date_utc', col('date').cast('timestamp'))
        
        approx_times = calendar_config['approx_times_utc']
        releases_with_hours = releases_df.withColumn(
            'anchor_hour',
            when(col('release_name') == 'Consumer Price Index', 13)
            .when(col('release_name') == 'Employment Situation', 13)  
            .when(col('release_name') == 'Gross Domestic Product', 12)
            .when(col('release_name') == 'Personal Income and Outlays', 12)
            .otherwise(13)
        ).withColumn(
            'event_ts',
            expr('date_utc + make_interval(0, 0, 0, 0, anchor_hour, 0, 0)')
        )
        
        print("Sample releases with anchor hours:")
        releases_with_hours.select('release_name', 'date', 'anchor_hour', 'event_ts').show(10, truncate=False)
        
        event_hours = releases_with_hours.select('event_ts').distinct()
        
        calendar_df = timeline_df.join(
            event_hours,
            timeline_df.ts == event_hours.event_ts,
            'left'
        ).withColumn(
            'event_0h_any',
            when(col('event_ts').isNotNull(), 1).otherwise(0)
        ).withColumn(
            'event_day_any', 
            when(col('event_ts').isNotNull(), 1).otherwise(0)
        ).withColumn(
            'pre_event_2h_any', lit(0)
        ).withColumn(
            'post_event_2h_any', lit(0)
        ).select('ts', 'event_day_any', 'pre_event_2h_any', 'event_0h_any', 'post_event_2h_any')
        
        calendar_df = calendar_df.fillna(0)
        
        event_0h_count = calendar_df.filter(col('event_0h_any') == 1).count()
        print(f"Hours with event_0h_any=1: {event_0h_count}")
        
        if event_0h_count > 0:
            print("Sample event_0h hours:")
            calendar_df.filter(col('event_0h_any') == 1).show(10)
        else:
            print("ERROR: No event_0h hours found!")
            
        table_path = 'delta/silver/silver_us_calendar_1h'
        Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        writer = calendar_df.write.mode('overwrite')
        write_table(writer, table_path, fmt)
        
        print(f"Wrote {calendar_df.count()} records to silver_us_calendar_1h")
        print(f"Event anchor hours: {event_0h_count}")
        
        return event_0h_count > 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        spark.stop()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
