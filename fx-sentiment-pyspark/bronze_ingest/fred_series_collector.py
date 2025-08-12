#!/usr/bin/env python3
"""
FRED Series Collector - Fetches macro economic series data
"""
import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDSeriesCollector:
    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.rate_limit_delay = rate_limit_delay
        
    def collect_series_data(self, series_ids: List[str], start_date: str, end_date: str, 
                           output_dir: str) -> bool:
        """Collect economic series data from FRED"""
        try:
            all_data = {}
            
            for series_id in series_ids:
                try:
                    params = {
                        'series_id': series_id,
                        'api_key': self.api_key,
                        'file_type': 'json',
                        'observation_start': start_date,
                        'observation_end': end_date
                    }
                    
                    url = f"{self.base_url}/series/observations"
                    logger.info(f"Fetching FRED data for series {series_id}")
                    
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if 'error_code' in data:
                        logger.error(f"API Error for {series_id}: {data.get('error_message', 'Unknown error')}")
                        continue
                        
                    observations = data.get('observations', [])
                    if not observations:
                        logger.warning(f"No observations found for {series_id}")
                        continue
                        
                    series_data = []
                    for obs in observations:
                        try:
                            date_str = obs.get('date', '')
                            value = obs.get('value', '')
                            
                            if not value or value == '.':
                                continue
                                
                            if start_date <= date_str <= end_date:
                                row = {
                                    'date': date_str,
                                    'series_id': series_id,
                                    'value': float(value) if value != '.' else None,
                                    'realtime_start': obs.get('realtime_start', ''),
                                    'realtime_end': obs.get('realtime_end', ''),
                                    'source': 'fred',
                                    'collected_at': datetime.utcnow().isoformat()
                                }
                                series_data.append(row)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error parsing observation for {series_id}: {e}")
                            continue
                            
                    if series_data:
                        all_data[series_id] = series_data
                        
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for series {series_id}: {e}")
                    continue
                    
            if not all_data:
                logger.warning("No FRED series data collected")
                return False
                
            # Create output directory
            output_base = Path(output_dir) / "fred_series"
            output_base.mkdir(parents=True, exist_ok=True)
            
            # Save each series to a separate Parquet file
            for series_id, series_data in all_data.items():
                try:
                    df = pd.DataFrame(series_data)
                    
                    # Create a unique filename for the series
                    file_path = output_base / f"{series_id}.parquet"
                    
                    # Save as Parquet if file doesn't exist
                    if not file_path.exists():
                        df.to_parquet(file_path, index=False)
                        logger.info(f"Saved {file_path} with {len(df)} observations")
                    else:
                        logger.info(f"File already exists, skipping: {file_path}")
                        
                except Exception as e:
                    logger.warning(f"Error saving series {series_id}: {e}")
                    continue
                    
            return True
            
        except Exception as e:
            logger.error(f"Error collecting FRED series data: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='FRED Economic Series Collector')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='landing', help='Output directory')
    parser.add_argument('--series', nargs='+', 
                       default=['CPIAUCSL', 'PCEPI', 'UNRATE', 'FEDFUNDS', 'DTWEXBGS'],
                       help='FRED series IDs to collect')
    
    args = parser.parse_args()
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        logger.warning("FRED_API_KEY environment variable not set, using demo mode")
        api_key = ''
        
    collector = FREDSeriesCollector(api_key)
    
    success = collector.collect_series_data(
        series_ids=args.series,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )
    
    return success

if __name__ == "__main__":
    main()
