#!/usr/bin/env python3
"""
FRED Releases Collector - Fetches release dates for economic indicators
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
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDReleasesCollector:
    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.rate_limit_delay = rate_limit_delay
        
    def get_release_list(self) -> List[Dict[str, Any]]:
        """Get list of all FRED releases"""
        try:
            params = {
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 1000,  # Maximum allowed
                'order_by': 'release_id',
                'sort_order': 'asc'
            }
            
            url = f"{self.base_url}/releases"
            logger.info("Fetching FRED releases list")
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error_code' in data:
                logger.error(f"API Error: {data.get('error_message', 'Unknown error')}")
                return []
                
            releases = data.get('releases', [])
            if not releases:
                logger.warning("No releases found")
                return []
                
            logger.info(f"Found {len(releases)} FRED releases")
            return releases
            
        except Exception as e:
            logger.error(f"Error fetching FRED releases: {e}")
            return []
    
    def get_release_dates(self, release_id: int, release_name: str, 
                         start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get release dates for a specific release"""
        try:
            params = {
                'release_id': release_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'include_release_dates_with_no_data': 'true',
                'sort_order': 'asc',
                'observation_start': start_date,
                'observation_end': end_date
            }
            
            url = f"{self.base_url}/release/dates"
            logger.info(f"Fetching dates for release: {release_name} (ID: {release_id})")
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error_code' in data:
                logger.error(f"API Error for {release_name}: {data.get('error_message', 'Unknown error')}")
                return []
                
            release_dates = data.get('release_dates', [])
            if not release_dates:
                logger.warning(f"No release dates found for {release_name}")
                return []
                
            # Add release name to each date entry
            for date_entry in release_dates:
                date_entry['release_name'] = release_name
                date_entry['country'] = 'US'  # Hardcoded for FRED
                
            logger.info(f"Found {len(release_dates)} dates for {release_name}")
            return release_dates
            
        except Exception as e:
            logger.error(f"Error fetching release dates for {release_name}: {e}")
            return []
    
    def collect_release_dates(self, release_names: List[str], start_date: str, end_date: str, 
                             output_dir: str) -> bool:
        """Collect release dates for specified releases"""
        try:
            # Get all releases first
            all_releases = self.get_release_list()
            if not all_releases:
                return False
                
            # Filter releases by name
            filtered_releases = [r for r in all_releases if r.get('name') in release_names]
            
            if not filtered_releases:
                logger.warning(f"None of the specified releases found: {release_names}")
                return False
                
            logger.info(f"Found {len(filtered_releases)} matching releases")
            
            all_dates = []
            
            for release in filtered_releases:
                release_id = release.get('id')
                release_name = release.get('name')
                
                dates = self.get_release_dates(
                    release_id=release_id,
                    release_name=release_name,
                    start_date=start_date,
                    end_date=end_date
                )
                
                all_dates.extend(dates)
                time.sleep(self.rate_limit_delay)
                
            if not all_dates:
                logger.warning("No release dates collected")
                return False
                
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(all_dates)
            
            # Ensure output directory exists
            output_base = Path(output_dir) / "fred_releases"
            output_base.mkdir(parents=True, exist_ok=True)
            
            # Group by year and month for file organization
            for (year, month), group_df in df.groupby([
                df['date'].str[:4],  # Year
                df['date'].str[5:7]  # Month
            ]):
                # Create directory structure
                file_dir = output_base / year / month
                file_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a unique filename
                file_path = file_dir / f"fred_releases_{year}_{month}.parquet"
                
                # Save as Parquet
                if not file_path.exists():
                    group_df.to_parquet(file_path, index=False)
                    logger.info(f"Saved {file_path}")
                else:
                    logger.info(f"File already exists, skipping: {file_path}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error collecting FRED release dates: {e}")
            return False

def load_release_names_from_config() -> List[str]:
    """Load release names from calendar.yaml config"""
    try:
        config_path = Path(__file__).parent.parent / "configs" / "calendar.yaml"
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return []
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        release_names = config.get('calendar', {}).get('fred_release_names', [])
        return release_names
        
    except Exception as e:
        logger.error(f"Error loading release names from config: {e}")
        return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description='FRED Releases Collector')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='landing', help='Output directory')
    parser.add_argument('--releases', nargs='+', help='FRED release names to collect (overrides config)')
    
    args = parser.parse_args()
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        logger.warning("FRED_API_KEY environment variable not set, using demo mode")
        api_key = ''
        
    collector = FREDReleasesCollector(api_key)
    
    # Use provided release names or load from config
    release_names = args.releases if args.releases else load_release_names_from_config()
    
    if not release_names:
        logger.error("No release names specified and none found in config")
        return False
        
    logger.info(f"Collecting dates for releases: {release_names}")
    
    success = collector.collect_release_dates(
        release_names=release_names,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )
    
    return success

if __name__ == "__main__":
    main()
