import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEconomicsCollector:
    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.rate_limit_delay = rate_limit_delay
        
    def collect_calendar_data(self, start_date: str, end_date: str, 
                             output_dir: str, countries: list = None) -> bool:
        """Collect economic calendar data from Trading Economics"""
        try:
            if countries is None:
                countries = ['United States', 'Euro Area', 'United Kingdom', 'Japan']
                
            all_data = []
            
            for country in countries:
                try:
                    params = {
                        'c': self.api_key,
                        'country': country,
                        'initDate': start_date,
                        'endDate': end_date
                    }
                    
                    url = f"{self.base_url}/calendar"
                    logger.info(f"Fetching calendar data for {country}")
                    
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if isinstance(data, dict) and 'error' in data:
                        logger.error(f"API Error for {country}: {data['error']}")
                        continue
                        
                    if not isinstance(data, list):
                        logger.warning(f"Unexpected data format for {country}")
                        continue
                        
                    for event in data:
                        try:
                            event_date = event.get('Date', '')
                            if event_date:
                                dt = datetime.strptime(event_date[:10], '%Y-%m-%d')
                                date_str = dt.strftime('%Y-%m-%d')
                                
                                if start_date <= date_str <= end_date:
                                    row = {
                                        'date': date_str,
                                        'country': event.get('Country', country),
                                        'event': event.get('Event', ''),
                                        'category': event.get('Category', ''),
                                        'importance': event.get('Importance', ''),
                                        'actual': event.get('Actual', ''),
                                        'previous': event.get('Previous', ''),
                                        'forecast': event.get('Forecast', ''),
                                        'currency': event.get('Currency', ''),
                                        'unit': event.get('Unit', ''),
                                        'source': 'tradingeconomics',
                                        'collected_at': datetime.utcnow().isoformat()
                                    }
                                    all_data.append(row)
                        except Exception as e:
                            logger.warning(f"Error parsing event: {e}")
                            continue
                            
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {country}: {e}")
                    continue
                    
            if not all_data:
                logger.warning("No calendar data collected")
                return False
                
            df = pd.DataFrame(all_data)
            
            for _, row in df.iterrows():
                try:
                    date_obj = datetime.strptime(row['date'], '%Y-%m-%d')
                    year = date_obj.year
                    month = f"{date_obj.month:02d}"
                    day = f"{date_obj.day:02d}"
                    
                    file_dir = Path(output_dir) / "tradingeconomics" / str(year) / month / day
                    file_dir.mkdir(parents=True, exist_ok=True)
                    
                    country_clean = row['country'].replace(' ', '_').replace('/', '_')
                    file_path = file_dir / f"calendar_{country_clean}_{row['date']}.csv"
                    
                    if file_path.exists():
                        existing_df = pd.read_csv(file_path)
                        combined_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
                        combined_df.drop_duplicates().to_csv(file_path, index=False)
                    else:
                        single_row_df = pd.DataFrame([row])
                        single_row_df.to_csv(file_path, index=False)
                        logger.info(f"Saved {file_path}")
                        
                except Exception as e:
                    logger.warning(f"Error saving row: {e}")
                    continue
                    
            return True
            
        except Exception as e:
            logger.error(f"Error collecting Trading Economics data: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Trading Economics Calendar Data Collector')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='landing', help='Output directory')
    parser.add_argument('--countries', nargs='+', 
                       default=['United States', 'Euro Area', 'United Kingdom', 'Japan'],
                       help='Countries to collect data for')
    
    args = parser.parse_args()
    
    api_key = os.getenv('TRADINGECONOMICS_API_KEY')
    if not api_key:
        logger.error("TRADINGECONOMICS_API_KEY environment variable not set")
        return False
        
    collector = TradingEconomicsCollector(api_key)
    
    success = collector.collect_calendar_data(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        countries=args.countries
    )
    
    return success

if __name__ == "__main__":
    main()
