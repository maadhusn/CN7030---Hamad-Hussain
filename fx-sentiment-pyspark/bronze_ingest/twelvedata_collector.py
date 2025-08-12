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

class TwelveDataCollector:
    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.rate_limit_delay = rate_limit_delay
        
    def collect_fx_hourly(self, symbol: str, start_date: str, end_date: str, 
                         output_dir: str) -> bool:
        """Collect hourly FX data from TwelveData"""
        try:
            
            params = {
                'symbol': symbol,
                'interval': '1h',
                'start_date': start_date,
                'end_date': end_date,
                'apikey': self.api_key,
                'format': 'JSON',
                'outputsize': 5000  # Max for free tier
            }
            
            url = f"{self.base_url}/time_series"
            logger.info(f"Fetching hourly FX data for {symbol}")
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'code' in data and data['code'] != 200:
                logger.error(f"API Error: {data.get('message', 'Unknown error')}")
                return False
                
            if 'status' in data and data['status'] == 'error':
                logger.error(f"API Error: {data.get('message', 'Unknown error')}")
                return False
                
            values = data.get('values', [])
            if not values:
                logger.error("No time series data found")
                return False
                
            df_data = []
            for item in values:
                try:
                    datetime_str = item['datetime']
                    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                    date_str = dt.strftime('%Y-%m-%d')
                    
                    if start_date <= date_str <= end_date:
                        row = {
                            'datetime': datetime_str,
                            'date': date_str,
                            'hour': dt.hour,
                            'symbol': symbol,
                            'open': float(item['open']),
                            'high': float(item['high']),
                            'low': float(item['low']),
                            'close': float(item['close']),
                            'volume': int(item.get('volume', 0)),
                            'source': 'twelvedata',
                            'collected_at': datetime.utcnow().isoformat()
                        }
                        df_data.append(row)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing data for {datetime_str}: {e}")
                    continue
                    
            if not df_data:
                logger.warning(f"No data found for {symbol} in date range")
                return False
                
            df = pd.DataFrame(df_data)
            
            for _, row in df.iterrows():
                date_obj = datetime.strptime(row['date'], '%Y-%m-%d')
                year = date_obj.year
                month = f"{date_obj.month:02d}"
                day = f"{date_obj.day:02d}"
                
                file_dir = Path(output_dir) / "twelvedata" / str(year) / month / day
                file_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = file_dir / f"fx_hourly_{symbol}_{row['date']}_h{row['hour']:02d}.csv"
                
                if not file_path.exists():
                    single_row_df = pd.DataFrame([row])
                    single_row_df.to_csv(file_path, index=False)
                    logger.info(f"Saved {file_path}")
                    
            time.sleep(self.rate_limit_delay)  # Rate limiting
            return True
            
        except Exception as e:
            logger.error(f"Error collecting TwelveData: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TwelveData FX Data Collector')
    parser.add_argument('--symbol', default='EUR/USD', help='FX symbol (default: EUR/USD)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='landing', help='Output directory')
    
    args = parser.parse_args()
    
    api_key = os.getenv('TWELVEDATA_API_KEY')
    if not api_key:
        logger.error("TWELVEDATA_API_KEY environment variable not set")
        return False
        
    collector = TwelveDataCollector(api_key)
    
    success = collector.collect_fx_hourly(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )
    
    return success

if __name__ == "__main__":
    main()
