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

class AlphaVantageCollector:
    def __init__(self, api_key: str, rate_limit_delay: float = 12.0):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = rate_limit_delay
        
    def collect_fx_daily(self, from_symbol: str, to_symbol: str, 
                        start_date: str, end_date: str, 
                        output_dir: str) -> bool:
        """Collect daily FX data from Alpha Vantage"""
        try:
            symbol = f"{from_symbol}{to_symbol}"
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            logger.info(f"Fetching FX data for {symbol}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"API Error: {data['Error Message']}")
                return False
                
            if 'Note' in data:
                logger.warning(f"API Note: {data['Note']}")
                time.sleep(60)  # Wait if rate limited
                return False
                
            time_series = data.get('Time Series (Daily)', {})
            if not time_series:
                logger.error("No time series data found")
                return False
                
            df_data = []
            for date_str, values in time_series.items():
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    if start_date <= date_str <= end_date:
                        row = {
                            'date': date_str,
                            'symbol': symbol,
                            'open': float(values['1. open']),
                            'high': float(values['2. high']),
                            'low': float(values['3. low']),
                            'close': float(values['4. close']),
                            'volume': None,
                            'source': 'alpha_vantage',
                            'collected_at': datetime.utcnow().isoformat()
                        }
                        df_data.append(row)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing data for {date_str}: {e}")
                    continue
                    
            if not df_data:
                logger.warning(f"No data found for {symbol} in date range")
                return False
                
            df = pd.DataFrame(df_data)
            
            for _, row in df.iterrows():
                date_str = row['date']
                file_dir = Path(output_dir) / "alpha_vantage" / "type=daily" / f"date={date_str}"
                file_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = file_dir / f"part-{int(time.time())}.parquet"
                
                if not any(file_dir.glob("*.parquet")):
                    single_row_df = pd.DataFrame([row])
                    single_row_df.to_parquet(file_path, index=False)
                    logger.info(f"Saved {file_path}")
                else:
                    logger.info(f"Parquet file already exists in: {file_dir}")
                    
            time.sleep(self.rate_limit_delay)  # Rate limiting
            return True
            
        except Exception as e:
            logger.error(f"Error collecting Alpha Vantage data: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Alpha Vantage FX Data Collector')
    parser.add_argument('--symbol', default='EURUSD', help='FX symbol (default: EURUSD)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='landing', help='Output directory')
    
    args = parser.parse_args()
    
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        logger.error("ALPHAVANTAGE_API_KEY environment variable not set")
        return False
        
    collector = AlphaVantageCollector(api_key)
    
    if len(args.symbol) == 6:
        from_symbol = args.symbol[:3]
        to_symbol = args.symbol[3:]
    else:
        logger.error("Symbol should be 6 characters (e.g., EURUSD)")
        return False
        
    success = collector.collect_fx_daily(
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )
    
    return success

if __name__ == "__main__":
    main()
