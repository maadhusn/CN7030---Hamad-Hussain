import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import zipfile
import io
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GDELTCollector:
    def __init__(self, rate_limit_delay: float = 1.0):
        self.base_url = "http://data.gdeltproject.org/gdeltv2"
        self.rate_limit_delay = rate_limit_delay
        
    def get_gkg_urls_for_date_range(self, start_date: str, end_date: str) -> list:
        """Generate GDELT GKG file URLs for date range"""
        urls = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_dt = start_dt
        while current_dt <= end_dt:
            for hour in range(24):
                for minute in [0, 15, 30, 45]:
                    timestamp = current_dt.strftime('%Y%m%d') + f"{hour:02d}{minute:02d}00"
                    url = f"{self.base_url}/{timestamp}.gkg.csv.zip"
                    urls.append((url, current_dt, hour, minute))
            current_dt += timedelta(days=1)
            
        return urls
        
    def collect_gkg_data(self, start_date: str, end_date: str, 
                        output_dir: str, max_files: int = 50) -> bool:
        """Collect GDELT GKG (Global Knowledge Graph) data"""
        try:
            urls = self.get_gkg_urls_for_date_range(start_date, end_date)
            
            urls = urls[:max_files]
            
            logger.info(f"Collecting {len(urls)} GDELT GKG files")
            
            collected_count = 0
            for url, date_obj, hour, minute in urls:
                try:
                    logger.info(f"Fetching {url}")
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 404:
                        logger.warning(f"File not found: {url}")
                        continue
                        
                    response.raise_for_status()
                    
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        csv_filename = zip_file.namelist()[0]
                        csv_content = zip_file.read(csv_filename)
                        
                    df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')), 
                                   sep='\t', header=None, low_memory=False)
                    
                    if df.empty:
                        continue
                        
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    
                    if len(df.columns) >= 15:
                        df_processed = pd.DataFrame({
                            'gkg_record_id': df['col_0'] if 'col_0' in df.columns else '',
                            'date': df['col_1'] if 'col_1' in df.columns else '',
                            'source_collection': df['col_2'] if 'col_2' in df.columns else '',
                            'source_name': df['col_3'] if 'col_3' in df.columns else '',
                            'document_id': df['col_4'] if 'col_4' in df.columns else '',
                            'counts': df['col_5'] if 'col_5' in df.columns else '',
                            'themes': df['col_7'] if 'col_7' in df.columns else '',
                            'locations': df['col_9'] if 'col_9' in df.columns else '',
                            'tone': df['col_15'] if 'col_15' in df.columns else '',
                            'hour': hour,
                            'minute': minute,
                            'source': 'gdelt_gkg',
                            'collected_at': datetime.utcnow().isoformat()
                        })
                    else:
                        df_processed = df.copy()
                        df_processed['hour'] = hour
                        df_processed['minute'] = minute
                        df_processed['source'] = 'gdelt_gkg'
                        df_processed['collected_at'] = datetime.utcnow().isoformat()
                    
                    year = date_obj.year
                    month = f"{date_obj.month:02d}"
                    day = f"{date_obj.day:02d}"
                    
                    file_dir = Path(output_dir) / "gdelt" / str(year) / month / day
                    file_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = f"{year}{month}{day}_{hour:02d}{minute:02d}"
                    file_path = file_dir / f"gkg_{timestamp}.csv"
                    
                    if not file_path.exists():
                        df_processed.to_csv(file_path, index=False)
                        logger.info(f"Saved {file_path} with {len(df_processed)} records")
                        collected_count += 1
                        
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Error processing {url}: {e}")
                    continue
                    
            logger.info(f"Successfully collected {collected_count} GDELT files")
            return collected_count > 0
            
        except Exception as e:
            logger.error(f"Error collecting GDELT data: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='GDELT GKG Data Collector')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='landing', help='Output directory')
    parser.add_argument('--max-files', type=int, default=50, help='Max files to collect')
    
    args = parser.parse_args()
    
    collector = GDELTCollector()
    
    success = collector.collect_gkg_data(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        max_files=args.max_files
    )
    
    return success

if __name__ == "__main__":
    main()
