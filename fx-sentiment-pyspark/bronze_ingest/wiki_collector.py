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

class WikipediaCollector:
    def __init__(self, rate_limit_delay: float = 1.0):
        self.base_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
        self.rate_limit_delay = rate_limit_delay
        
    def collect_pageviews(self, articles: List[str], start_date: str, end_date: str, 
                         output_dir: str, project: str = 'en.wikipedia') -> bool:
        """Collect Wikipedia pageview data"""
        try:
            all_data = []
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            for article in articles:
                try:
                    start_formatted = start_dt.strftime('%Y%m%d')
                    end_formatted = end_dt.strftime('%Y%m%d')
                    
                    url = f"{self.base_url}/per-article/{project}/all-access/user/{article}/daily/{start_formatted}/{end_formatted}"
                    
                    logger.info(f"Fetching pageviews for {article}")
                    
                    headers = {
                        'User-Agent': 'FX-Sentiment-Analysis/1.0 (https://github.com/maadhusn/CN7030---Hamad-Hussain)'
                    }
                    
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    items = data.get('items', [])
                    if not items:
                        logger.warning(f"No pageview data found for {article}")
                        continue
                        
                    for item in items:
                        try:
                            timestamp = item.get('timestamp', '')
                            if len(timestamp) >= 8:
                                date_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                                
                                if start_date <= date_str <= end_date:
                                    row = {
                                        'date': date_str,
                                        'article': item.get('article', article),
                                        'project': item.get('project', project),
                                        'access': item.get('access', 'all-access'),
                                        'agent': item.get('agent', 'user'),
                                        'views': int(item.get('views', 0)),
                                        'source': 'wikipedia',
                                        'collected_at': datetime.utcnow().isoformat()
                                    }
                                    all_data.append(row)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error parsing pageview data for {article}: {e}")
                            continue
                            
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Error collecting pageviews for {article}: {e}")
                    continue
                    
            if not all_data:
                logger.warning("No Wikipedia pageview data collected")
                return False
                
            df = pd.DataFrame(all_data)
            
            for _, row in df.iterrows():
                try:
                    date_obj = datetime.strptime(row['date'], '%Y-%m-%d')
                    year = date_obj.year
                    month = f"{date_obj.month:02d}"
                    day = f"{date_obj.day:02d}"
                    
                    file_dir = Path(output_dir) / "wikipedia" / str(year) / month / day
                    file_dir.mkdir(parents=True, exist_ok=True)
                    
                    article_clean = row['article'].replace(' ', '_').replace('/', '_')
                    file_path = file_dir / f"pageviews_{article_clean}_{row['date']}.csv"
                    
                    if not file_path.exists():
                        single_row_df = pd.DataFrame([row])
                        single_row_df.to_csv(file_path, index=False)
                        logger.info(f"Saved {file_path}")
                        
                except Exception as e:
                    logger.warning(f"Error saving row: {e}")
                    continue
                    
            return True
            
        except Exception as e:
            logger.error(f"Error collecting Wikipedia data: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Wikipedia Pageviews Data Collector')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='landing', help='Output directory')
    parser.add_argument('--articles', nargs='+', 
                       default=['Euro', 'United_States_dollar', 'European_Central_Bank', 
                               'Federal_Reserve', 'Brexit', 'European_Union'],
                       help='Wikipedia articles to collect pageviews for')
    parser.add_argument('--project', default='en.wikipedia', 
                       help='Wikipedia project (default: en.wikipedia)')
    
    args = parser.parse_args()
    
    collector = WikipediaCollector()
    
    success = collector.collect_pageviews(
        articles=args.articles,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        project=args.project
    )
    
    return success

if __name__ == "__main__":
    main()
