#!/usr/bin/env python3
"""
Orchestration script for running data collection backfill
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha_vantage_collector import AlphaVantageCollector
from twelvedata_collector import TwelveDataCollector
from gdelt_collector import GDELTCollector
from tradingecon_collector import TradingEconomicsCollector
from fred_collector import FREDCollector
from wiki_collector import WikipediaCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_date_format(date_string: str) -> bool:
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def run_alpha_vantage(start_date: str, end_date: str, output_dir: str) -> bool:
    """Run Alpha Vantage collector"""
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        logger.warning("ALPHAVANTAGE_API_KEY not set, skipping Alpha Vantage collection")
        return False
        
    logger.info("Starting Alpha Vantage collection...")
    collector = AlphaVantageCollector(api_key)
    
    success = collector.collect_fx_daily(
        from_symbol='EUR',
        to_symbol='USD',
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    if success:
        logger.info("Alpha Vantage collection completed successfully")
    else:
        logger.error("Alpha Vantage collection failed")
        
    return success

def run_twelvedata(start_date: str, end_date: str, output_dir: str) -> bool:
    """Run TwelveData collector"""
    api_key = os.getenv('TWELVEDATA_API_KEY')
    if not api_key:
        logger.warning("TWELVEDATA_API_KEY not set, skipping TwelveData collection")
        return False
        
    logger.info("Starting TwelveData collection...")
    collector = TwelveDataCollector(api_key)
    
    success = collector.collect_fx_hourly(
        symbol='EUR/USD',
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    if success:
        logger.info("TwelveData collection completed successfully")
    else:
        logger.error("TwelveData collection failed")
        
    return success

def run_gdelt(start_date: str, end_date: str, output_dir: str) -> bool:
    """Run GDELT collector"""
    logger.info("Starting GDELT GKG collection...")
    collector = GDELTCollector()
    
    success = collector.collect_gkg_data(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        max_files=20
    )
    
    if success:
        logger.info("GDELT collection completed successfully")
    else:
        logger.error("GDELT collection failed")
        
    return success

def run_tradingeconomics(start_date: str, end_date: str, output_dir: str) -> bool:
    """Run Trading Economics collector"""
    api_key = os.getenv('TRADINGECONOMICS_API_KEY')
    if not api_key:
        logger.warning("TRADINGECONOMICS_API_KEY not set, skipping Trading Economics collection")
        return False
        
    logger.info("Starting Trading Economics collection...")
    collector = TradingEconomicsCollector(api_key)
    
    success = collector.collect_calendar_data(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        countries=['United States', 'Euro Area']
    )
    
    if success:
        logger.info("Trading Economics collection completed successfully")
    else:
        logger.error("Trading Economics collection failed")
        
    return success

def run_fred(start_date: str, end_date: str, output_dir: str) -> bool:
    """Run FRED collector"""
    api_key = os.getenv('FRED_API_KEY')
    logger.info("Starting FRED collection...")
    
    collector = FREDCollector(api_key or '')
    
    series_ids = ['DGS10', 'DEXUSEU', 'UNRATE', 'CPIAUCSL']
    
    success = collector.collect_series_data(
        series_ids=series_ids,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    if success:
        logger.info("FRED collection completed successfully")
    else:
        logger.error("FRED collection failed")
        
    return success

def run_wikipedia(start_date: str, end_date: str, output_dir: str) -> bool:
    """Run Wikipedia collector"""
    logger.info("Starting Wikipedia pageviews collection...")
    collector = WikipediaCollector()
    
    articles = ['Euro', 'United_States_dollar', 'European_Central_Bank', 'Federal_Reserve']
    
    success = collector.collect_pageviews(
        articles=articles,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    if success:
        logger.info("Wikipedia collection completed successfully")
    else:
        logger.error("Wikipedia collection failed")
        
    return success

def main():
    parser = argparse.ArgumentParser(description='Run data collection backfill')
    
    parser.add_argument('--alpha', action='store_true', help='Collect Alpha Vantage data')
    parser.add_argument('--twelve', action='store_true', help='Collect TwelveData data')
    parser.add_argument('--gkg', action='store_true', help='Collect GDELT GKG data')
    parser.add_argument('--econ', action='store_true', help='Collect Trading Economics data')
    parser.add_argument('--fred', action='store_true', help='Collect FRED data')
    parser.add_argument('--wiki', action='store_true', help='Collect Wikipedia data')
    parser.add_argument('--all', action='store_true', help='Collect all sources')
    
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    
    parser.add_argument('--output', default='landing', help='Output directory (default: landing)')
    
    args = parser.parse_args()
    
    if not validate_date_format(args.start):
        logger.error(f"Invalid start date format: {args.start}. Use YYYY-MM-DD")
        return False
        
    if not validate_date_format(args.end):
        logger.error(f"Invalid end date format: {args.end}. Use YYYY-MM-DD")
        return False
        
    if args.start > args.end:
        logger.error("Start date must be before or equal to end date")
        return False
        
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    collectors_to_run = []
    
    if args.all:
        collectors_to_run = ['alpha', 'twelve', 'gkg', 'econ', 'fred', 'wiki']
    else:
        if args.alpha:
            collectors_to_run.append('alpha')
        if args.twelve:
            collectors_to_run.append('twelve')
        if args.gkg:
            collectors_to_run.append('gkg')
        if args.econ:
            collectors_to_run.append('econ')
        if args.fred:
            collectors_to_run.append('fred')
        if args.wiki:
            collectors_to_run.append('wiki')
            
    if not collectors_to_run:
        logger.error("No collectors specified. Use --all or specify individual collectors.")
        return False
        
    logger.info(f"Running collectors: {collectors_to_run}")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Output directory: {args.output}")
    
    results = {}
    
    for collector_name in collectors_to_run:
        try:
            if collector_name == 'alpha':
                results['alpha'] = run_alpha_vantage(args.start, args.end, args.output)
            elif collector_name == 'twelve':
                results['twelve'] = run_twelvedata(args.start, args.end, args.output)
            elif collector_name == 'gkg':
                results['gkg'] = run_gdelt(args.start, args.end, args.output)
            elif collector_name == 'econ':
                results['econ'] = run_tradingeconomics(args.start, args.end, args.output)
            elif collector_name == 'fred':
                results['fred'] = run_fred(args.start, args.end, args.output)
            elif collector_name == 'wiki':
                results['wiki'] = run_wikipedia(args.start, args.end, args.output)
        except Exception as e:
            logger.error(f"Error running {collector_name} collector: {e}")
            results[collector_name] = False
            
    logger.info("Collection Results:")
    successful = 0
    for collector, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {collector}: {status}")
        if success:
            successful += 1
            
    logger.info(f"Successfully collected from {successful}/{len(results)} sources")
    
    return successful > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
