# Data Sources Documentation

## Maximum Data Ranges

### FX Data (EURUSD Hourly)
**Source**: Alpha Vantage (daily) + TwelveData (1h)
**Range**: ALL available from provider used in ingestion connector
**Configuration**: Set credentials in `.env` file
**Volume**: ~8760 hours/year × multiple years = substantial dataset
**Quality**: TwelveData preferred for intraday, Alpha Vantage for daily fallback

```yaml
# conf/project.yaml
data_ranges:
  fx:
    symbol: "EURUSD"
    start: null  # ALL available history
    end: null
```

### FRED Economic Indicators
**Series**: CPI (CPIAUCSL) and PCE (PCEPI)
**Range**: Code requests earliest observation by leaving start unset
**Frequency**: Monthly data with forward-fill to hourly
**Surprise Proxies**: 12-month rolling baselines for deviation calculation
**Quality**: High-quality official US economic data

```yaml
data_ranges:
  fred:
    series: ["CPIAUCSL","PCEPI"]
    start: null  # ALL available history
    end: null
```

### GDELT Global Knowledge Graph
**Version**: GKG 2.1 format
**Range**: Code default start 2015-02-18T00:00:00Z (earliest reliable)
**Frequency**: 15-minute updates aggregated to hourly
**Features**: Tone sentiment, event counts, keyword matching
**Volume**: ~35,000+ records per hour globally
**Quality**: Real-time global news sentiment, some noise expected

```yaml
data_ranges:
  gdelt:
    start: "2015-02-18T00:00:00Z"  # earliest reliable for GKG 2.1
    end: null
```

### US Economic Calendar
**Source**: FRED release dates (fallback from TradingEconomics)
**Range**: ALL available per source (start unset)
**Events**: CPI, Employment, GDP, Personal Income releases
**Timing**: Day-level flags with optional approximate UTC times
**Quality**: Official release schedule, high reliability

```yaml
data_ranges:
  calendar_us:
    start: null
    end: null
```

## Data Collection Configuration

### API Keys Required
```bash
# .env file
ALPHAVANTAGE_API_KEY=your_key_here
TWELVEDATA_API_KEY=your_key_here
FRED_API_KEY=your_key_here
# TRADINGECONOMICS_API_KEY=  # disabled in current config
```

### Rate Limiting
- **Alpha Vantage**: 5 calls/minute (free tier)
- **TwelveData**: 800 calls/day (free tier)
- **FRED**: 120 calls/minute
- **GDELT**: No rate limits (public data)

### Storage Format
- **Landing**: Parquet files in `landing/<source>/YYYY/MM/DD/`
- **Bronze**: Delta tables partitioned by year/month/day
- **Silver**: Hourly aggregated features with rolling windows
- **Gold**: Training matrix with chronological splits

## Data Quality Notes

### Known Limitations
1. **FX Weekends**: No trading data Saturday/Sunday
2. **FRED Delays**: Monthly data released with 1-2 week lag
3. **GDELT Coverage**: English-language bias in global news
4. **Calendar Timing**: Approximate UTC times, DST not handled

### Data Validation
- Timestamp consistency across sources
- Missing value handling with forward-fill
- Outlier detection for extreme price movements
- Schema validation on ingestion

### Backfill Considerations
- **Incremental**: Use watermarks for streaming updates
- **Full Refresh**: Set `start: null` for complete history
- **Partial**: Specify date ranges for targeted backfills
- **Idempotency**: Safe to re-run collectors multiple times

## Feature Engineering Pipeline

### Price Features (FX)
- Returns: 1h, 3h, 6h horizons
- Volatility: 6h, 24h rolling windows
- EMA: 6h, 24h exponential moving averages

### Sentiment Features (GDELT)
- Tone: Hourly mean sentiment score
- Events: Keyword-based event counts
- Attention: Total document counts

### Macro Features (FRED)
- Levels: CPI, PCE monthly values
- Changes: YoY percentage changes
- Surprises: Deviation from 12-month rolling mean

### Calendar Features (US)
- Event flags: Day-level economic release indicators
- Timing: Pre/during/post event windows
- Intensity: Multiple releases on same day

## Monitoring and Maintenance

### Data Freshness
- FX: Updated every hour during trading
- FRED: Monthly releases, check for updates weekly
- GDELT: 15-minute updates, aggregate hourly
- Calendar: Updated when new releases scheduled

### Quality Metrics
- Completeness: % of expected records present
- Timeliness: Lag between event and ingestion
- Accuracy: Validation against external sources
- Consistency: Cross-source timestamp alignment

### Alerting
- Missing data beyond expected lag
- Extreme values outside historical ranges
- Schema changes in source data
- API rate limit violations
