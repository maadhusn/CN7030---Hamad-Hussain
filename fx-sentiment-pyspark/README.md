# FX Sentiment Analysis with PySpark and Delta Lake

A comprehensive data pipeline for FX sentiment analysis using PySpark, Delta Lake, and MLflow. This project collects data from multiple free sources and processes it through a medallion architecture (Bronze → Silver → Gold) for machine learning model training.

## 🏗️ Architecture

```
fx-sentiment-pyspark/
├── configs/                    # Configuration files
│   ├── data.yaml              # Data sources and symbols
│   ├── features.yaml          # Feature engineering settings
│   ├── train.yaml             # Model training configuration
│   └── profiles.yaml          # Environment profiles
├── bronze_ingest/             # Data collection scripts
│   ├── alpha_vantage_collector.py
│   ├── twelvedata_collector.py
│   ├── gdelt_collector.py
│   ├── tradingecon_collector.py
│   ├── fred_collector.py
│   ├── wiki_collector.py
│   └── run_backfill.py        # Orchestration script
├── spark_jobs/                # Spark processing jobs
│   └── bronze_to_delta.py     # Bronze layer ingestion
├── delta/                     # Delta Lake storage
│   ├── bronze/               # Raw data tables
│   ├── silver/               # Cleaned data tables
│   └── gold/                 # Feature-engineered tables
├── mlruns/                   # MLflow experiment tracking
├── app/fastapi_gateway/      # API gateway (placeholder)
└── tests/                    # Test files
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required for full functionality
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key_here
TWELVEDATA_API_KEY=your_twelvedata_key_here
TRADINGECONOMICS_API_KEY=your_tradingeconomics_key_here

# Optional (FRED works without key but with limitations)
FRED_API_KEY=your_fred_key_here
```

#### Getting Free API Keys

- **Alpha Vantage**: [Get free key](https://www.alphavantage.co/support/#api-key) (500 requests/day)
- **TwelveData**: [Get free key](https://twelvedata.com/pricing) (800 requests/day)
- **Trading Economics**: [Get free key](https://tradingeconomics.com/api) (1000 requests/month)
- **FRED**: [Get free key](https://fred.stlouisfed.org/docs/api/api_key.html) (optional, improves rate limits)

### 3. Data Collection

Run backfill for a small date window (recommended for testing):

```bash
# Collect data from all sources for a 10-day period
python bronze_ingest/run_backfill.py --all --start 2019-01-01 --end 2019-01-10

# Or collect from specific sources
python bronze_ingest/run_backfill.py --alpha --twelve --gkg --fred --wiki --start 2019-01-01 --end 2019-01-10
```

Available collector flags:
- `--alpha`: Alpha Vantage FX daily data
- `--twelve`: TwelveData FX hourly data  
- `--gkg`: GDELT Global Knowledge Graph
- `--econ`: Trading Economics calendar
- `--fred`: Federal Reserve Economic Data
- `--wiki`: Wikipedia pageviews
- `--all`: All sources

### 4. Bronze Layer Processing

Load collected data into Delta Bronze tables:

```bash
# Batch processing (default)
spark-submit spark_jobs/bronze_to_delta.py --mode batch

# With custom paths
spark-submit spark_jobs/bronze_to_delta.py --mode batch --landing-path landing --delta-path delta
```

### 5. Verify Data Ingestion

Check that Delta tables were created with data:

```bash
# List bronze tables
ls -la delta/bronze/

# Check table contents using PySpark
python -c "
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

builder = SparkSession.builder.appName('DataCheck')
spark = configure_spark_with_delta_pip(builder).getOrCreate()

tables = ['bronze_fx', 'bronze_gkg', 'bronze_econ', 'bronze_fred', 'bronze_wiki']
for table in tables:
    try:
        df = spark.read.format('delta').load(f'delta/bronze/{table}')
        print(f'{table}: {df.count()} rows')
    except Exception as e:
        print(f'{table}: ERROR - {e}')

spark.stop()
"
```

## 📊 Data Sources

| Source | Type | Frequency | Description |
|--------|------|-----------|-------------|
| Alpha Vantage | FX Rates | Daily | EUR/USD daily OHLC data |
| TwelveData | FX Rates | Hourly | EUR/USD hourly OHLC data |
| GDELT | News Sentiment | 15min | Global news sentiment and events |
| Trading Economics | Economic Events | Daily | Economic calendar and indicators |
| FRED | Economic Data | Daily | Federal Reserve economic series |
| Wikipedia | Public Interest | Daily | Pageviews for relevant articles |

## ⚙️ Configuration

### Environment Profiles

Three pre-configured profiles in `configs/profiles.yaml`:

- **devin_local**: Small date window (2019-01-01 to 2019-01-10) for quick testing
- **standard_lab**: Standard environment (2019-01-01 to 2023-12-31)
- **full_power**: Full production environment (2019-01-01 to 2024-12-31)

### Feature Engineering

Configure in `configs/features.yaml`:
- **Horizon**: 1h (fixed for this implementation)
- **Rolling windows**: 24h, 168h (1 week), 720h (1 month)
- **Technical indicators**: SMA, EMA, RSI, Bollinger Bands
- **Sentiment features**: GDELT tone, Wikipedia pageviews

### Model Training

Configure in `configs/train.yaml`:
- **Baseline models**: Logistic Regression, Random Forest (enabled)
- **Ensemble methods**: Voting, Stacking (enabled)
- **Advanced models**: XGBoost, Neural Networks (disabled)

## 🔄 Streaming Mode (Future)

The pipeline is designed to support streaming ingestion:

```bash
# Streaming mode (placeholder for future implementation)
spark-submit spark_jobs/bronze_to_delta.py --mode stream
```

## 📈 Scaling Notes

This architecture is designed for easy scaling:

1. **Cloud Storage**: Point `--landing-path` and `--delta-path` to S3/ADLS
2. **Distributed Processing**: Run on Spark clusters (Databricks, EMR, etc.)
3. **Streaming**: Enable `--mode stream` for real-time processing
4. **API Gateway**: Implement FastAPI service in `app/fastapi_gateway/`

## 🧪 Testing

```bash
# Run tests (when implemented)
python -m pytest tests/

# Lint code
flake8 bronze_ingest/ spark_jobs/

# Type checking
mypy bronze_ingest/ spark_jobs/
```

## 📝 Development

### Adding New Data Sources

1. Create collector in `bronze_ingest/new_source_collector.py`
2. Add to `run_backfill.py` orchestration
3. Update `bronze_to_delta.py` to process new source
4. Add configuration to `configs/data.yaml`

### Extending to Silver/Gold Layers

1. Create `spark_jobs/silver_transformations.py`
2. Create `spark_jobs/gold_features.py`
3. Add feature engineering logic
4. Integrate with MLflow for model training

## 🚨 Troubleshooting

### Common Issues

1. **API Rate Limits**: Increase `RATE_LIMIT_DELAY` in `.env`
2. **Missing Data**: Check API keys and network connectivity
3. **Spark Memory**: Adjust driver/executor memory in profiles
4. **Delta Lake**: Ensure proper Spark configuration for Delta

### Logs

All components use structured logging. Check logs for detailed error information:

```bash
# View recent logs
tail -f /path/to/logfile

# Debug specific collector
python bronze_ingest/alpha_vantage_collector.py --help
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review API documentation for data sources

---

**Note**: This is a development/research project. For production use, implement proper error handling, monitoring, and security measures.
