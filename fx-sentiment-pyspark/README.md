# Big-Data PySpark FX Sentiment Analysis

This is a Big-Data PySpark project for FX sentiment analysis using Delta Lake, MLflow, Feast feature store, and Streamlit UI. All pipelines are configured for maximum available history across FX/FRED/GDELT/Calendar data sources and designed for cluster deployment.

**Expected Data Volumes**: Qualify as big-data; run on cluster (Spark standalone/YARN/Databricks)  
**Training Target**: ≥70% accuracy with LR/RF/GBT/XGBoost and time-based CV (no guarantee; depends on data)  
**Architecture**: Medallion (Bronze → Silver → Gold) with unlimited data processing

## 🏗️ Repo Map & Responsibilities

### Core Pipeline Components
- **code_spark/**: Big-data PySpark modules for unlimited history processing
  - `ingest_*.py`: Data ingestion jobs (FX, FRED, GDELT, Calendar) → Delta Bronze partitioned by year/month/day
  - `silver_*.py`: Feature engineering and aggregation → Delta Silver 1h bars
  - `gold_training_matrix.py`: Labeled training data with chronological splits
  - `train_big.py`: Enhanced model training (LR/RF/GBT/XGBoost) with chronological CV and calibration
  - `predict_batch.py`: Uncapped batch prediction over date ranges
  - `data_loader.py`: Configuration-driven data loading with big-data mode (no caps)
  - `models.py`, `evaluate.py`, `calibration.py`: ML pipeline components

### Legacy Components (Small-Scale Development)
- **bronze_ingest/**: Original collectors for development/testing
- **spark_jobs/**: Original processing jobs with row caps
- **configs/**: Legacy YAML configuration (superseded by conf/project.yaml)

### UI & Feature Store
- **apps/signal_ui/streamlit_app.py**: Signal dashboard reading artifacts/predictions/*.parquet and artifacts/metrics/*.csv
- **feature_repo/**: Feast feature store with Delta offline sources and SQLite online store

### Infrastructure
- **conf/project.yaml**: Single configuration file for big-data operations
- **Makefile**: Cluster-oriented targets for ingestion, training, and UI
- **docs/**: Deployment guides and data source documentation

## 🔄 Data Flow & Logic

### Bronze → Silver Pipeline
1. **Data Ingestion**: `code_spark/ingest_*.py` → Delta Bronze tables partitioned by date
   - FX: Alpha Vantage (daily) + TwelveData (1h) → `bronze_fx`
   - FRED: Economic series (CPIAUCSL, PCEPI, UNRATE, FEDFUNDS, DTWEXBGS) → `bronze_fred_series`
   - GDELT: Global Knowledge Graph sentiment → `bronze_gkg`
   - Calendar: US economic release dates → `bronze_fred_releases`

2. **Silver Processing**: Hourly aggregation and feature preparation
   - `silver_fx_features.py`: 1h EURUSD bars with technical indicators
   - `silver_us_calendar_1h.py`: Event flags and timing windows

### Features Engineering
Built in `spark_jobs/features_build.py` with no row caps in big-data mode:
- **Price Features**: Returns (1h, 3h, 6h, 12h, 24h), realized volatility (6h, 24h, 72h), EMA (α=2/(w+1), 6h, 24h, 72h)
- **Sentiment Features**: GDELT tone (24h mean), event counts for keywords (CPI, inflation, payrolls, FOMC, ECB)
- **Macro Features**: FRED series with YoY/MoM transformations
- **Surprise Proxy**: Absolute deviation from 12-month rolling mean/median at event_0h anchor hours only
- **Event Flags**: `event_day_any`, `event_0h_any`, `pre_event_2h`, `post_event_2h`

### Labels & Training Logic
Generated in `spark_jobs/labels_make.py`:
- **3-Class Labels**: DOWN (-1), FLAT (0), UP (1) based on 1h forward returns
- **Dead-Zone**: Adaptive (40th percentile) or fixed (3 bps) threshold to handle noise
- **Chronological Split**: 70% train / 15% valid / 15% test with strict time ordering
- **Leakage Guards**: Assert max(train.ts) < min(valid.ts) < min(test.ts)

## ⚙️ Configuration (No-Code Parameter Changes)

### Primary Configuration: conf/project.yaml
```yaml
mode: bigdata        # {bigdata|dev} - controls row caps
spark:
  shuffle_partitions: 400
  dynamic_allocation: true
data_ranges:
  fx:
    symbol: "EURUSD"
    start: null       # ALL available history
    end: null
  fred:
    series: ["CPIAUCSL","PCEPI"]
    start: null       # ALL available history
  gdelt:
    start: "2015-02-18T00:00:00Z"  # earliest reliable for GKG 2.1
training:
  model_candidates: ["lr","rf","gbt","xgb"]
  cv_folds: 5
  calibrate: true
limits:
  enabled: true      # dev mode caps
  max_rows: 1000000
big_run:
  enabled: false     # safety guard
  unlimited_data: true
```

### CLI Flag Overrides
- `--algorithms lr rf gbt`: Override model candidates
- `--output-dir artifacts/models/custom`: Change model save location
- `--start 2020-01-01 --end 2023-12-31`: Override date ranges

### Environment Overrides
- `BIG_RUN=1`: Enable unlimited data processing
- `MAX_TRAIN_ROWS=500000`: Override row caps in dev mode

## 🔑 API Keys & Secrets

Create `conf/secrets.env` (never commit):
```bash
# Required for full functionality
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key_here
TWELVEDATA_API_KEY=your_twelvedata_key_here

# Optional (improves rate limits)
FRED_API_KEY=your_fred_key_here

# Disabled in current config
TRADINGECONOMICS_API_KEY=
```

### Getting Free API Keys
- **Alpha Vantage**: [Get free key](https://www.alphavantage.co/support/#api-key) (500 requests/day) - Powers daily FX data
- **TwelveData**: [Get free key](https://twelvedata.com/pricing) (800 requests/day) - Powers 1h FX bars
- **FRED**: [Get free key](https://fred.stlouisfed.org/docs/api/api_key.html) (optional) - Powers macro series

## 🚀 How to Run

### Local Development
```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure secrets
cp conf/secrets.env.example conf/secrets.env
# Edit conf/secrets.env with your API keys

# 3. Set development mode
export BIG_RUN=0  # Safety guard

# 4. Run small-scale pipeline (legacy)
make train  # Uses spark_jobs/ with row caps

# 5. Run Streamlit UI
make ui-run  # streamlit run apps/signal_ui/streamlit_app.py --server.headless true
```

### Colab Deployment
Create `notebooks/Colab_Quickstart.ipynb`:
```python
# Installation cell
!pip install pyspark==3.5.0 delta-spark==3.0.0 mlflow==2.10.2
!pip install pandas numpy requests python-dotenv pyarrow pyyaml tqdm
!pip install scikit-learn matplotlib feast streamlit plotly

# Setup cell
import os
os.environ['ALPHAVANTAGE_API_KEY'] = 'your_key_here'
os.environ['TWELVEDATA_API_KEY'] = 'your_key_here'
os.environ['BIG_RUN'] = '0'  # Keep safe for Colab

# Run pipeline cell
!python -m code_spark.train_big --algorithms lr rf --output-dir /content/models
```

### Cluster Deployment (Production)
```bash
# 1. Set cloud storage paths
export DELTA_PATH="s3a://your-bucket/delta"
export ARTIFACTS_PATH="s3a://your-bucket/artifacts"

# 2. Enable big-data mode
export BIG_RUN=1

# 3. Submit with XGBoost4J-Spark package
spark-submit \
  --packages ml.dmlc:xgboost4j-spark_2.12:1.7.6 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --driver-memory 8g --executor-memory 16g \
  code_spark/train_big.py --algorithms lr rf gbt xgb --calibrate

# 4. Batch prediction (unlimited data)
make predict-batch  # Uses code_spark/predict_batch.py with no caps
```

## 🎯 Training & Evaluation

### Algorithms & Features
- **Models**: XGBoost (default), LogisticRegression, RandomForest, GBTClassifier
- **Features Used**: 16-feature allow-list from price/sentiment/macro/event data
- **Cross-Validation**: Time-series splits with chronological ordering
- **Calibration**: Isotonic regression on validation scores
- **Threshold Tuning**: Maximize F1 score on validation set

### Hyperparameter Hints for ≥70% Accuracy
```python
# XGBoost4J-Spark (recommended)
xgb_params = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# Logistic Regression (fallback)
lr_params = {
    "regParam": 0.01,
    "elasticNetParam": 0.5,
    "maxIter": 100
}
```

### Evaluation Metrics Produced
**Classification Metrics** (saved to `artifacts/metrics/*.csv`):
- Accuracy, Precision, Recall, F1 Score
- AUROC, AUPRC (Area Under Precision-Recall Curve)
- Log Loss, Brier Score (calibration quality)

**Visualizations** (saved to `artifacts/plots/`):
- ROC Curve, Precision-Recall Curve
- Calibration Plot (reliability diagram)
- Confusion Matrix (at 0.5 and best-F1 thresholds)
- Probability Histogram
- Feature Importance (tree-based models)
- Training vs Validation Learning Curves

**Regression Mode Metrics** (if applicable):
- RMSE, MAE, R²
- RMSE² (squared RMSE for penalty emphasis)

## 📊 Streamlit Signal UI

### Launch Command
```bash
streamlit run apps/signal_ui/streamlit_app.py --server.headless true
# Or: make ui-run
```

### Dashboard Features
- **Metrics Tab**: Model performance summary from `artifacts/metrics/*.csv`
- **Predictions Tab**: Recent predictions from `artifacts/predictions/*.parquet` with time-series plots
- **Performance Tab**: ROC/PR curves, confusion matrices, calibration plots
- **Model Info Tab**: Metadata, data sources, feature engineering summary

### Interactive Controls
- Classification threshold slider (0.0 - 1.0)
- Prediction row limit (100 - 10,000)
- Refresh data button

### Data Sources Read
- `artifacts/predictions/*.parquet`: Batch prediction results with ts, symbol, probability, label columns
- `artifacts/metrics/*.csv`: Training metrics with accuracy, f1, auc, precision, recall columns
- `artifacts/models/latest/metadata.json`: Model algorithm, timestamp, performance summary

## 🍽️ Optional: Feast Feature Store

### Setup (Development)
```bash
# Apply feature definitions
make feast-apply  # cd feature_repo && feast apply

# Materialize features (optional)
cd feature_repo && python materialize_online.py
```

### Configuration
- **Offline Store**: File-based pointing to Delta silver paths with no range restrictions
- **Online Store**: SQLite for development (Redis recommended for production)
- **Feature Views**: EURUSD hourly features with 16-feature allow-list

### Production Notes
- **Protobuf Compatibility**: Pin `protobuf>=3.20.0,<5.0.0` in requirements.txt
- **Cluster Materialization**: Use `feast materialize-incremental` on schedule
- **Online Serving**: Switch to Redis/DynamoDB for production workloads

### Workaround for Import Issues
If Feast import fails due to protobuf conflicts:
```bash
pip install protobuf==3.20.3  # Specific compatible version
# Or skip Feast integration - it's optional
```

## 🔧 Troubleshooting / FAQ

### Q: Do removed row caps cause data leakage?
**A**: No. Chronological splits with strict time ordering prevent leakage. The `split_timewise()` function enforces `max(train.ts) < min(valid.ts) < min(test.ts)` regardless of data volume.

### Q: How to tune for ≥70% accuracy?
**A**: 
1. Enable XGBoost with `--algorithms xgb` and cluster deployment
2. Increase feature engineering windows in `configs/features.yaml`
3. Adjust dead-zone threshold in label generation
4. Use more economic event keywords in calendar configuration
5. Extend data history by setting `start: null` in conf/project.yaml

### Q: Provider timezone/rate limit issues?
**A**:
- **Alpha Vantage**: 500 calls/day, UTC timestamps
- **TwelveData**: 800 calls/day, market timezone aware
- **FRED**: No key required but rate limits apply
- **GDELT**: 15-minute delay, UTC timestamps

### Q: Common runtime/import issues?
**A**:
```bash
# PySpark not found
export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

# Delta Lake issues
pip install delta-spark==3.0.0

# Memory issues
export SPARK_DRIVER_MEMORY=8g
export SPARK_EXECUTOR_MEMORY=16g
```

## 📋 Appendix: Public Function Signatures

### code_spark.train_big
```python
def prepare_spark() -> Tuple[SparkSession, bool]
def load_datasets() -> DataFrame
def build_feature_df() -> DataFrame
def train_model(algorithm: str = "lr") -> Dict[str, Any]
def evaluate_model(model, test_data) -> Dict[str, float]
def save_artifacts(results, output_dir: str = "artifacts/models/latest") -> str
def train_models(algorithms: Optional[List[str]] = None, calibrate: bool = True) -> Dict[str, Any]
def save_best_model(best: Dict[str, Any], outdir: str = "artifacts/models/latest") -> str
```

### code_spark.predict_batch
```python
def predict_batch(output_path: str, start_ts: str, end_ts: str, symbol: str = "EURUSD") -> str
```

### apps.signal_ui.streamlit_app
```python
def load_metrics_table() -> pd.DataFrame
def load_predictions(limit: int | None = None) -> pd.DataFrame
def compute_classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict
def plot_confusion(y_true, y_score, threshold: float = 0.5, title: str = "") -> plt.Figure
def plot_roc(y_true, y_score, title: str = "") -> plt.Figure
def plot_pr(y_true, y_score, title: str = "") -> plt.Figure
def plot_calibration(y_true, y_score, title: str = "") -> plt.Figure
def build_app(test_mode=False) -> bool
```

### code_spark.data_loader
```python
def get_row_caps() -> Optional[Dict[str, int]]
def build_spark(app_name: str = "eurusd-ml") -> SparkSession
def get_feature_columns(df: DataFrame, label_col: str = "label", drop_cols: Optional[List[str]] = None, max_missing_frac: float = 0.20) -> List[str]
def split_timewise(df: DataFrame, ts_col: str = "ts", splits: Tuple[float, float, float] = (0.70, 0.15, 0.15), caps: Optional[Dict[str, int]] = None) -> Tuple[DataFrame, DataFrame, DataFrame]
```

---

**Note**: This is a production big-data project. For development/testing, use legacy `make train` with row caps. For production deployment, use `make train-big` on clusters with unlimited data processing.
