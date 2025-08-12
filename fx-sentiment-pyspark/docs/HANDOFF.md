# Big-Data PySpark Handoff Guide

## Cluster Deployment

### Spark Configuration
```bash
# Recommended cluster settings
spark.sql.adaptive.enabled=true
spark.sql.adaptive.coalescePartitions.enabled=true
spark.sql.adaptive.skewJoin.enabled=true
spark.serializer=org.apache.spark.serializer.KryoSerializer
spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension
spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog
```

### Deployment Platforms
- **Spark Standalone**: Use `spark-submit` with cluster mode
- **YARN**: Configure `--master yarn --deploy-mode cluster`
- **Databricks**: Upload as workspace notebooks or jobs
- **EMR**: Use EMR Steps with Delta Lake support

## Backfill Strategy

### Per Dataset Configuration
All datasets support `start: null` for maximum available history:

```yaml
# conf/project.yaml
data_ranges:
  fx:
    start: null  # ALL available from provider
  fred:
    start: null  # ALL available series history
  gdelt:
    start: "2015-02-18T00:00:00Z"  # Earliest reliable GKG 2.1
```

### Execution Order
1. `make ingest-fx` - FX data from Alpha Vantage + TwelveData
2. `make ingest-fred` - FRED series and release dates
3. `make ingest-gdelt` - GDELT Global Knowledge Graph
4. `make ingest-calendar` - US economic calendar events
5. `make silver-fx` - Hourly FX features with rolling windows
6. `make silver-calendar` - Calendar event flags
7. `make gold-matrix` - Training matrix with chronological splits

## Model Training

### Algorithms and Hyperparameters
- **Logistic Regression**: L2 regularization, elastic net support
- **Random Forest**: 200 trees, max depth 8, class weights
- **Gradient Boosting**: 200 iterations, max depth 5
- **XGBoost**: Requires `--packages ml.dmlc:xgboost4j-spark_2.12:1.7.6`

### Chronological Cross-Validation
- 5-fold time series splits with strict chronological ordering
- Assert `max(train.ts) < min(valid.ts) < min(test.ts)`
- Threshold tuning to maximize F1 score on validation

### Calibration
- Isotonic regression on validation scores
- Applied to test set for final evaluation
- Brier score and reliability diagrams for assessment

## Artifacts and UI

### Generated Artifacts
- `artifacts/models/latest/` - Best model + metadata
- `artifacts/predictions/*.parquet` - Batch prediction results
- `artifacts/metrics/*.csv` - Training and evaluation metrics
- `artifacts/plots/` - ROC, PR, calibration curves

### Streamlit Dashboard
```bash
make ui-run
# Access at http://localhost:8501
```

**Features**:
- Real-time metrics display
- Prediction visualization
- Model performance analysis
- ROC/PR/calibration plots
- Confusion matrices at multiple thresholds

## Feast Feature Store

### Offline Store (Delta)
```bash
make feast-apply  # Configure feature views
```

Points to Delta silver tables with no range restrictions:
- `delta/silver/silver_eurusd_1h_features`
- Entity: EURUSD currency pair
- Features: 16 engineered features (price, sentiment, macro, calendar)

### Online Store (SQLite/Redis)
```bash
cd feature_repo
feast materialize-incremental $(date -d '7 days ago' -I) $(date -I)
```

For production, replace SQLite with Redis/DynamoDB for low-latency serving.

### Feature Serving
```python
from feast import FeatureStore
store = FeatureStore(repo_path="feature_repo")
features = store.get_online_features(
    features=["eurusd_features:ret_1", "eurusd_features:rv_6"],
    entity_rows=[{"eurusd": "EURUSD"}]
)
```

## Data Quality and Monitoring

### Partitioning Strategy
All Delta tables partitioned by `year/month/day` for efficient querying and maintenance.

### Watermarks and Incremental Processing
- Structured Streaming ready (set `--mode stream`)
- Watermarks for late data handling
- Incremental merges with upsert semantics

### Data Lineage
- MLflow experiment tracking
- Delta Lake time travel for data versioning
- Feast feature lineage and monitoring

## Troubleshooting

### Common Issues
1. **Memory**: Increase driver/executor memory for large datasets
2. **Partitions**: Tune `spark.sql.shuffle.partitions` based on data size
3. **Delta**: Ensure Delta Lake packages in classpath
4. **XGBoost**: Verify XGBoost4J-Spark package availability

### Performance Optimization
- Use broadcast joins for small dimension tables
- Cache frequently accessed DataFrames
- Optimize file sizes with `OPTIMIZE` and `Z-ORDER`
- Monitor Spark UI for bottlenecks

### Monitoring
- MLflow tracking URI for experiment history
- Feast feature monitoring for drift detection
- Delta Lake audit logs for data lineage
- Streamlit dashboard for real-time monitoring
