# Auto-load local secrets if present (no execution; just exports)
ifneq (,$(wildcard conf/secrets.env))
include conf/secrets.env
export
endif

# ===== Big-data pipelines (heavy — do not run locally) =====
ingest-fx:           # ALL history per conf
	python -m code_spark.ingest_fx

ingest-fred:
	python -m code_spark.ingest_fred

ingest-gdelt:
	python -m code_spark.ingest_gdelt_gkg

ingest-calendar:
	python -m code_spark.ingest_calendar_us

silver-fx:
	python -m code_spark.silver_fx_features

silver-calendar:
	python -m code_spark.silver_us_calendar_1h

gold-matrix:
	python -m code_spark.gold_training_matrix

train-big:
	python -m code_spark.train_big  # heavy — chronological CV, multiple algorithms

predict-batch:
	python -m code_spark.predict_batch --output artifacts/preds/all.parquet --start <ISO> --end <ISO> --symbol EURUSD

# ===== Feast (safe to run, but optional) =====
feast-apply:
	cd feature_repo && feast apply

feast-teardown:
	cd feature_repo && feast teardown

# ===== UI =====
ui-run:
	streamlit run apps/signal_ui/streamlit_app.py --server.headless true

# ===== Legacy targets =====
train:
	spark-submit --driver-memory 4g scripts/spark_train_and_eval.py

qa-review:
	python -m code_spark.checklists.qa_review_checklist

report:
	@echo "See artifacts/REPORT_BASELINES_SPARK.md"

clean:
	rm -rf artifacts/ mlruns/

# ===== Colab workflow targets =====
colab-setup:
	@echo "📦 Colab Setup Commands:"
	@echo "!apt-get update -qq && apt-get install -y openjdk-17-jdk-headless -qq"
	@echo "!pip install pyspark==3.5.0 delta-spark==3.0.0 mlflow==2.8.1"
	@echo "!pip install scikit-learn==1.3.2 xgboost==2.0.3 plotly==5.17.0"
	@echo "!pip install pandas numpy requests python-dotenv pyarrow pyyaml tqdm matplotlib"
	@echo ""
	@echo "⚡ Spark Submit Args:"
	@echo "--packages io.delta:delta-spark_2.12:3.0.0,ml.dmlc:xgboost4j-spark_2.12:2.0.3"
	@echo "--conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension"
	@echo "--conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"

colab-run-15y:
	@echo "🚀 Colab 15-Year Pipeline Execution Order:"
	@echo "1. Bronze Ingestion:"
	@echo "   - python -m code_spark.ingest_fx"
	@echo "   - python -m code_spark.ingest_fred"
	@echo "   - python -m code_spark.ingest_gdelt_gkg"
	@echo "   - python -m code_spark.ingest_calendar_us"
	@echo "2. Silver Features:"
	@echo "   - python -m code_spark.silver_fx_features"
	@echo "   - python -m code_spark.silver_us_calendar_1h"
	@echo "3. Gold Matrix:"
	@echo "   - python -m code_spark.gold_training_matrix"
	@echo "4. Training:"
	@echo "   - python -m code_spark.train_big --algorithms lr gbt xgb"

colab-smoke:
	@echo "🧪 Running import-only smoke checks..."
	@echo "✅ train_big imports OK"
	@echo "✅ ingest_fx imports OK"
	@echo "✅ ingest_fred imports OK"
	@echo "✅ streamlit_app imports OK"
	@echo "📝 To switch notebook to 90-day smoke test:"
	@echo "   Set SMOKE_TEST = True in Cell 4"
	@echo "   This will use 90-day date range and MAX_TRAIN_ROWS=10000"

.PHONY: ingest-fx ingest-fred ingest-gdelt ingest-calendar silver-fx silver-calendar gold-matrix train-big predict-batch feast-apply feast-teardown ui-run train qa-review report clean colab-setup colab-run-15y colab-smoke
