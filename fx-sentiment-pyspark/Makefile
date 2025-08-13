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

.PHONY: ingest-fx ingest-fred ingest-gdelt ingest-calendar silver-fx silver-calendar gold-matrix train-big predict-batch feast-apply feast-teardown ui-run train qa-review report clean
