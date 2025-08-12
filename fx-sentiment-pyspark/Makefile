train:
	spark-submit --driver-memory 4g scripts/spark_train_and_eval.py

qa-review:
	python -m code_spark.checklists.qa_review_checklist

report:
	@echo "See artifacts/REPORT_BASELINES_SPARK.md"

clean:
	rm -rf artifacts/ mlruns/

.PHONY: train qa-review report clean
