train:
	spark-submit --driver-memory 4g scripts/spark_train_and_eval.py

report:
	@echo "See artifacts/REPORT_BASELINES_SPARK.md"

clean:
	rm -rf artifacts/ mlruns/

.PHONY: train report clean
