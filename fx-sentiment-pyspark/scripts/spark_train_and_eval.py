#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.spark
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_spark.data_loader import (
    build_spark, load_first_available, sanitize_schema, get_feature_columns,
    split_timewise, print_split_summary, MAX_TRAIN_ROWS, MAX_VALID_ROWS, MAX_TEST_ROWS, PLOT_ROWS
)
from code_spark.models import lr_pipeline, rf_pipeline, gbt_pipeline, compute_class_weights
from code_spark.evaluate import add_prob_column, evaluate_split, find_best_threshold
from code_spark.calibration import fit_isotonic_calibrator, apply_isotonic

def main(max_train_rows: int = MAX_TRAIN_ROWS,
         max_valid_rows: int = MAX_VALID_ROWS,
         max_test_rows: int = MAX_TEST_ROWS,
         plot_rows: int = PLOT_ROWS,
         choose_metric: str = "f1") -> None:
    
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("spark_baselines")
    
    spark = build_spark("spark-baselines")
    
    try:
        print("=== Loading Data ===")
        
        paths = [
            "delta/silver/silver_eurusd_1h_features_noleak",
            "delta/silver/silver_eurusd_1h_features"
        ]
        df = load_first_available(spark, paths)
        df = sanitize_schema(df)
        
        print(f"Loaded {df.count()} rows with {len(df.columns)} columns")
        
        feature_cols = get_feature_columns(df)
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        if "label" not in df.columns:
            print("Creating label from r_fwd1...")
            from pyspark.sql.functions import when, col
            df = df.withColumn("label", when(col("r_fwd1") > 0, 1).otherwise(0))
        else:
            print("Converting 3-class labels (-1,0,1) to binary (0,1) for Spark ML...")
            from pyspark.sql.functions import when, col
            df = df.withColumn("label", 
                when(col("label") == 1, 1)   # UP -> 1 (positive)
                .otherwise(0)                # DOWN/FLAT -> 0 (negative)
            )
        
        print("=== Splitting Data ===")
        
        caps = {"train": max_train_rows, "valid": max_valid_rows, "test": max_test_rows}
        train_df, valid_df, test_df = split_timewise(df, caps=caps)
        
        print_split_summary(train_df, valid_df, test_df)
        
        train_df = compute_class_weights(train_df)
        
        print("=== Training Models ===")
        
        models = {
            "lr_l2": lr_pipeline(feature_cols, elastic_net=0.0, weight_col="weight"),
            "lr_elastic": lr_pipeline(feature_cols, elastic_net=0.5, weight_col="weight"),
            "rf": rf_pipeline(feature_cols, num_trees=200, max_depth=8),
            "gbt": gbt_pipeline(feature_cols, max_iter=200, max_depth=5)
        }
        
        fitted_models = {}
        best_model_name = None
        best_valid_auc = 0
        
        for model_name, pipeline in models.items():
            print(f"\nTraining {model_name}...")
            
            with mlflow.start_run(run_name=f"baseline_{model_name}"):
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("max_train_rows", max_train_rows)
                mlflow.log_param("max_valid_rows", max_valid_rows)
                mlflow.log_param("max_test_rows", max_test_rows)
                mlflow.log_param("num_features", len(feature_cols))
                
                fitted_model = pipeline.fit(train_df)
                fitted_models[model_name] = fitted_model
                
                valid_pred = fitted_model.transform(valid_df)
                valid_pred = add_prob_column(valid_pred)
                
                valid_metrics = evaluate_split(
                    valid_pred, "valid", 
                    plot_limit=plot_rows,
                    artifacts_dir="artifacts/plots",
                    model_tag=model_name
                )
                
                valid_pandas = valid_pred.select("label", "prob").limit(plot_rows).toPandas()
                if len(valid_pandas) > 0:
                    y_true = valid_pandas["label"].values
                    y_prob = valid_pandas["prob"].values
                    
                    y_true_binary = y_true
                    
                    best_thr, best_metrics = find_best_threshold(y_true_binary, y_prob, choose_metric)
                    mlflow.log_param("best_threshold", best_thr)
                    
                    for metric_name, value in valid_metrics.items():
                        mlflow.log_metric(f"valid_{metric_name}", value)
                    
                    if valid_metrics["auroc"] > best_valid_auc:
                        best_valid_auc = valid_metrics["auroc"]
                        best_model_name = model_name
                    
                    print(f"{model_name} - Valid AUC: {valid_metrics['auroc']:.3f}, Best thr: {best_thr:.3f}")
                
                # mlflow.spark.log_model(fitted_model, f"model_{model_name}")
                print(f"Model {model_name} trained successfully")
        
        print(f"\nBest model: {best_model_name} (Valid AUC: {best_valid_auc:.3f})")
        
        print("=== Calibration and Final Evaluation ===")
        
        best_model = fitted_models[best_model_name]
        
        valid_pred = best_model.transform(valid_df)
        valid_pred = add_prob_column(valid_pred)
        
        calibrator = fit_isotonic_calibrator(valid_pred, score_col="prob")
        
        test_pred = best_model.transform(test_df)
        test_pred = add_prob_column(test_pred)
        test_pred_cal = apply_isotonic(test_pred, calibrator)
        
        test_metrics_raw = evaluate_split(
            test_pred, "test_raw",
            plot_limit=plot_rows,
            artifacts_dir="artifacts/plots",
            model_tag=f"{best_model_name}_raw"
        )
        
        test_metrics_cal = evaluate_split(
            test_pred_cal, "test_calibrated",
            prob_col="calibrated_prob",
            plot_limit=plot_rows,
            artifacts_dir="artifacts/plots",
            model_tag=f"{best_model_name}_calibrated"
        )
        
        with mlflow.start_run(run_name="final_results"):
            mlflow.log_param("best_model", best_model_name)
            
            for metric_name, value in test_metrics_raw.items():
                mlflow.log_metric(f"test_raw_{metric_name}", value)
            
            for metric_name, value in test_metrics_cal.items():
                mlflow.log_metric(f"test_cal_{metric_name}", value)
        
        print("=== Generating Report ===")
        
        os.makedirs("artifacts", exist_ok=True)
        
        report_content = f"""# Spark ML Baselines Report

- Total rows: {df.count()}
- Features: {len(feature_cols)}
- Train/Valid/Test: {train_df.count()}/{valid_df.count()}/{test_df.count()}

- Validation AUC: {best_valid_auc:.3f}


| Metric | Raw | Calibrated |
|--------|-----|------------|
| AUROC | {test_metrics_raw.get('auroc', 0):.3f} | {test_metrics_cal.get('auroc', 0):.3f} |
| AUPRC | {test_metrics_raw.get('auprc', 0):.3f} | {test_metrics_cal.get('auprc', 0):.3f} |
| F1@0.5 | {test_metrics_raw.get('f1_05', 0):.3f} | {test_metrics_cal.get('f1_05', 0):.3f} |
| Brier RMSE | {test_metrics_raw.get('brier_rmse', 0):.3f} | {test_metrics_cal.get('brier_rmse', 0):.3f} |

- ROC curves: `artifacts/plots/roc_*.png`
- PR curves: `artifacts/plots/pr_*.png`
- Calibration plots: `artifacts/plots/calibration_*.png`
- Confusion matrices: `artifacts/plots/confusion_*.png`

- Tracking URI: ./mlruns
- Experiment: spark_baselines
"""
        
        with open("artifacts/REPORT_BASELINES_SPARK.md", "w") as f:
            f.write(report_content)
        
        print("Report saved to artifacts/REPORT_BASELINES_SPARK.md")
        print("Training completed successfully!")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark ML Baselines Training")
    parser.add_argument("--max-train-rows", type=int, default=MAX_TRAIN_ROWS)
    parser.add_argument("--max-valid-rows", type=int, default=MAX_VALID_ROWS)
    parser.add_argument("--max-test-rows", type=int, default=MAX_TEST_ROWS)
    parser.add_argument("--plot-rows", type=int, default=PLOT_ROWS)
    parser.add_argument("--choose-metric", default="f1")
    
    args = parser.parse_args()
    
    main(
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        max_test_rows=args.max_test_rows,
        plot_rows=args.plot_rows,
        choose_metric=args.choose_metric
    )
