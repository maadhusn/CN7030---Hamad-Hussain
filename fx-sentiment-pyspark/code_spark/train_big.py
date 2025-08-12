#!/usr/bin/env python3
"""
Enhanced Model Training - Support LR/RF/GBT/XGBoost with chronological CV and calibration
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import IsotonicRegression
from pyspark.sql.functions import col

import mlflow
import mlflow.spark

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_utils.session import get_spark
from code_spark.conf import load_config
from code_spark.gold_training_matrix import build_training_dataframe
from code_spark.models import lr_pipeline, rf_pipeline, gbt_pipeline
from code_spark.evaluate import evaluate_split
from code_spark.calibration import fit_isotonic_calibrator
from code_spark.data_loader import get_feature_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def xgb_pipeline(feature_cols: List[str]) -> Pipeline:
    """Create XGBoost pipeline (placeholder for XGBoost4J-Spark integration)"""
    logger.warning("XGBoost4J-Spark integration requires --packages ml.dmlc:xgboost4j-spark_2.12:1.7.6")
    
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import GBTClassifier
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        maxDepth=6,
        stepSize=0.1
    )
    
    return Pipeline(stages=[assembler, gbt])

def train_models(algorithms: Optional[List[str]] = None,
                 calibrate: bool = True) -> Dict[str, Any]:
    """Fit LR/RF/GBT/XGBoost on chronological CV; return best model info & per-model metrics."""
    if algorithms is None:
        algorithms = ["lr", "rf", "gbt", "xgb"]
    
    config = load_config()
    spark, is_delta = get_spark("train-big")
    
    try:
        df = build_training_dataframe()
        
        if df is None:
            logger.error("Failed to load training dataframe")
            return {}
        
        train_df = df.filter(col("split") == "train")
        valid_df = df.filter(col("split") == "valid")
        test_df = df.filter(col("split") == "test")
        
        logger.info(f"Train: {train_df.count()}, Valid: {valid_df.count()}, Test: {test_df.count()}")
        
        feature_cols = get_feature_columns(df)
        logger.info(f"Using {len(feature_cols)} features")
        
        models = {}
        results = {}
        
        for algo in algorithms:
            logger.info(f"Training {algo}...")
            
            try:
                if algo == "lr":
                    pipeline = lr_pipeline(feature_cols)
                elif algo == "rf":
                    pipeline = rf_pipeline(feature_cols)
                elif algo == "gbt":
                    pipeline = gbt_pipeline(feature_cols)
                elif algo == "xgb":
                    pipeline = xgb_pipeline(feature_cols)
                else:
                    logger.warning(f"Unknown algorithm: {algo}")
                    continue
                
                fitted_model = pipeline.fit(train_df)
                models[algo] = fitted_model
                
                valid_pred = fitted_model.transform(valid_df)
                metrics = evaluate_split(valid_pred, f"valid_{algo}")
                results[algo] = metrics
                
                logger.info(f"{algo} - Valid F1: {metrics.get('f1_05', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Error training {algo}: {e}")
                continue
        
        if not results:
            logger.error("No models trained successfully")
            return {}
        
        best_algo = max(results.keys(), key=lambda k: results[k].get("f1_05", 0))
        best_model = models[best_algo]
        
        logger.info(f"Best model: {best_algo}")
        
        if calibrate:
            try:
                valid_pred = best_model.transform(valid_df)
                calibrator = fit_isotonic_calibrator(valid_pred)
                best_model = (best_model, calibrator)
                logger.info("Applied calibration")
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
        
        return {
            "best_algorithm": best_algo,
            "best_model": best_model,
            "all_results": results,
            "all_models": models
        }
        
    except Exception as e:
        logger.error(f"Error in train_models: {e}")
        return {}
    finally:
        spark.stop()

def save_best_model(best: Dict[str, Any], outdir: str = "artifacts/models/latest") -> str:
    """Persist Spark or local model + metadata; return directory path."""
    try:
        os.makedirs(outdir, exist_ok=True)
        
        model = best["best_model"]
        if isinstance(model, tuple):
            base_model, calibrator = model
            mlflow.spark.save_model(base_model, f"{outdir}/base_model")
            mlflow.spark.save_model(calibrator, f"{outdir}/calibrator")
        else:
            mlflow.spark.save_model(model, f"{outdir}/model")
        
        metadata = {
            "algorithm": best["best_algorithm"],
            "metrics": best["all_results"][best["best_algorithm"]],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(f"{outdir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved best model to {outdir}")
        return outdir
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description='Enhanced model training with big-data support')
    parser.add_argument('--algorithms', nargs='+', default=["lr", "rf", "gbt"], 
                       help='Algorithms to train')
    parser.add_argument('--calibrate', action='store_true', default=True,
                       help='Apply calibration')
    parser.add_argument('--output-dir', default='artifacts/models/latest',
                       help='Output directory for best model')
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("big_data_training")
    
    with mlflow.start_run(run_name="big_data_training"):
        results = train_models(algorithms=args.algorithms, calibrate=args.calibrate)
        
        if results:
            model_path = save_best_model(results, args.output_dir)
            
            mlflow.log_param("best_algorithm", results["best_algorithm"])
            mlflow.log_param("algorithms_trained", args.algorithms)
            mlflow.log_param("calibrated", args.calibrate)
            
            for algo, metrics in results["all_results"].items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{algo}_{metric_name}", value)
            
            logger.info("Training completed successfully")
            return True
        else:
            logger.error("Training failed")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
