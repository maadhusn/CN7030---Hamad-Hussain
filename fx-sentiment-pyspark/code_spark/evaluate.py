from typing import Dict, Tuple, List
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve
)

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
import os

def add_prob_column(pred_df: DataFrame,
                    prob_col: str = "probability",
                    out_col: str = "prob") -> DataFrame:
    from pyspark.sql.functions import when, isnan, isnull
    from pyspark.sql.types import DoubleType
    
    if prob_col not in pred_df.columns:
        return pred_df.withColumn(out_col, 
            when(col("prediction") == 1, 0.7)  # Positive class gets higher score
            .otherwise(0.3)  # Negative class gets lower score
        )
    
    @udf(returnType=DoubleType())
    def extract_prob_1(prob_vector):
        if prob_vector is None:
            return None
        if len(prob_vector) == 2:
            return float(prob_vector[1])
        else:
            return float(max(prob_vector))
    
    return pred_df.withColumn(out_col, extract_prob_1(col(prob_col)))

def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return log_loss(y_true, y_prob)

def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    brier_mse = brier_score_loss(y_true, y_prob)
    brier_rmse = np.sqrt(brier_mse)
    return brier_mse, brier_rmse

def metrics_at_threshold(y_true: np.ndarray,
                         y_prob: np.ndarray,
                         thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    
    y_true_binary = y_true
    
    acc = accuracy_score(y_true_binary, y_pred)
    prec = precision_score(y_true_binary, y_pred, zero_division=0)
    rec = recall_score(y_true_binary, y_pred, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true_binary, y_pred)
    mcc = matthews_corrcoef(y_true_binary, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred).ravel()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def find_best_threshold(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = -1
    best_thr = 0.5
    best_metrics = {}
    
    for thr in thresholds:
        metrics = metrics_at_threshold(y_true, y_prob, thr)
        score = metrics[metric]
        
        if score > best_score:
            best_score = score
            best_thr = thr
            best_metrics = metrics
    
    return best_thr, best_metrics

def curve_data(y_true: np.ndarray,
               y_prob: np.ndarray) -> Dict[str, np.ndarray]:
    y_true_binary = y_true
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
    auc = roc_auc_score(y_true_binary, y_prob)
    
    precision, recall, _ = precision_recall_curve(y_true_binary, y_prob)
    auprc = average_precision_score(y_true_binary, y_prob)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_binary, y_prob, n_bins=10
    )
    
    thresholds = np.linspace(0.01, 0.99, 99)
    sweep_metrics = {"thr": [], "acc": [], "prec": [], "rec": [], "f1": []}
    
    for thr in thresholds:
        metrics = metrics_at_threshold(y_true_binary, y_prob, thr)
        sweep_metrics["thr"].append(thr)
        sweep_metrics["acc"].append(metrics["accuracy"])
        sweep_metrics["prec"].append(metrics["precision"])
        sweep_metrics["rec"].append(metrics["recall"])
        sweep_metrics["f1"].append(metrics["f1"])
    
    pos_scores = y_prob[y_true_binary == 1]
    neg_scores = y_prob[y_true_binary == 0]
    
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        score_range = np.linspace(0, 1, 100)
        cdf_pos = np.array([np.mean(pos_scores <= s) for s in score_range])
        cdf_neg = np.array([np.mean(neg_scores <= s) for s in score_range])
        ks_stat = np.max(np.abs(cdf_pos - cdf_neg))
    else:
        score_range = np.linspace(0, 1, 100)
        cdf_pos = np.zeros_like(score_range)
        cdf_neg = np.zeros_like(score_range)
        ks_stat = 0.0
    
    return {
        "fpr": fpr, "tpr": tpr, "auc": auc,
        "precision": precision, "recall": recall, "auprc": auprc,
        "cal_true": fraction_of_positives, "cal_prob": mean_predicted_value,
        "sweep_thr": np.array(sweep_metrics["thr"]),
        "sweep_acc": np.array(sweep_metrics["acc"]),
        "sweep_prec": np.array(sweep_metrics["prec"]),
        "sweep_rec": np.array(sweep_metrics["rec"]),
        "sweep_f1": np.array(sweep_metrics["f1"]),
        "score_range": score_range, "cdf_pos": cdf_pos, "cdf_neg": cdf_neg, "ks": ks_stat
    }

def evaluate_split(pred_df: DataFrame,
                   split_name: str,
                   label_col: str = "label",
                   prob_col: str = "prob",
                   choose_thr_from: str = "valid",
                   chosen_thr: float = 0.5,
                   plot_limit: int = 50000,
                   artifacts_dir: str = "artifacts/plots",
                   model_tag: str = "model") -> Dict[str, float]:
    sample_df = pred_df.limit(plot_limit)
    pandas_df = sample_df.toPandas()
    
    if len(pandas_df) == 0:
        print(f"No data for {split_name}")
        return {}
    
    y_true = pandas_df[label_col].values
    y_prob = pandas_df[prob_col].values
    
    y_true_binary = y_true
    
    auc = roc_auc_score(y_true_binary, y_prob)
    auprc = average_precision_score(y_true_binary, y_prob)
    logloss = compute_log_loss(y_true_binary, y_prob)
    brier_mse, brier_rmse = compute_brier(y_true_binary, y_prob)
    
    metrics_05 = metrics_at_threshold(y_true_binary, y_prob, 0.5)
    metrics_opt = metrics_at_threshold(y_true_binary, y_prob, chosen_thr)
    
    curves = curve_data(y_true_binary, y_prob)
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(curves["fpr"], curves["tpr"], label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_tag} ({split_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{artifacts_dir}/roc_{model_tag}_{split_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(curves["recall"], curves["precision"], label=f'PR (AUPRC = {auprc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_tag} ({split_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{artifacts_dir}/pr_{model_tag}_{split_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(curves["cal_prob"], curves["cal_true"], 'o-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Plot - {model_tag} ({split_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{artifacts_dir}/calibration_{model_tag}_{split_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true_binary, (y_prob >= 0.5).astype(int))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_tag} ({split_name})')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.savefig(f"{artifacts_dir}/confusion_{model_tag}_{split_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    metrics = {
        "auroc": auc,
        "auprc": auprc,
        "logloss": logloss,
        "brier_mse": brier_mse,
        "brier_rmse": brier_rmse,
        "ks_statistic": curves["ks"],
        "acc_05": metrics_05["accuracy"],
        "prec_05": metrics_05["precision"],
        "rec_05": metrics_05["recall"],
        "f1_05": metrics_05["f1"],
        "bal_acc_05": metrics_05["balanced_accuracy"],
        "mcc_05": metrics_05["mcc"],
        "acc_opt": metrics_opt["accuracy"],
        "prec_opt": metrics_opt["precision"],
        "rec_opt": metrics_opt["recall"],
        "f1_opt": metrics_opt["f1"],
        "bal_acc_opt": metrics_opt["balanced_accuracy"],
        "mcc_opt": metrics_opt["mcc"],
    }
    
    return metrics
