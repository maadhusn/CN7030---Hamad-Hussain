import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import os
from typing import Dict, Tuple, List

def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, 
                         title: str = "", save_path: str = None) -> plt.Figure:
    """Generate confusion matrix plot"""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix{" - " + title if title else ""}')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "", 
                   save_path: str = None) -> plt.Figure:
    """Generate ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "", 
                                save_path: str = None) -> plt.Figure:
    """Generate Precision-Recall curve plot"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPRC = {auprc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "", 
                          save_path: str = None) -> plt.Figure:
    """Generate calibration plot"""
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Plot{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_performance_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                               save_path: str = None) -> plt.Figure:
    """Generate performance comparison bar chart"""
    models = list(metrics_dict.keys())
    metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall', 'auroc']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics_to_plot):
        values = []
        labels = []
        
        for model in models:
            if metric in metrics_dict[model]:
                values.append(metrics_dict[model][metric])
                labels.append(model.upper())
        
        if values:
            bars = axes[i].bar(labels, values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def save_all_plots(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, 
                   split_name: str, output_dir: str) -> Dict[str, str]:
    """Save all standard plots for a model/split combination"""
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = {}
    
    conf_path = f"{output_dir}/confusion_{model_name}_{split_name}.png"
    plot_confusion_matrix(y_true, y_prob, save_path=conf_path)
    plot_paths['confusion'] = conf_path
    
    roc_path = f"{output_dir}/roc_{model_name}_{split_name}.png"
    plot_roc_curve(y_true, y_prob, save_path=roc_path)
    plot_paths['roc'] = roc_path
    
    pr_path = f"{output_dir}/pr_{model_name}_{split_name}.png"
    plot_precision_recall_curve(y_true, y_prob, save_path=pr_path)
    plot_paths['pr'] = pr_path
    
    cal_path = f"{output_dir}/calibration_{model_name}_{split_name}.png"
    plot_calibration_curve(y_true, y_prob, save_path=cal_path)
    plot_paths['calibration'] = cal_path
    
    plt.close('all')
    
    return plot_paths
