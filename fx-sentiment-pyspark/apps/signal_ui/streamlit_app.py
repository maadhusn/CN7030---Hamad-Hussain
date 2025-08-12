#!/usr/bin/env python3
"""
FX Signal Dashboard - Streamlit UI for model predictions and metrics visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import glob
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    calibration_curve, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, log_loss, brier_score_loss
)

def load_metrics_table() -> pd.DataFrame:
    """Load metrics from artifacts/metrics/*.csv"""
    try:
        metrics_files = glob.glob("artifacts/metrics/*.csv")
        if not metrics_files:
            return pd.DataFrame()
        
        dfs = []
        for file in metrics_files:
            df = pd.read_csv(file)
            df['source_file'] = Path(file).name
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return pd.DataFrame()

def load_predictions(limit: int | None = None) -> pd.DataFrame:
    """Load predictions from artifacts/predictions/*.parquet"""
    try:
        pred_files = glob.glob("artifacts/predictions/*.parquet")
        if not pred_files:
            return pd.DataFrame()
        
        dfs = []
        for file in pred_files:
            df = pd.read_parquet(file)
            if limit and len(df) > limit:
                df = df.head(limit)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

def compute_classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Compute classification metrics at given threshold"""
    try:
        y_pred = (y_score >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0,
            'logloss': log_loss(y_true, y_score, eps=1e-15),
            'brier': brier_score_loss(y_true, y_score)
        }
        
        return metrics
    except Exception as e:
        st.error(f"Error computing metrics: {e}")
        return {}

def plot_confusion(y_true, y_score, threshold: float = 0.5, title: str = "") -> plt.Figure:
    """Generate confusion matrix plot"""
    try:
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               title=title or f'Confusion Matrix (threshold={threshold:.2f})',
               ylabel='True label',
               xlabel='Predicted label')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating confusion matrix: {e}")
        return plt.figure()

def plot_roc(y_true, y_score, title: str = "") -> plt.Figure:
    """Generate ROC curve plot"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title or 'ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating ROC curve: {e}")
        return plt.figure()

def plot_pr(y_true, y_score, title: str = "") -> plt.Figure:
    """Generate Precision-Recall curve plot"""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label='PR curve')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title or 'Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating PR curve: {e}")
        return plt.figure()

def plot_calibration(y_true, y_score, title: str = "") -> plt.Figure:
    """Generate calibration plot"""
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_score, n_bins=10
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                color='blue', label='Model')
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title or 'Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating calibration plot: {e}")
        return plt.figure()

def main():
    st.set_page_config(
        page_title="FX Signal Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 FX Signal Dashboard")
    st.markdown("Real-time model predictions and performance metrics for EURUSD trading signals")
    
    st.sidebar.header("Controls")
    
    refresh_data = st.sidebar.button("🔄 Refresh Data")
    
    prediction_limit = st.sidebar.slider(
        "Prediction Rows to Display", 
        min_value=100, 
        max_value=10000, 
        value=1000, 
        step=100
    )
    
    threshold = st.sidebar.slider(
        "Classification Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.01
    )
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "🔮 Predictions", "📈 Performance", "ℹ️ Model Info"])
    
    with tab1:
        st.header("Model Metrics")
        
        metrics_df = load_metrics_table()
        
        if not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True)
            
            if 'accuracy' in metrics_df.columns:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics_df['accuracy'].iloc[-1]:.3f}")
                with col2:
                    st.metric("F1 Score", f"{metrics_df.get('f1', [0])[-1]:.3f}")
                with col3:
                    st.metric("AUC", f"{metrics_df.get('auc', [0])[-1]:.3f}")
                with col4:
                    st.metric("Precision", f"{metrics_df.get('precision', [0])[-1]:.3f}")
        else:
            st.warning("No metrics data found. Run model training to generate metrics.")
    
    with tab2:
        st.header("Recent Predictions")
        
        pred_df = load_predictions(limit=prediction_limit)
        
        if not pred_df.empty:
            st.dataframe(pred_df.head(100), use_container_width=True)
            
            if 'ts' in pred_df.columns and 'probability' in pred_df.columns:
                fig = px.line(
                    pred_df.head(500), 
                    x='ts', 
                    y='probability',
                    title='Prediction Probability Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No prediction data found. Run batch prediction to generate predictions.")
    
    with tab3:
        st.header("Model Performance")
        
        pred_df = load_predictions(limit=prediction_limit)
        
        if not pred_df.empty and 'label' in pred_df.columns and 'probability' in pred_df.columns:
            y_true = pred_df['label'].values
            y_score = pred_df['probability'].values
            
            if len(np.unique(y_true)) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(plot_roc(y_true, y_score))
                    st.pyplot(plot_confusion(y_true, y_score, threshold))
                
                with col2:
                    st.pyplot(plot_pr(y_true, y_score))
                    st.pyplot(plot_calibration(y_true, y_score))
                
                metrics = compute_classification_metrics(y_true, y_score, threshold)
                
                st.subheader("Performance Metrics")
                metric_cols = st.columns(len(metrics))
                for i, (metric, value) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        st.metric(metric.upper(), f"{value:.3f}")
            else:
                st.warning("Insufficient label diversity for performance analysis.")
        else:
            st.warning("No labeled prediction data found for performance analysis.")
    
    with tab4:
        st.header("Model Information")
        
        try:
            metadata_path = "artifacts/models/latest/metadata.json"
            if Path(metadata_path).exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                st.json(metadata)
            else:
                st.warning("No model metadata found.")
        except Exception as e:
            st.error(f"Error loading model metadata: {e}")
        
        st.subheader("Data Sources")
        st.markdown("""
        - **FX Data**: Alpha Vantage (daily) + TwelveData (1h)
        - **GDELT**: Global Knowledge Graph sentiment
        - **FRED**: US economic indicators (CPI, PCE)
        - **Calendar**: US economic release dates
        - **Wikipedia**: USD page views
        """)
        
        st.subheader("Feature Engineering")
        st.markdown("""
        - **Price Features**: Returns (1h, 3h, 6h), volatility, EMA
        - **Sentiment**: GDELT tone and event counts
        - **Macro**: CPI/PCE YoY changes and surprise proxies
        - **Calendar**: Event flags and timing windows
        """)

if __name__ == "__main__":
    main()
