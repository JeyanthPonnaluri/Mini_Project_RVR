"""
Model evaluation module for clinical stage classification.
Implements metrics and visualization for model performance assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    auc, 
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray or pd.Series
        True test labels
        
    Returns:
    --------
    dict
        Dictionary containing:
        - auc: AUC-ROC score
        - accuracy: Classification accuracy
        - confusion_matrix: 2x2 confusion matrix
        - y_pred: Predicted labels
        - y_pred_proba: Predicted probabilities
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    acc_score = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"Accuracy: {acc_score:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Early Stage (T1/T2)', 'Advanced Stage (T3/T4)']))
    
    results = {
        'auc': auc_score,
        'accuracy': acc_score,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """
    Plot ROC curve with AUC score.
    
    Parameters:
    -----------
    y_test : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Clinical Stage Classification (T3/T4 vs Others)', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return fig


def plot_confusion_matrix(conf_matrix, save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Parameters:
    -----------
    conf_matrix : np.ndarray
        2x2 confusion matrix
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Early Stage\n(T1/T2)', 'Advanced Stage\n(T3/T4)']
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if conf_matrix[i, j] > thresh else "black",
                   fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig
