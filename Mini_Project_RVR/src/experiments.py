"""
Experiment utilities for comparing centralized and federated learning.
Implements centralized training using NumPy-based logistic regression.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import os

from logistic_numpy import (
    initialize_weights,
    local_train,
    compute_loss,
    predict_proba,
    predict
)


def centralized_train_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    random_seed: int = 42
) -> Dict:
    """
    Train centralized logistic regression model using NumPy.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict
        Dictionary containing:
        - w: Trained weights
        - train_loss: Final training loss
        - test_loss: Final test loss
        - train_auc: Training AUC
        - test_auc: Test AUC
        - test_accuracy: Test accuracy
        - confusion_matrix: Confusion matrix
        - loss_history: Training loss per epoch
    """
    print(f"\n{'='*60}")
    print(f"CENTRALIZED TRAINING (NumPy)")
    print(f"{'='*60}")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")
    
    # Initialize weights
    n_features = X_train.shape[1]
    w_init = initialize_weights(n_features, random_seed)
    
    # Train
    w, loss_history = local_train(X_train, y_train, w_init, epochs, lr)
    
    # Evaluate on training set
    train_loss = compute_loss(X_train, y_train, w)
    y_train_proba = predict_proba(X_train, w)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Evaluate on test set
    test_loss = compute_loss(X_test, y_test, w)
    y_test_proba = predict_proba(X_test, w)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    y_test_pred = predict(X_test, w)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    print(f"Training complete!")
    print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'w': w,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_accuracy': test_accuracy,
        'confusion_matrix': conf_matrix,
        'loss_history': loss_history,
        'y_test_proba': y_test_proba,
        'y_test_pred': y_test_pred
    }


def save_fedavg_metrics(
    round_metrics: list,
    save_dir: str = 'reports/version2'
) -> None:
    """
    Save FedAvg round-wise metrics to CSV.
    
    Parameters:
    -----------
    round_metrics : list
        List of dictionaries containing metrics per round
    save_dir : str
        Directory to save metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame(round_metrics)
    save_path = os.path.join(save_dir, 'fedavg_round_metrics.csv')
    df.to_csv(save_path, index=False)
    
    print(f"FedAvg metrics saved to {save_path}")


def save_comparison_summary(
    centralized_results: Dict,
    fedavg_results: Dict,
    local_aucs: list,
    num_hospitals: int,
    rounds: int,
    epochs: int,
    lr: float,
    save_dir: str = 'reports/version2'
) -> None:
    """
    Save comparison summary between centralized and federated learning.
    
    Parameters:
    -----------
    centralized_results : Dict
        Results from centralized training
    fedavg_results : Dict
        Results from FedAvg training
    local_aucs : list
        AUC scores from local models
    num_hospitals : int
        Number of hospitals
    rounds : int
        Number of communication rounds
    epochs : int
        Number of local epochs
    lr : float
        Learning rate
    save_dir : str
        Directory to save summary
    """
    os.makedirs(save_dir, exist_ok=True)
    
    summary = []
    summary.append("="*70)
    summary.append("VERSION-2: FEDERATED LEARNING COMPARISON SUMMARY")
    summary.append("="*70)
    summary.append("")
    
    summary.append("EXPERIMENT CONFIGURATION:")
    summary.append(f"  Number of hospitals: {num_hospitals}")
    summary.append(f"  Communication rounds: {rounds}")
    summary.append(f"  Local epochs per round: {epochs}")
    summary.append(f"  Learning rate: {lr}")
    summary.append("")
    
    summary.append("RESULTS:")
    summary.append("-"*70)
    summary.append(f"Centralized (NumPy) AUC:     {centralized_results['test_auc']:.4f}")
    summary.append(f"FedAvg AUC:                  {fedavg_results['round_aucs'][-1]:.4f}")
    summary.append(f"Average Local AUC:           {np.mean(local_aucs):.4f}")
    summary.append("")
    
    summary.append("LOCAL MODEL AUCs:")
    for i, auc in enumerate(local_aucs):
        summary.append(f"  Hospital {i+1}: {auc:.4f}")
    summary.append("")
    
    summary.append("PERFORMANCE COMPARISON:")
    fedavg_auc = fedavg_results['round_aucs'][-1]
    cent_auc = centralized_results['test_auc']
    gap = cent_auc - fedavg_auc
    gap_pct = (gap / cent_auc) * 100
    
    summary.append(f"  FedAvg vs Centralized gap:   {gap:.4f} ({gap_pct:.2f}%)")
    summary.append(f"  FedAvg vs Avg Local gain:    {fedavg_auc - np.mean(local_aucs):.4f}")
    summary.append("")
    
    summary.append("CONVERGENCE:")
    summary.append(f"  Initial FedAvg AUC (Round 1): {fedavg_results['round_aucs'][0]:.4f}")
    summary.append(f"  Final FedAvg AUC:             {fedavg_results['round_aucs'][-1]:.4f}")
    summary.append(f"  Improvement:                  {fedavg_results['round_aucs'][-1] - fedavg_results['round_aucs'][0]:.4f}")
    summary.append("")
    
    summary.append("="*70)
    
    # Save to file
    save_path = os.path.join(save_dir, 'comparison_summary.txt')
    with open(save_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"Comparison summary saved to {save_path}")
    
    # Also print to console
    print('\n'.join(summary))
