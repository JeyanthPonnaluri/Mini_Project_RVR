"""
Sustainability and free-rider analysis for federated learning.
Implements learning curve experiments and free-rider detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score
import os

from federated import partition_equal, partition_imbalanced, fedavg_train, train_local_models
from logistic_numpy import predict_proba


def run_learning_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hospital_counts: List[int],
    rounds: int,
    epochs: int,
    lr: float,
    trials: int = 10,
    partition_type: str = 'equal',
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Run learning curve experiment: study how performance changes with number of hospitals.
    
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
    hospital_counts : List[int]
        List of hospital counts to test (e.g., [2, 3, 5, 7, 10])
    rounds : int
        Number of communication rounds
    epochs : int
        Local epochs per round
    lr : float
        Learning rate
    trials : int
        Number of Monte Carlo trials per configuration
    partition_type : str
        'equal' or 'imbalanced'
    random_seed : int
        Base random seed
        
    Returns:
    --------
    pd.DataFrame
        Results with columns: K, trial, global_auc, avg_local_auc
    """
    print(f"\n{'='*70}")
    print(f"LEARNING CURVE EXPERIMENT")
    print(f"{'='*70}")
    print(f"Hospital counts: {hospital_counts}")
    print(f"Trials per configuration: {trials}")
    print(f"Partition type: {partition_type}")
    print(f"Rounds: {rounds}, Epochs: {epochs}, LR: {lr}")
    print(f"{'='*70}\n")
    
    results = []
    
    for K in hospital_counts:
        print(f"\n--- Testing K = {K} hospitals ---")
        
        for trial in range(trials):
            # Set seed for reproducibility
            trial_seed = random_seed + trial * 100 + K
            
            # Partition data
            if partition_type == 'equal':
                hospitals = partition_equal(X_train, y_train, K, trial_seed)
            else:
                # Generate imbalanced distribution
                distribution = generate_imbalanced_distribution(K, trial_seed)
                hospitals = partition_imbalanced(X_train, y_train, distribution, trial_seed)
            
            # Train FedAvg
            fedavg_results = fedavg_train(
                hospitals, X_test, y_test,
                rounds=rounds,
                epochs=epochs,
                lr=lr,
                random_seed=trial_seed
            )
            
            global_auc = fedavg_results['round_aucs'][-1]
            
            # Train local models
            local_aucs = train_local_models(
                hospitals, X_test, y_test,
                epochs=epochs * rounds,
                lr=lr,
                random_seed=trial_seed
            )
            
            avg_local_auc = np.mean(local_aucs)
            
            results.append({
                'K': K,
                'trial': trial + 1,
                'global_auc': global_auc,
                'avg_local_auc': avg_local_auc
            })
            
            print(f"  Trial {trial + 1}/{trials}: Global AUC = {global_auc:.4f}, Avg Local AUC = {avg_local_auc:.4f}")
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"LEARNING CURVE COMPLETE")
    print(f"{'='*70}\n")
    
    # Print summary statistics
    summary = df.groupby('K').agg({
        'global_auc': ['mean', 'std'],
        'avg_local_auc': ['mean', 'std']
    }).round(4)
    print("Summary Statistics:")
    print(summary)
    print()
    
    return df


def run_free_rider_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hospital_counts: List[int],
    rounds: int,
    epochs: int,
    lr: float,
    trials: int = 10,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Run free-rider experiment: study how excluded hospitals benefit from federation.
    
    Free-Rider Scenario:
    - Partition data into K hospitals
    - Randomly select 1 hospital as "free-rider" (excluded from training)
    - Train FedAvg on remaining K-1 hospitals
    - Evaluate global model on free-rider's test data
    
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
    hospital_counts : List[int]
        List of hospital counts to test
    rounds : int
        Number of communication rounds
    epochs : int
        Local epochs per round
    lr : float
        Learning rate
    trials : int
        Number of Monte Carlo trials
    random_seed : int
        Base random seed
        
    Returns:
    --------
    pd.DataFrame
        Results with columns: K, trial, free_rider_auc, global_auc
    """
    print(f"\n{'='*70}")
    print(f"FREE-RIDER EXPERIMENT")
    print(f"{'='*70}")
    print(f"Hospital counts: {hospital_counts}")
    print(f"Trials per configuration: {trials}")
    print(f"Rounds: {rounds}, Epochs: {epochs}, LR: {lr}")
    print(f"{'='*70}\n")
    
    results = []
    
    for K in hospital_counts:
        if K < 2:
            print(f"Skipping K={K} (need at least 2 hospitals for free-rider)")
            continue
        
        print(f"\n--- Testing K = {K} hospitals ---")
        
        for trial in range(trials):
            # Set seed for reproducibility
            trial_seed = random_seed + trial * 100 + K
            np.random.seed(trial_seed)
            
            # Partition data into K hospitals
            hospitals = partition_equal(X_train, y_train, K, trial_seed)
            
            # Randomly select one hospital as free-rider
            free_rider_idx = np.random.randint(0, K)
            
            # Get free-rider's data
            X_fr, y_fr = hospitals[free_rider_idx]
            
            # Create participating hospitals (exclude free-rider)
            participating_hospitals = [hospitals[i] for i in range(K) if i != free_rider_idx]
            
            # Train FedAvg on participating hospitals only
            fedavg_results = fedavg_train(
                participating_hospitals, X_test, y_test,
                rounds=rounds,
                epochs=epochs,
                lr=lr,
                random_seed=trial_seed
            )
            
            w_global = fedavg_results['w_global']
            global_auc = fedavg_results['round_aucs'][-1]
            
            # Evaluate global model on free-rider's data
            y_fr_proba = predict_proba(X_fr, w_global)
            free_rider_auc = roc_auc_score(y_fr, y_fr_proba)
            
            results.append({
                'K': K,
                'trial': trial + 1,
                'free_rider_auc': free_rider_auc,
                'global_auc': global_auc,
                'participating_hospitals': K - 1
            })
            
            print(f"  Trial {trial + 1}/{trials}: Free-Rider AUC = {free_rider_auc:.4f}, Global AUC = {global_auc:.4f}")
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"FREE-RIDER EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")
    
    # Print summary statistics
    summary = df.groupby('K').agg({
        'free_rider_auc': ['mean', 'std'],
        'global_auc': ['mean', 'std']
    }).round(4)
    print("Summary Statistics:")
    print(summary)
    print()
    
    return df


def generate_imbalanced_distribution(K: int, random_seed: int = 42) -> List[float]:
    """
    Generate imbalanced distribution for K hospitals.
    
    Parameters:
    -----------
    K : int
        Number of hospitals
    random_seed : int
        Random seed
        
    Returns:
    --------
    List[float]
        Distribution that sums to 1.0
    """
    np.random.seed(random_seed)
    
    # Generate random weights
    weights = np.random.exponential(scale=1.0, size=K)
    
    # Normalize to sum to 1
    distribution = weights / weights.sum()
    
    return distribution.tolist()


def plot_learning_curve(
    df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot learning curve: K vs AUC.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results from run_learning_curve
    save_path : str
        Path to save plot
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    # Compute statistics
    stats = df.groupby('K').agg({
        'global_auc': ['mean', 'std'],
        'avg_local_auc': ['mean', 'std']
    })
    
    K_values = stats.index.values
    global_mean = stats['global_auc']['mean'].values
    global_std = stats['global_auc']['std'].values
    local_mean = stats['avg_local_auc']['mean'].values
    local_std = stats['avg_local_auc']['std'].values
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot global AUC
    ax.plot(K_values, global_mean, 'o-', linewidth=2, markersize=8, 
            label='FedAvg (Global)', color='#2E86AB')
    ax.fill_between(K_values, global_mean - global_std, global_mean + global_std,
                     alpha=0.2, color='#2E86AB')
    
    # Plot local AUC
    ax.plot(K_values, local_mean, 's-', linewidth=2, markersize=8,
            label='Local Models (Avg)', color='#A23B72')
    ax.fill_between(K_values, local_mean - local_std, local_mean + local_std,
                     alpha=0.2, color='#A23B72')
    
    ax.set_xlabel('Number of Hospitals (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=13, fontweight='bold')
    ax.set_title('Learning Curve: Federated Learning Scalability', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([min(local_mean) - 0.05, max(global_mean) + 0.02])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve plot saved to {save_path}")
    
    return fig


def plot_free_rider_curve(
    df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot free-rider curve: K vs Free-Rider AUC.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results from run_free_rider_experiment
    save_path : str
        Path to save plot
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    # Compute statistics
    stats = df.groupby('K').agg({
        'free_rider_auc': ['mean', 'std'],
        'global_auc': ['mean', 'std']
    })
    
    K_values = stats.index.values
    fr_mean = stats['free_rider_auc']['mean'].values
    fr_std = stats['free_rider_auc']['std'].values
    global_mean = stats['global_auc']['mean'].values
    global_std = stats['global_auc']['std'].values
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot free-rider AUC
    ax.plot(K_values, fr_mean, 'o-', linewidth=2, markersize=8,
            label='Free-Rider AUC', color='#F18F01')
    ax.fill_between(K_values, fr_mean - fr_std, fr_mean + fr_std,
                     alpha=0.2, color='#F18F01')
    
    # Plot global AUC for reference
    ax.plot(K_values, global_mean, 's-', linewidth=2, markersize=8,
            label='Global AUC (Participants)', color='#2E86AB')
    ax.fill_between(K_values, global_mean - global_std, global_mean + global_std,
                     alpha=0.2, color='#2E86AB')
    
    ax.set_xlabel('Number of Hospitals (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=13, fontweight='bold')
    ax.set_title('Free-Rider Analysis: Non-Participating Hospital Performance',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Free-rider plot saved to {save_path}")
    
    return fig


def save_sustainability_results(
    learning_curve_df: pd.DataFrame,
    free_rider_df: pd.DataFrame,
    save_dir: str = 'reports/version3'
) -> None:
    """
    Save sustainability experiment results.
    
    Parameters:
    -----------
    learning_curve_df : pd.DataFrame
        Learning curve results
    free_rider_df : pd.DataFrame
        Free-rider results
    save_dir : str
        Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save CSVs
    lc_path = os.path.join(save_dir, 'learning_curve_results.csv')
    fr_path = os.path.join(save_dir, 'free_rider_results.csv')
    
    learning_curve_df.to_csv(lc_path, index=False)
    free_rider_df.to_csv(fr_path, index=False)
    
    print(f"\nResults saved:")
    print(f"  - {lc_path}")
    print(f"  - {fr_path}")
