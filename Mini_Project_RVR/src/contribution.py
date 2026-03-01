"""
Hospital contribution analysis for federated learning.
Measures the impact of each hospital on global model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from sklearn.metrics import roc_auc_score

from federated import fedavg_train, fedprox_train
from logistic_numpy import predict_proba


def measure_hospital_contribution(
    hospitals: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 30,
    epochs: int = 5,
    lr: float = 0.1,
    algorithm: str = 'fedavg',
    mu: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Measure each hospital's contribution to federated learning performance.
    
    Uses leave-one-out approach:
    1. Train with all hospitals → baseline AUC
    2. For each hospital k:
       - Train without hospital k
       - Measure AUC drop
       - Contribution = baseline_AUC - without_k_AUC
    
    Parameters:
    -----------
    hospitals : List[Tuple[np.ndarray, np.ndarray]]
        List of (X_k, y_k) for each hospital
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    rounds : int
        Communication rounds
    epochs : int
        Local epochs per round
    lr : float
        Learning rate
    algorithm : str
        'fedavg' or 'fedprox'
    mu : float
        Proximal coefficient (for FedProx)
    random_seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Contribution analysis with columns:
        - hospital_id
        - num_samples
        - baseline_auc (with all hospitals)
        - without_auc (without this hospital)
        - contribution (AUC drop)
        - contribution_pct (percentage contribution)
    """
    print(f"\n{'='*70}")
    print(f"HOSPITAL CONTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Total hospitals: {len(hospitals)}")
    print(f"Rounds: {rounds}, Epochs: {epochs}, LR: {lr}")
    print(f"{'='*70}\n")
    
    # Train with all hospitals (baseline)
    print("Training baseline model with ALL hospitals...")
    if algorithm == 'fedavg':
        baseline_results = fedavg_train(
            hospitals, X_test, y_test,
            rounds=rounds, epochs=epochs, lr=lr,
            random_seed=random_seed
        )
    else:  # fedprox
        baseline_results = fedprox_train(
            hospitals, X_test, y_test,
            rounds=rounds, epochs=epochs, lr=lr, mu=mu,
            random_seed=random_seed
        )
    
    baseline_auc = baseline_results['round_aucs'][-1]
    print(f"Baseline AUC (all hospitals): {baseline_auc:.4f}\n")
    
    # Measure contribution of each hospital
    contributions = []
    
    for k in range(len(hospitals)):
        print(f"--- Analyzing Hospital {k+1}/{len(hospitals)} ---")
        
        # Create hospital list without hospital k
        hospitals_without_k = [hospitals[i] for i in range(len(hospitals)) if i != k]
        
        # Get hospital k's sample count
        num_samples_k = len(hospitals[k][1])
        
        # Train without hospital k
        if algorithm == 'fedavg':
            without_k_results = fedavg_train(
                hospitals_without_k, X_test, y_test,
                rounds=rounds, epochs=epochs, lr=lr,
                random_seed=random_seed
            )
        else:  # fedprox
            without_k_results = fedprox_train(
                hospitals_without_k, X_test, y_test,
                rounds=rounds, epochs=epochs, lr=lr, mu=mu,
                random_seed=random_seed
            )
        
        without_k_auc = without_k_results['round_aucs'][-1]
        contribution = baseline_auc - without_k_auc
        
        contributions.append({
            'hospital_id': k + 1,
            'num_samples': num_samples_k,
            'baseline_auc': baseline_auc,
            'without_auc': without_k_auc,
            'contribution': contribution,
            'contribution_pct': (contribution / baseline_auc) * 100 if baseline_auc > 0 else 0
        })
        
        print(f"  Samples: {num_samples_k}")
        print(f"  Without Hospital {k+1} AUC: {without_k_auc:.4f}")
        print(f"  Contribution: {contribution:.4f} ({contributions[-1]['contribution_pct']:.2f}%)\n")
    
    df = pd.DataFrame(contributions)
    
    print(f"{'='*70}")
    print(f"CONTRIBUTION ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    # Summary statistics
    print("Summary:")
    print(f"  Mean contribution: {df['contribution'].mean():.4f}")
    print(f"  Std contribution: {df['contribution'].std():.4f}")
    print(f"  Max contributor: Hospital {df.loc[df['contribution'].idxmax(), 'hospital_id']}")
    print(f"  Min contributor: Hospital {df.loc[df['contribution'].idxmin(), 'hospital_id']}")
    print()
    
    return df


def plot_contribution_analysis(
    contribution_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot hospital contribution analysis.
    
    Parameters:
    -----------
    contribution_df : pd.DataFrame
        Results from measure_hospital_contribution
    save_path : str
        Path to save plot
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    hospital_ids = contribution_df['hospital_id'].values
    contributions = contribution_df['contribution'].values
    num_samples = contribution_df['num_samples'].values
    
    # Plot 1: Contribution bar chart
    colors = ['#2E86AB' if c >= 0 else '#C73E1D' for c in contributions]
    bars = ax1.bar(hospital_ids, contributions, color=colors, alpha=0.7, edgecolor='black')
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Hospital ID', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Contribution (ΔAUC)', fontsize=13, fontweight='bold')
    ax1.set_title('Hospital Contribution to Federated Learning', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9)
    
    # Plot 2: Contribution vs Sample Size
    ax2.scatter(num_samples, contributions, s=200, alpha=0.6, c=contributions,
               cmap='RdYlGn', edgecolors='black', linewidth=1.5)
    
    # Add hospital ID labels
    for i, (x, y, hid) in enumerate(zip(num_samples, contributions, hospital_ids)):
        ax2.annotate(f'H{hid}', (x, y), fontsize=10, ha='center', va='center')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Number of Samples', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Contribution (ΔAUC)', fontsize=13, fontweight='bold')
    ax2.set_title('Contribution vs Hospital Size', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Contribution plot saved to {save_path}")
    
    return fig
