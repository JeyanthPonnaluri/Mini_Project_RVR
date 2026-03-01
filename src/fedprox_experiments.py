"""
FedProx experiments and comparison with FedAvg.
Analyzes convergence stability under different data heterogeneity settings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score
import os

from federated import (
    partition_equal,
    partition_imbalanced,
    partition_dirichlet,
    fedavg_train,
    fedprox_train,
    generate_imbalanced_distribution
)


def run_fedavg_vs_fedprox_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_hospitals: int,
    partition_type: str = 'equal',
    alpha: Optional[float] = None,
    mu_values: List[float] = [0.01, 0.1, 0.5],
    rounds: int = 50,
    epochs: int = 5,
    lr: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Compare FedAvg vs FedProx under different partition strategies.
    
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
    num_hospitals : int
        Number of hospitals
    partition_type : str
        'equal', 'imbalanced', or 'dirichlet'
    alpha : Optional[float]
        Dirichlet concentration parameter (required if partition_type='dirichlet')
    mu_values : List[float]
        List of proximal coefficients to test for FedProx
    rounds : int
        Number of communication rounds
    epochs : int
        Local epochs per round
    lr : float
        Learning rate
    random_seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Comparison results with columns:
        - algorithm: 'FedAvg' or 'FedProx'
        - mu: proximal coefficient (0 for FedAvg)
        - partition_type: partition strategy
        - alpha: Dirichlet alpha (if applicable)
        - final_auc: final test AUC
        - convergence_std: std of AUC in last 10 rounds
        - avg_weight_drift: average weight change per round
    """
    print(f"\n{'='*70}")
    print(f"FEDAVG VS FEDPROX COMPARISON")
    print(f"{'='*70}")
    print(f"Partition: {partition_type}")
    if partition_type == 'dirichlet':
        print(f"Alpha: {alpha}")
    print(f"Hospitals: {num_hospitals}")
    print(f"Rounds: {rounds}, Epochs: {epochs}, LR: {lr}")
    print(f"{'='*70}\n")
    
    # Partition data
    if partition_type == 'equal':
        hospitals = partition_equal(X_train, y_train, num_hospitals, random_seed)
    elif partition_type == 'imbalanced':
        distribution = generate_imbalanced_distribution(num_hospitals, random_seed)
        hospitals = partition_imbalanced(X_train, y_train, distribution, random_seed)
    elif partition_type == 'dirichlet':
        if alpha is None:
            raise ValueError("alpha must be specified for dirichlet partition")
        hospitals = partition_dirichlet(X_train, y_train, num_hospitals, alpha, random_seed)
    else:
        raise ValueError(f"Unknown partition_type: {partition_type}")
    
    results = []
    
    # Run FedAvg (mu = 0)
    print(f"\n{'='*60}")
    print(f"Running FedAvg (baseline)")
    print(f"{'='*60}\n")
    
    fedavg_results = fedavg_train(
        hospitals, X_test, y_test,
        rounds=rounds,
        epochs=epochs,
        lr=lr,
        random_seed=random_seed
    )
    
    # Calculate convergence stability (std of last 10 rounds)
    last_10_aucs = fedavg_results['round_aucs'][-10:]
    convergence_std = np.std(last_10_aucs)
    
    # Calculate average weight drift
    avg_drift = np.mean([m['weight_drift'] for m in fedavg_results['round_metrics']]) if 'weight_drift' in fedavg_results['round_metrics'][0] else 0.0
    
    results.append({
        'algorithm': 'FedAvg',
        'mu': 0.0,
        'partition_type': partition_type,
        'alpha': alpha if partition_type == 'dirichlet' else None,
        'final_auc': fedavg_results['round_aucs'][-1],
        'convergence_std': convergence_std,
        'avg_weight_drift': avg_drift,
        'round_aucs': fedavg_results['round_aucs'],
        'round_losses': fedavg_results['round_losses']
    })
    
    # Run FedProx with different mu values
    for mu in mu_values:
        print(f"\n{'='*60}")
        print(f"Running FedProx with mu = {mu}")
        print(f"{'='*60}\n")
        
        fedprox_results = fedprox_train(
            hospitals, X_test, y_test,
            rounds=rounds,
            epochs=epochs,
            lr=lr,
            mu=mu,
            random_seed=random_seed
        )
        
        # Calculate convergence stability
        last_10_aucs = fedprox_results['round_aucs'][-10:]
        convergence_std = np.std(last_10_aucs)
        
        # Calculate average weight drift
        avg_drift = np.mean(fedprox_results['weight_drifts'])
        
        results.append({
            'algorithm': 'FedProx',
            'mu': mu,
            'partition_type': partition_type,
            'alpha': alpha if partition_type == 'dirichlet' else None,
            'final_auc': fedprox_results['round_aucs'][-1],
            'convergence_std': convergence_std,
            'avg_weight_drift': avg_drift,
            'round_aucs': fedprox_results['round_aucs'],
            'round_losses': fedprox_results['round_losses']
        })
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*70}\n")
    
    # Print summary
    summary = df[['algorithm', 'mu', 'final_auc', 'convergence_std', 'avg_weight_drift']].copy()
    print("Summary:")
    print(summary.to_string(index=False))
    print()
    
    return df


def plot_convergence_curves(
    results_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot convergence curves comparing FedAvg vs FedProx.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_fedavg_vs_fedprox_experiment
    save_path : str
        Path to save plot
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Plot AUC convergence
    for i, row in results_df.iterrows():
        label = f"{row['algorithm']}" + (f" (μ={row['mu']})" if row['mu'] > 0 else "")
        rounds = list(range(1, len(row['round_aucs']) + 1))
        
        ax1.plot(rounds, row['round_aucs'], 
                linewidth=2.5, marker='o', markersize=4, 
                label=label, color=colors[i % len(colors)])
    
    ax1.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
    ax1.set_title('Convergence: AUC vs Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Plot Loss convergence
    for i, row in results_df.iterrows():
        label = f"{row['algorithm']}" + (f" (μ={row['mu']})" if row['mu'] > 0 else "")
        rounds = list(range(1, len(row['round_losses']) + 1))
        
        ax2.plot(rounds, row['round_losses'],
                linewidth=2.5, marker='s', markersize=4,
                label=label, color=colors[i % len(colors)])
    
    ax2.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=13, fontweight='bold')
    ax2.set_title('Convergence: Loss vs Rounds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    return fig


def plot_stability_comparison(
    results_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot stability comparison (convergence std and weight drift).
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_fedavg_vs_fedprox_experiment
    save_path : str
        Path to save plot
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    algorithms = []
    for _, row in results_df.iterrows():
        if row['algorithm'] == 'FedAvg':
            algorithms.append('FedAvg')
        else:
            algorithms.append(f"FedProx\n(μ={row['mu']})")
    
    convergence_stds = results_df['convergence_std'].values
    weight_drifts = results_df['avg_weight_drift'].values
    
    colors = ['#2E86AB'] + ['#A23B72', '#F18F01', '#C73E1D', '#6A994E'][:len(algorithms)-1]
    
    # Plot convergence stability
    bars1 = ax1.bar(algorithms, convergence_stds, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Convergence Std (Last 10 Rounds)', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence Stability', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot weight drift
    bars2 = ax2.bar(algorithms, weight_drifts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average Weight Drift (L2 Norm)', fontsize=12, fontweight='bold')
    ax2.set_title('Weight Update Magnitude', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stability comparison plot saved to {save_path}")
    
    return fig


def save_fedprox_results(
    results_df: pd.DataFrame,
    save_dir: str = 'reports/version4_fedprox'
) -> None:
    """
    Save FedProx experiment results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_fedavg_vs_fedprox_experiment
    save_dir : str
        Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save summary CSV (without round-wise data)
    summary_df = results_df[['algorithm', 'mu', 'partition_type', 'alpha', 
                              'final_auc', 'convergence_std', 'avg_weight_drift']].copy()
    summary_path = os.path.join(save_dir, 'comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed results
    detailed_path = os.path.join(save_dir, 'detailed_results.txt')
    with open(detailed_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FEDAVG VS FEDPROX COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for _, row in results_df.iterrows():
            algo = row['algorithm']
            mu = row['mu']
            
            f.write(f"\n{algo}" + (f" (μ={mu})" if mu > 0 else "") + "\n")
            f.write("-" * 50 + "\n")
            f.write(f"Partition Type: {row['partition_type']}\n")
            if row['alpha'] is not None:
                f.write(f"Dirichlet Alpha: {row['alpha']}\n")
            f.write(f"Final AUC: {row['final_auc']:.4f}\n")
            f.write(f"Convergence Std (last 10 rounds): {row['convergence_std']:.4f}\n")
            f.write(f"Average Weight Drift: {row['avg_weight_drift']:.4f}\n")
            f.write("\n")
        
        # Analysis
        f.write("="*70 + "\n")
        f.write("ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        fedavg_row = results_df[results_df['algorithm'] == 'FedAvg'].iloc[0]
        fedprox_rows = results_df[results_df['algorithm'] == 'FedProx']
        
        best_fedprox = fedprox_rows.loc[fedprox_rows['final_auc'].idxmax()]
        
        f.write(f"FedAvg Final AUC: {fedavg_row['final_auc']:.4f}\n")
        f.write(f"Best FedProx Final AUC: {best_fedprox['final_auc']:.4f} (μ={best_fedprox['mu']})\n")
        f.write(f"Improvement: {best_fedprox['final_auc'] - fedavg_row['final_auc']:.4f}\n\n")
        
        f.write(f"FedAvg Convergence Std: {fedavg_row['convergence_std']:.4f}\n")
        f.write(f"Best FedProx Convergence Std: {best_fedprox['convergence_std']:.4f} (μ={best_fedprox['mu']})\n")
        f.write(f"Stability Improvement: {fedavg_row['convergence_std'] - best_fedprox['convergence_std']:.4f}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        if best_fedprox['final_auc'] > fedavg_row['final_auc']:
            f.write("✓ FedProx achieves better final performance\n")
        else:
            f.write("✗ FedAvg achieves better final performance\n")
        
        if best_fedprox['convergence_std'] < fedavg_row['convergence_std']:
            f.write("✓ FedProx provides more stable convergence\n")
        else:
            f.write("✗ FedAvg provides more stable convergence\n")
        
        f.write("\n")
    
    print(f"\nFedProx results saved:")
    print(f"  - {summary_path}")
    print(f"  - {detailed_path}")
