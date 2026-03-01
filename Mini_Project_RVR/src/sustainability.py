"""
Sustainability and free-rider analysis for federated learning.
Implements learning curve experiments, free-rider detection, and partition comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel
import os

from federated import (
    partition_equal, 
    partition_imbalanced, 
    fedavg_train, 
    train_local_models,
    generate_imbalanced_distribution
)
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


def compare_partitions(
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
    Compare equal vs imbalanced partition strategies.
    
    For each K, runs both equal and imbalanced experiments and compares:
    - Global AUC (FedAvg)
    - Free-rider AUC
    - Statistical significance (paired t-test)
    
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
        Comparison results with columns:
        - K
        - equal_global_auc_mean, equal_global_auc_std
        - imbalanced_global_auc_mean, imbalanced_global_auc_std
        - equal_free_rider_auc_mean, equal_free_rider_auc_std
        - imbalanced_free_rider_auc_mean, imbalanced_free_rider_auc_std
        - global_auc_pvalue (paired t-test)
        - free_rider_auc_pvalue (paired t-test)
    """
    print(f"\n{'='*70}")
    print(f"PARTITION COMPARISON EXPERIMENT")
    print(f"{'='*70}")
    print(f"Hospital counts: {hospital_counts}")
    print(f"Trials per configuration: {trials}")
    print(f"Rounds: {rounds}, Epochs: {epochs}, LR: {lr}")
    print(f"{'='*70}\n")
    
    comparison_results = []
    
    for K in hospital_counts:
        if K < 2:
            print(f"Skipping K={K} (need at least 2 hospitals)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing K = {K} hospitals")
        print(f"{'='*60}")
        
        equal_global_aucs = []
        imbalanced_global_aucs = []
        equal_fr_aucs = []
        imbalanced_fr_aucs = []
        
        for trial in range(trials):
            trial_seed = random_seed + trial * 100 + K
            np.random.seed(trial_seed)
            
            print(f"\n--- Trial {trial + 1}/{trials} ---")
            
            # EQUAL PARTITION EXPERIMENT
            print("  Running EQUAL partition...")
            hospitals_equal = partition_equal(X_train, y_train, K, trial_seed)
            
            # Train FedAvg with equal partition
            fedavg_equal = fedavg_train(
                hospitals_equal, X_test, y_test,
                rounds=rounds,
                epochs=epochs,
                lr=lr,
                random_seed=trial_seed
            )
            equal_global_auc = fedavg_equal['round_aucs'][-1]
            equal_global_aucs.append(equal_global_auc)
            
            # Free-rider experiment with equal partition
            free_rider_idx = np.random.randint(0, K)
            X_fr, y_fr = hospitals_equal[free_rider_idx]
            participating_equal = [hospitals_equal[i] for i in range(K) if i != free_rider_idx]
            
            fedavg_equal_fr = fedavg_train(
                participating_equal, X_test, y_test,
                rounds=rounds,
                epochs=epochs,
                lr=lr,
                random_seed=trial_seed + 1000
            )
            y_fr_proba = predict_proba(X_fr, fedavg_equal_fr['w_global'])
            equal_fr_auc = roc_auc_score(y_fr, y_fr_proba)
            equal_fr_aucs.append(equal_fr_auc)
            
            print(f"    Equal - Global AUC: {equal_global_auc:.4f}, FR AUC: {equal_fr_auc:.4f}")
            
            # IMBALANCED PARTITION EXPERIMENT
            print("  Running IMBALANCED partition...")
            distribution = generate_imbalanced_distribution(K, trial_seed)
            hospitals_imbalanced = partition_imbalanced(X_train, y_train, distribution, trial_seed)
            
            # Train FedAvg with imbalanced partition
            fedavg_imbalanced = fedavg_train(
                hospitals_imbalanced, X_test, y_test,
                rounds=rounds,
                epochs=epochs,
                lr=lr,
                random_seed=trial_seed
            )
            imbalanced_global_auc = fedavg_imbalanced['round_aucs'][-1]
            imbalanced_global_aucs.append(imbalanced_global_auc)
            
            # Free-rider experiment with imbalanced partition
            X_fr_imb, y_fr_imb = hospitals_imbalanced[free_rider_idx]
            participating_imbalanced = [hospitals_imbalanced[i] for i in range(K) if i != free_rider_idx]
            
            fedavg_imbalanced_fr = fedavg_train(
                participating_imbalanced, X_test, y_test,
                rounds=rounds,
                epochs=epochs,
                lr=lr,
                random_seed=trial_seed + 1000
            )
            y_fr_imb_proba = predict_proba(X_fr_imb, fedavg_imbalanced_fr['w_global'])
            imbalanced_fr_auc = roc_auc_score(y_fr_imb, y_fr_imb_proba)
            imbalanced_fr_aucs.append(imbalanced_fr_auc)
            
            print(f"    Imbalanced - Global AUC: {imbalanced_global_auc:.4f}, FR AUC: {imbalanced_fr_auc:.4f}")
        
        # Compute statistics
        equal_global_mean = np.mean(equal_global_aucs)
        equal_global_std = np.std(equal_global_aucs)
        imbalanced_global_mean = np.mean(imbalanced_global_aucs)
        imbalanced_global_std = np.std(imbalanced_global_aucs)
        
        equal_fr_mean = np.mean(equal_fr_aucs)
        equal_fr_std = np.std(equal_fr_aucs)
        imbalanced_fr_mean = np.mean(imbalanced_fr_aucs)
        imbalanced_fr_std = np.std(imbalanced_fr_aucs)
        
        # Paired t-test
        global_auc_ttest = ttest_rel(equal_global_aucs, imbalanced_global_aucs)
        fr_auc_ttest = ttest_rel(equal_fr_aucs, imbalanced_fr_aucs)
        
        comparison_results.append({
            'K': K,
            'equal_global_auc_mean': equal_global_mean,
            'equal_global_auc_std': equal_global_std,
            'imbalanced_global_auc_mean': imbalanced_global_mean,
            'imbalanced_global_auc_std': imbalanced_global_std,
            'equal_free_rider_auc_mean': equal_fr_mean,
            'equal_free_rider_auc_std': equal_fr_std,
            'imbalanced_free_rider_auc_mean': imbalanced_fr_mean,
            'imbalanced_free_rider_auc_std': imbalanced_fr_std,
            'global_auc_pvalue': global_auc_ttest.pvalue,
            'free_rider_auc_pvalue': fr_auc_ttest.pvalue
        })
        
        print(f"\n--- K = {K} Summary ---")
        print(f"  Equal Global AUC: {equal_global_mean:.4f} ± {equal_global_std:.4f}")
        print(f"  Imbalanced Global AUC: {imbalanced_global_mean:.4f} ± {imbalanced_global_std:.4f}")
        print(f"  Global AUC p-value: {global_auc_ttest.pvalue:.4f}")
        print(f"  Equal FR AUC: {equal_fr_mean:.4f} ± {equal_fr_std:.4f}")
        print(f"  Imbalanced FR AUC: {imbalanced_fr_mean:.4f} ± {imbalanced_fr_std:.4f}")
        print(f"  FR AUC p-value: {fr_auc_ttest.pvalue:.4f}")
    
    df = pd.DataFrame(comparison_results)
    
    print(f"\n{'='*70}")
    print(f"PARTITION COMPARISON COMPLETE")
    print(f"{'='*70}\n")
    
    return df


def plot_partition_comparison(
    df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot comparison between equal and imbalanced partitions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results from compare_partitions
    save_path : str
        Path to save plot
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    K_values = df['K'].values
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Global AUC Comparison
    ax1.plot(K_values, df['equal_global_auc_mean'], 'o-', linewidth=2.5, markersize=9,
             label='Equal Partition', color='#2E86AB')
    ax1.fill_between(K_values,
                      df['equal_global_auc_mean'] - df['equal_global_auc_std'],
                      df['equal_global_auc_mean'] + df['equal_global_auc_std'],
                      alpha=0.2, color='#2E86AB')
    
    ax1.plot(K_values, df['imbalanced_global_auc_mean'], 's-', linewidth=2.5, markersize=9,
             label='Imbalanced Partition', color='#A23B72')
    ax1.fill_between(K_values,
                      df['imbalanced_global_auc_mean'] - df['imbalanced_global_auc_std'],
                      df['imbalanced_global_auc_mean'] + df['imbalanced_global_auc_std'],
                      alpha=0.2, color='#A23B72')
    
    ax1.set_xlabel('Number of Hospitals (K)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Global AUC (FedAvg)', fontsize=13, fontweight='bold')
    ax1.set_title('Global Model Performance: Equal vs Imbalanced',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Add significance markers
    for i, row in df.iterrows():
        if row['global_auc_pvalue'] < 0.05:
            ax1.text(row['K'], max(row['equal_global_auc_mean'], row['imbalanced_global_auc_mean']) + 0.01,
                    '*', fontsize=16, ha='center', color='red')
    
    # Plot 2: Free-Rider AUC Comparison
    ax2.plot(K_values, df['equal_free_rider_auc_mean'], 'o-', linewidth=2.5, markersize=9,
             label='Equal Partition', color='#F18F01')
    ax2.fill_between(K_values,
                      df['equal_free_rider_auc_mean'] - df['equal_free_rider_auc_std'],
                      df['equal_free_rider_auc_mean'] + df['equal_free_rider_auc_std'],
                      alpha=0.2, color='#F18F01')
    
    ax2.plot(K_values, df['imbalanced_free_rider_auc_mean'], 's-', linewidth=2.5, markersize=9,
             label='Imbalanced Partition', color='#C73E1D')
    ax2.fill_between(K_values,
                      df['imbalanced_free_rider_auc_mean'] - df['imbalanced_free_rider_auc_std'],
                      df['imbalanced_free_rider_auc_mean'] + df['imbalanced_free_rider_auc_std'],
                      alpha=0.2, color='#C73E1D')
    
    ax2.set_xlabel('Number of Hospitals (K)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Free-Rider AUC', fontsize=13, fontweight='bold')
    ax2.set_title('Free-Rider Performance: Equal vs Imbalanced',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Add significance markers
    for i, row in df.iterrows():
        if row['free_rider_auc_pvalue'] < 0.05:
            ax2.text(row['K'], max(row['equal_free_rider_auc_mean'], row['imbalanced_free_rider_auc_mean']) + 0.01,
                    '*', fontsize=16, ha='center', color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Partition comparison plot saved to {save_path}")
    
    return fig


def save_partition_comparison_results(
    comparison_df: pd.DataFrame,
    save_dir: str = 'reports/version3_partition_comparison'
) -> None:
    """
    Save partition comparison results with statistical analysis.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison results from compare_partitions
    save_dir : str
        Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save comparison CSV
    csv_path = os.path.join(save_dir, 'comparison_results.csv')
    comparison_df.to_csv(csv_path, index=False)
    
    # Save statistical test results
    stats_path = os.path.join(save_dir, 'statistical_test_results.txt')
    with open(stats_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PARTITION COMPARISON: STATISTICAL TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write("Paired t-test comparing Equal vs Imbalanced partitions\n")
        f.write("Null hypothesis: No difference in performance\n")
        f.write("Significance level: α = 0.05\n\n")
        f.write("="*70 + "\n\n")
        
        for _, row in comparison_df.iterrows():
            K = int(row['K'])
            f.write(f"K = {K} Hospitals\n")
            f.write("-" * 50 + "\n")
            
            # Global AUC comparison
            f.write(f"Global AUC (FedAvg):\n")
            f.write(f"  Equal:      {row['equal_global_auc_mean']:.4f} ± {row['equal_global_auc_std']:.4f}\n")
            f.write(f"  Imbalanced: {row['imbalanced_global_auc_mean']:.4f} ± {row['imbalanced_global_auc_std']:.4f}\n")
            f.write(f"  Difference: {row['equal_global_auc_mean'] - row['imbalanced_global_auc_mean']:.4f}\n")
            f.write(f"  p-value:    {row['global_auc_pvalue']:.4f}")
            if row['global_auc_pvalue'] < 0.05:
                f.write(" ***SIGNIFICANT***")
            f.write("\n\n")
            
            # Free-rider AUC comparison
            f.write(f"Free-Rider AUC:\n")
            f.write(f"  Equal:      {row['equal_free_rider_auc_mean']:.4f} ± {row['equal_free_rider_auc_std']:.4f}\n")
            f.write(f"  Imbalanced: {row['imbalanced_free_rider_auc_mean']:.4f} ± {row['imbalanced_free_rider_auc_std']:.4f}\n")
            f.write(f"  Difference: {row['equal_free_rider_auc_mean'] - row['imbalanced_free_rider_auc_mean']:.4f}\n")
            f.write(f"  p-value:    {row['free_rider_auc_pvalue']:.4f}")
            if row['free_rider_auc_pvalue'] < 0.05:
                f.write(" ***SIGNIFICANT***")
            f.write("\n\n")
            f.write("="*70 + "\n\n")
        
        # Summary interpretation
        f.write("\nRESEARCH INTERPRETATION:\n")
        f.write("-" * 70 + "\n\n")
        
        avg_global_diff = (comparison_df['equal_global_auc_mean'] - comparison_df['imbalanced_global_auc_mean']).mean()
        avg_fr_diff = (comparison_df['equal_free_rider_auc_mean'] - comparison_df['imbalanced_free_rider_auc_mean']).mean()
        
        f.write(f"1. Global AUC Impact:\n")
        f.write(f"   Average difference: {avg_global_diff:.4f}\n")
        if avg_global_diff > 0:
            f.write(f"   → Equal partition performs BETTER on average\n")
            f.write(f"   → Data heterogeneity degrades federated performance\n")
        else:
            f.write(f"   → Imbalanced partition performs BETTER on average\n")
            f.write(f"   → Surprising result - investigate further\n")
        f.write("\n")
        
        f.write(f"2. Free-Rider Impact:\n")
        f.write(f"   Average difference: {avg_fr_diff:.4f}\n")
        if avg_fr_diff > 0:
            f.write(f"   → Free-riders benefit MORE from equal partitions\n")
            f.write(f"   → Imbalanced data reduces free-rider advantage\n")
        else:
            f.write(f"   → Free-riders benefit MORE from imbalanced partitions\n")
        f.write("\n")
        
        f.write(f"3. Implications:\n")
        f.write(f"   - Data heterogeneity is a key challenge in federated learning\n")
        f.write(f"   - FedProx or personalized FL may be needed for imbalanced scenarios\n")
        f.write(f"   - Free-rider problem persists but is affected by data distribution\n")
        f.write("\n")
    
    print(f"\nPartition comparison results saved:")
    print(f"  - {csv_path}")
    print(f"  - {stats_path}")


    print(f"\nPartition comparison results saved:")
    print(f"  - {csv_path}")
    print(f"  - {stats_path}")


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
