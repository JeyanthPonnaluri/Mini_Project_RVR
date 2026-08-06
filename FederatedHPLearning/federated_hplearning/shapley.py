"""
Federated Shapley Value valuation for hospitals.
Quantifies client contribution using game-theoretic Shapley Values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import math
from typing import List, Tuple, Dict
from sklearn.metrics import roc_auc_score
from federated import fedavg_train, fedprox_train
from logistic_numpy import predict_proba


def compute_federated_shapley_values(
    hospitals: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 30,
    epochs: int = 5,
    lr: float = 0.1,
    algorithm: str = 'fedavg',
    mu: float = 0.1,
    n_permutations: int = 20,
    random_seed: int = 42,
    dp_enabled: bool = False,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clipping_norm: float = 1.0
) -> pd.DataFrame:
    """
    Compute Federated Shapley Values for hospitals using Monte Carlo permutation approach.
    Includes caching of coalition utilities to avoid redundant training.
    
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
    n_permutations : int
        Number of permutations to sample for Monte Carlo Shapley
    random_seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Shapley values analysis with columns:
        - hospital_id
        - num_samples
        - baseline_auc
        - shapley_value
        - shapley_value_pct
    """
    print(f"\n{'='*70}")
    print(f"FEDERATED SHAPLEY VALUE VALUATION")
    print(f"{'='*70}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Total hospitals: {len(hospitals)}")
    print(f"Rounds: {rounds}, Epochs: {epochs}, LR: {lr}")
    print(f"{'='*70}\n")
    
    K = len(hospitals)
    client_ids = list(range(K))
    
    # Cache to store coalition utilities: tuple(sorted(coalition_ids)) -> AUC
    utility_cache = {}
    
    # Utility function v(S)
    def get_utility(coalition: Tuple[int, ...]) -> float:
        if not coalition:
            return 0.5  # Random guessing baseline for AUC
            
        key = tuple(sorted(coalition))
        if key in utility_cache:
            return utility_cache[key]
            
        # Get subset of hospitals in coalition
        subset_hospitals = [hospitals[i] for i in coalition]
        
        # Train federated model
        if algorithm.lower() == 'fedavg':
            res = fedavg_train(
                subset_hospitals, X_test, y_test,
                rounds=rounds, epochs=epochs, lr=lr,
                random_seed=random_seed,
                dp_enabled=dp_enabled, epsilon=epsilon, delta=delta, clipping_norm=clipping_norm
            )
        else:  # fedprox
            res = fedprox_train(
                subset_hospitals, X_test, y_test,
                rounds=rounds, epochs=epochs, lr=lr, mu=mu,
                random_seed=random_seed,
                dp_enabled=dp_enabled, epsilon=epsilon, delta=delta, clipping_norm=clipping_norm
            )
            
        auc = res['round_aucs'][-1]
        utility_cache[key] = auc
        return auc

    # Determine permutations
    all_perms_possible = math.factorial(K)
    
    if all_perms_possible <= n_permutations:
        # Run exact Shapley Value by iterating over all permutations
        permutations = list(itertools.permutations(client_ids))
        print(f"Running exact Shapley Value over all {len(permutations)} permutations...")
    else:
        # Sample permutations
        permutations = []
        seen = set()
        for _ in range(n_permutations):
            # Try to get a unique permutation
            for _ in range(100):  # limit retries
                p = tuple(np.random.permutation(client_ids))
                if p not in seen:
                    seen.add(p)
                    permutations.append(p)
                    break
        print(f"Running permutation-based Shapley Value over {len(permutations)} sampled permutations...")

    # Compute marginal contributions
    marginal_contribs = {i: [] for i in range(K)}
    
    for idx, perm in enumerate(permutations):
        # We walk through the permutation and calculate marginal utilities
        running_coalition = []
        prev_utility = get_utility(tuple(running_coalition))
        
        for client in perm:
            running_coalition.append(client)
            current_utility = get_utility(tuple(running_coalition))
            marginal_contrib = current_utility - prev_utility
            marginal_contribs[client].append(marginal_contrib)
            prev_utility = current_utility
            
    # Calculate Shapley values (average marginal contribution)
    shapley_values = []
    baseline_auc = get_utility(tuple(client_ids))
    
    for i in range(K):
        sv = np.mean(marginal_contribs[i])
        num_samples = len(hospitals[i][1])
        shapley_values.append({
            'hospital_id': i + 1,
            'num_samples': num_samples,
            'baseline_auc': baseline_auc,
            'shapley_value': sv,
            'shapley_value_pct': (sv / baseline_auc) * 100 if baseline_auc > 0 else 0
        })
        
    df = pd.DataFrame(shapley_values)
    
    print(f"{'='*70}")
    print(f"SHAPLEY VALUE VALUATION COMPLETE")
    print(f"{'='*70}\n")
    
    return df


def plot_shapley_comparison(
    shapley_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot comparison between Leave-One-Out (LOO) contribution and Shapley Values.
    
    Parameters:
    -----------
    shapley_df : pd.DataFrame
        Shapley values dataframe
    contribution_df : pd.DataFrame
        Leave-one-out contribution dataframe
    save_path : str
        Path to save plot
        
    Returns:
    --------
    plt.Figure
        Comparison figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Merge datasets on hospital_id for comparison
    merged = pd.merge(
        shapley_df[['hospital_id', 'num_samples', 'shapley_value']],
        contribution_df[['hospital_id', 'contribution']],
        on='hospital_id'
    )
    
    hospital_ids = merged['hospital_id'].values
    svs = merged['shapley_value'].values
    loos = merged['contribution'].values
    num_samples = merged['num_samples'].values
    
    # Plot 1: Side-by-side bar chart
    x = np.arange(len(hospital_ids))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, svs, width, label='Shapley Value', color='#2E86AB', alpha=0.8, edgecolor='black')
    rects2 = ax1.bar(x + width/2, loos, width, label='Leave-One-Out', color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Hospital ID', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Valuation Score (ΔAUC)', fontsize=13, fontweight='bold')
    ax1.set_title('Hospital Valuation Comparison: Shapley vs LOO', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Hospital {hid}" for hid in hospital_ids])
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    
    # Plot 2: Scatter plot of Valuation vs Client Size
    ax2.scatter(num_samples, svs, s=150, color='#2E86AB', alpha=0.7, edgecolor='black', label='Shapley Value')
    ax2.scatter(num_samples, loos, s=150, color='#A23B72', alpha=0.7, edgecolor='black', marker='s', label='Leave-One-Out')
    
    for i, txt in enumerate(hospital_ids):
        ax2.annotate(f'H{txt}', (num_samples[i], svs[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        ax2.annotate(f'H{txt}', (num_samples[i], loos[i]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
        
    ax2.set_xlabel('Number of Samples (Hospital Size)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Valuation Score (ΔAUC)', fontsize=13, fontweight='bold')
    ax2.set_title('Valuation Scores vs Hospital Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Shapley comparison plot saved to {save_path}")
        
    return fig
