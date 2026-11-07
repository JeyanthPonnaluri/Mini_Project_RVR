import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics import roc_auc_score
from federated import partition_dirichlet, fedavg_train, fedprox_train
from shapley import compute_federated_shapley_values
from cox_numpy import compute_concordance_index, local_cox_train

def compute_bootstrap_ci_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 100,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Computes Bootstrap Confidence Interval for binary classification AUC.
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # Sample indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            # Skip if only one class is sampled
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
        
    bootstrapped_scores = np.array(bootstrapped_scores)
    if len(bootstrapped_scores) == 0:
        return 0.5, 0.5, np.array([0.5])
        
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = np.percentile(bootstrapped_scores, lower_percentile)
    upper_bound = np.percentile(bootstrapped_scores, upper_percentile)
    
    return float(lower_bound), float(upper_bound), bootstrapped_scores

def compute_bootstrap_ci_cindex(
    w: np.ndarray,
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n_bootstraps: int = 100,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Computes Bootstrap Confidence Interval for Cox model concordance index.
    """
    np.random.seed(random_seed)
    n_samples = len(times)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        score = compute_concordance_index(w, X[indices], times[indices], events[indices])
        bootstrapped_scores.append(score)
        
    bootstrapped_scores = np.array(bootstrapped_scores)
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = np.percentile(bootstrapped_scores, lower_percentile)
    upper_bound = np.percentile(bootstrapped_scores, upper_percentile)
    
    return float(lower_bound), float(upper_bound), bootstrapped_scores

def run_shapley_stability_analysis(
    hospitals: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 10,
    epochs: int = 3,
    lr: float = 0.1,
    n_seeds: int = 5
) -> pd.DataFrame:
    """
    Evaluates Shapley Value stability by running calculations across multiple seeds.
    """
    num_hospitals = len(hospitals)
    seed_results = []
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        # Run Shapley calculation with small permutations to run fast
        res_df = compute_federated_shapley_values(
            hospitals, X_test, y_test,
            rounds=rounds, epochs=epochs, lr=lr,
            algorithm='fedavg', n_permutations=10, random_seed=seed
        )
        # Store shapley value per hospital
        seed_results.append(res_df['shapley_value'].values)
        
    seed_results = np.array(seed_results) # shape: (n_seeds, num_hospitals)
    
    # Calculate mean and standard deviation per hospital
    means = np.mean(seed_results, axis=0)
    stds = np.std(seed_results, axis=0)
    
    summary_df = pd.DataFrame({
        'Hospital ID': [f"Hospital {i+1}" for i in range(num_hospitals)],
        'Mean Shapley Value': means,
        'Std Dev (across seeds)': stds,
        'COV (%)': (stds / (np.abs(means) + 1e-9)) * 100.0
    })
    
    return summary_df

def run_dirichlet_heterogeneity_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_hospitals: int = 3,
    alphas: List[float] = [10.0, 1.0, 0.5, 0.1],
    rounds: int = 15,
    epochs: int = 3,
    lr: float = 0.1,
    mu: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Sweeps over Dirichlet parameter alphas to compare FedAvg and FedProx convergence stability and weight drift.
    """
    sweep_data = []
    
    for alpha in alphas:
        # Partition data
        hospitals = partition_dirichlet(X_train, y_train, num_hospitals, alpha, random_seed)
        
        # Run FedAvg
        res_fedavg = fedavg_train(
            hospitals, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr, random_seed=random_seed
        )
        
        # Run FedProx
        res_fedprox = fedprox_train(
            hospitals, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr, mu=mu, random_seed=random_seed
        )
        
        # Log FedAvg results
        sweep_data.append({
            'Alpha': alpha,
            'Algorithm': 'FedAvg',
            'Final AUC': res_fedavg['round_aucs'][-1],
            'Convergence Std (last 5)': np.std(res_fedavg['round_aucs'][-5:]),
            'Avg Weight Drift': np.mean([m['weight_drift'] for m in res_fedavg['round_metrics']]) if 'weight_drift' in res_fedavg['round_metrics'][0] else 0.0
        })
        
        # Log FedProx results
        sweep_data.append({
            'Alpha': alpha,
            'Algorithm': 'FedProx',
            'Final AUC': res_fedprox['round_aucs'][-1],
            'Convergence Std (last 5)': np.std(res_fedprox['round_aucs'][-5:]),
            'Avg Weight Drift': np.mean(res_fedprox['weight_drifts'])
        })
        
    return pd.DataFrame(sweep_data)
