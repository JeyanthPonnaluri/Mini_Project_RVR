"""
Federated learning implementation using FedAvg algorithm.
Implements hospital data partitioning and federated averaging.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .logistic_numpy import (
    initialize_weights,
    local_train,
    compute_loss,
    predict_proba,
    compute_gradient,
    local_train_dp,
    local_train_fedprox_dp
)


def partition_equal(
    X: np.ndarray,
    y: np.ndarray,
    num_hospitals: int,
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition dataset equally across hospitals with stratification.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Labels of shape (n_samples,)
    num_hospitals : int
        Number of hospitals to partition data into
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (X_k, y_k) tuples for each hospital
    """
    np.random.seed(random_seed)
    
    # Convert to numpy arrays if needed
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    n_samples = len(y)
    samples_per_hospital = n_samples // num_hospitals
    
    # Create stratified indices
    indices = np.arange(n_samples)
    
    # Shuffle with stratification
    class_0_indices = indices[y == 0]
    class_1_indices = indices[y == 1]
    
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    hospitals = []
    
    for k in range(num_hospitals):
        # Calculate samples per class for this hospital
        n_class_0 = len(class_0_indices) // num_hospitals
        n_class_1 = len(class_1_indices) // num_hospitals
        
        # Get indices for this hospital
        start_0 = k * n_class_0
        end_0 = (k + 1) * n_class_0 if k < num_hospitals - 1 else len(class_0_indices)
        
        start_1 = k * n_class_1
        end_1 = (k + 1) * n_class_1 if k < num_hospitals - 1 else len(class_1_indices)
        
        hospital_indices = np.concatenate([
            class_0_indices[start_0:end_0],
            class_1_indices[start_1:end_1]
        ])
        
        # Shuffle hospital indices
        np.random.shuffle(hospital_indices)
        
        X_k = X[hospital_indices]
        y_k = y[hospital_indices]
        
        hospitals.append((X_k, y_k))
        
        print(f"Hospital {k+1}: {len(y_k)} samples, class distribution: {dict(zip(*np.unique(y_k, return_counts=True)))}")
    
    return hospitals


# Default imbalanced distribution (long-tail)
DEFAULT_IMBALANCED_DISTRIBUTION = [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]


def normalize_distribution(distribution: List[float]) -> List[float]:
    """
    Normalize distribution to sum to 1.0.
    
    Parameters:
    -----------
    distribution : List[float]
        Distribution values
        
    Returns:
    --------
    List[float]
        Normalized distribution
    """
    total = sum(distribution)
    if total == 0:
        raise ValueError("Distribution sum cannot be zero")
    return [d / total for d in distribution]


def generate_imbalanced_distribution(K: int, random_seed: int = 42) -> List[float]:
    """
    Generate imbalanced distribution for K hospitals.
    Uses default distribution if K <= 7, otherwise generates long-tail.
    
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
    if K <= len(DEFAULT_IMBALANCED_DISTRIBUTION):
        # Use truncated default distribution
        distribution = DEFAULT_IMBALANCED_DISTRIBUTION[:K]
        return normalize_distribution(distribution)
    else:
        # Generate long-tail distribution
        np.random.seed(random_seed)
        weights = np.random.exponential(scale=1.0, size=K)
        # Sort descending for long-tail effect
        weights = np.sort(weights)[::-1]
        return normalize_distribution(weights.tolist())


def partition_imbalanced(
    X: np.ndarray,
    y: np.ndarray,
    distribution: List[float],
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition dataset with imbalanced distribution across hospitals.
    Maintains stratified class distribution within each hospital.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Labels of shape (n_samples,)
    distribution : List[float]
        Distribution of samples per hospital
        Will be automatically normalized if sum ≠ 1.0
        Example: [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (X_k, y_k) tuples for each hospital
        
    Raises:
    -------
    ValueError
        If distribution is empty or contains invalid values
    """
    np.random.seed(random_seed)
    
    # Convert to numpy arrays if needed
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # Validate and normalize distribution
    if len(distribution) == 0:
        raise ValueError("Distribution cannot be empty")
    
    if any(d < 0 for d in distribution):
        raise ValueError("Distribution values must be non-negative")
    
    # Automatically normalize if sum ≠ 1.0
    if not np.isclose(sum(distribution), 1.0):
        print(f"  Note: Distribution sum = {sum(distribution):.4f}, normalizing to 1.0")
        distribution = normalize_distribution(distribution)
    
    num_hospitals = len(distribution)
    n_samples = len(y)
    
    # Create stratified indices
    indices = np.arange(n_samples)
    
    # Shuffle with stratification
    class_0_indices = indices[y == 0]
    class_1_indices = indices[y == 1]
    
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    hospitals = []
    
    start_0 = 0
    start_1 = 0
    
    for k in range(num_hospitals):
        # Calculate samples for this hospital based on distribution
        n_samples_k = int(n_samples * distribution[k])
        
        # Calculate samples per class (maintain stratification)
        n_class_0_k = int(len(class_0_indices) * distribution[k])
        n_class_1_k = int(len(class_1_indices) * distribution[k])
        
        # Handle last hospital (take remaining samples)
        if k == num_hospitals - 1:
            end_0 = len(class_0_indices)
            end_1 = len(class_1_indices)
        else:
            end_0 = start_0 + n_class_0_k
            end_1 = start_1 + n_class_1_k
        
        # Get indices for this hospital
        hospital_indices = np.concatenate([
            class_0_indices[start_0:end_0],
            class_1_indices[start_1:end_1]
        ])
        
        # Shuffle hospital indices
        np.random.shuffle(hospital_indices)
        
        X_k = X[hospital_indices]
        y_k = y[hospital_indices]
        
        hospitals.append((X_k, y_k))
        
        print(f"Hospital {k+1}: {len(y_k)} samples ({distribution[k]*100:.1f}%), class distribution: {dict(zip(*np.unique(y_k, return_counts=True)))}")
        
        start_0 = end_0
        start_1 = end_1
    
    return hospitals


def fedavg_train(
    hospitals: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int,
    epochs: int,
    lr: float,
    random_seed: int = 42,
    dp_enabled: bool = False,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clipping_norm: float = 1.0,
    dropout_rate: float = 0.0,
    bandwidth_mbps: float = 10.0,
    latency_ms: float = 50.0
) -> Dict:
    """
    Train federated model using FedAvg algorithm with client dropouts and virtual latency tracking.
    
    Parameters:
    -----------
    hospitals : List[Tuple[np.ndarray, np.ndarray]]
        List of (X_k, y_k) for each hospital
    X_test : np.ndarray
        Test features for evaluation
    y_test : np.ndarray
        Test labels for evaluation
    rounds : int
        Number of communication rounds
    epochs : int
        Number of local training epochs per round
    lr : float
        Learning rate
    random_seed : int
        Random seed for reproducibility
    dp_enabled : bool
        Whether to enable Differential Privacy
    epsilon : float
        Differential privacy budget epsilon
    delta : float
        Differential privacy parameter delta
    clipping_norm : float
        Differential privacy gradient clipping norm
    dropout_rate : float
        Rate of random client dropouts per round (0.0 to 1.0)
    bandwidth_mbps : float
        Network bandwidth in Mbps
    latency_ms : float
        Network latency in milliseconds
        
    Returns:
    --------
    Dict
        Dictionary containing:
        - w_global: Final global weights
        - round_losses: List of global losses per round
        - round_aucs: List of global AUCs per round
        - round_metrics: Detailed metrics per round
        - cumulative_bytes: Cumulative bytes transferred
        - cumulative_time_s: Cumulative virtual time in seconds
    """
    # Get number of features from first hospital
    n_features = hospitals[0][0].shape[1]
    
    # Initialize global weights
    w_global = initialize_weights(n_features, random_seed)
    
    print(f"\n{'='*60}")
    print(f"FEDERATED LEARNING - FedAvg")
    print(f"{'='*60}")
    print(f"Number of hospitals: {len(hospitals)}")
    print(f"Communication rounds: {rounds}")
    print(f"Local epochs per round: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Client dropout rate: {dropout_rate}")
    print(f"Network: {bandwidth_mbps} Mbps, {latency_ms} ms latency")
    print(f"{'='*60}\n")
    
    round_losses = []
    round_aucs = []
    round_metrics = []
    
    # Trackers for virtual latency and communication
    cumulative_bytes = []
    cumulative_time_s = []
    total_bytes = 0
    total_time = 0.0
    
    # Pre-calculate network bandwidth and latency in standard units
    bandwidth_bps = bandwidth_mbps * 1e6
    latency_s = latency_ms / 1000.0
    
    for round_idx in range(rounds):
        # 1. Simulate client dropouts
        np.random.seed(random_seed + round_idx * 100)
        active_indices = []
        for k in range(len(hospitals)):
            if np.random.rand() >= dropout_rate:
                active_indices.append(k)
        
        # Ensure at least one client remains active to make progress
        if len(active_indices) == 0:
            active_indices = [np.random.randint(len(hospitals))]
            
        # Re-normalize aggregation weights among active clients
        active_samples = sum(len(hospitals[idx][1]) for idx in active_indices)
        active_weights = {idx: len(hospitals[idx][1]) / active_samples for idx in active_indices}
        
        # Store local weights from active hospitals
        local_weights = {}
        local_losses = []
        
        # 2. Local Training for active clients
        for k in active_indices:
            X_k, y_k = hospitals[k]
            # Local training
            if dp_enabled:
                w_k, loss_history = local_train_dp(
                    X_k, y_k, w_global, epochs, lr,
                    epsilon=epsilon, delta=delta, clipping_norm=clipping_norm,
                    random_seed=random_seed + k + round_idx * 10
                )
            else:
                w_k, loss_history = local_train(X_k, y_k, w_global, epochs, lr)
            
            local_weights[k] = w_k
            local_losses.append(loss_history[-1])  # Last epoch loss
        
        # 3. Weighted Aggregation (FedAvg over active clients)
        w_global_new = np.zeros_like(w_global)
        for k, w_k in local_weights.items():
            w_global_new += active_weights[k] * w_k
        w_global = w_global_new
        
        # 4. Compute Communication Costs & Virtual Latency
        # Bytes transferred: (download weight + upload weight) * active clients
        # w_global size is n_features, float64 elements are 8 bytes each
        round_bytes = 2 * len(active_indices) * n_features * 8
        total_bytes += round_bytes
        cumulative_bytes.append(total_bytes)
        
        # Virtual round duration: local training time + download time + upload time + roundtrip latency
        local_train_time_s = epochs * 0.02  # Estimate 20ms per local epoch
        net_tx_time_s = (2 * n_features * 64) / bandwidth_bps  # 64 bits per weight element
        net_latency_s = 2 * latency_s  # one roundtrip
        round_time_s = local_train_time_s + net_tx_time_s + net_latency_s
        total_time += round_time_s
        cumulative_time_s.append(total_time)
        
        # Evaluate global model on test set
        test_loss = compute_loss(X_test, y_test, w_global)
        y_pred_proba = predict_proba(X_test, w_global)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        round_losses.append(test_loss)
        round_aucs.append(test_auc)
        
        # Store detailed metrics
        round_metrics.append({
            'round': round_idx + 1,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'avg_local_loss': np.mean(local_losses),
            'active_clients': len(active_indices),
            'round_bytes': round_bytes,
            'cumulative_bytes': total_bytes,
            'round_time_s': round_time_s,
            'cumulative_time_s': total_time
        })
        
        # Print progress every 10 rounds
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            print(f"Round {round_idx + 1}/{rounds} - Active Clients: {len(active_indices)} - Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, CumTime: {total_time:.2f}s")
    
    print(f"\n{'='*60}")
    print(f"FedAvg Training Complete!")
    print(f"Final Test AUC: {round_aucs[-1]:.4f}")
    print(f"Total Bytes Transferred: {total_bytes / (1024*1024):.2f} MB")
    print(f"Total Virtual Time: {total_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    return {
        'w_global': w_global,
        'round_losses': round_losses,
        'round_aucs': round_aucs,
        'round_metrics': round_metrics,
        'cumulative_bytes': cumulative_bytes,
        'cumulative_time_s': cumulative_time_s
    }


def train_local_models(
    hospitals: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    random_seed: int = 42
) -> List[float]:
    """
    Train separate local models for each hospital (no federation).
    
    Parameters:
    -----------
    hospitals : List[Tuple[np.ndarray, np.ndarray]]
        List of (X_k, y_k) for each hospital
    X_test : np.ndarray
        Test features for evaluation
    y_test : np.ndarray
        Test labels for evaluation
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    List[float]
        List of AUC scores for each hospital's local model
    """
    n_features = hospitals[0][0].shape[1]
    local_aucs = []
    
    print(f"\n{'='*60}")
    print(f"TRAINING LOCAL MODELS (No Federation)")
    print(f"{'='*60}\n")
    
    for k, (X_k, y_k) in enumerate(hospitals):
        # Initialize weights for this hospital
        w_k = initialize_weights(n_features, random_seed + k)
        
        # Train locally
        w_k, loss_history = local_train(X_k, y_k, w_k, epochs, lr)
        
        # Evaluate on test set
        y_pred_proba = predict_proba(X_test, w_k)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        local_aucs.append(auc)
        
        print(f"Hospital {k+1} - Local AUC: {auc:.4f}")
    
    avg_local_auc = np.mean(local_aucs)
    print(f"\nAverage Local AUC: {avg_local_auc:.4f}")
    print(f"{'='*60}\n")
    
    return local_aucs



def partition_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_hospitals: int,
    alpha: float = 0.5,
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition dataset using Dirichlet distribution for non-IID simulation.
    
    Lower alpha → stronger non-IID (heterogeneous class distributions)
    Higher alpha → closer to IID (homogeneous class distributions)
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Labels of shape (n_samples,)
    num_hospitals : int
        Number of hospitals to partition data into
    alpha : float
        Dirichlet concentration parameter
        - alpha < 1: Strong non-IID (heterogeneous)
        - alpha = 1: Moderate non-IID
        - alpha > 10: Nearly IID (homogeneous)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (X_k, y_k) tuples for each hospital
    """
    np.random.seed(random_seed)
    
    # Convert to numpy arrays if needed
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = len(classes)
    
    print(f"\n{'='*60}")
    print(f"DIRICHLET NON-IID PARTITION")
    print(f"{'='*60}")
    print(f"Alpha: {alpha} ({'Strong non-IID' if alpha < 1 else 'Moderate' if alpha < 5 else 'Nearly IID'})")
    print(f"Hospitals: {num_hospitals}")
    print(f"Classes: {n_classes}")
    print(f"{'='*60}\n")
    
    # Initialize hospital assignments
    hospital_indices = [[] for _ in range(num_hospitals)]
    
    # For each class, sample proportions from Dirichlet and assign samples
    for c in classes:
        # Get indices of samples belonging to this class
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_hospitals)
        
        # Ensure minimum samples per hospital (at least 1 sample per hospital if possible)
        min_samples_per_hospital = max(1, len(class_indices) // (num_hospitals * 10))
        
        # Assign samples to hospitals based on proportions
        proportions = (proportions * len(class_indices)).astype(int)
        
        # Ensure each hospital gets at least min_samples if available
        for k in range(num_hospitals):
            if proportions[k] < min_samples_per_hospital and len(class_indices) >= num_hospitals * min_samples_per_hospital:
                proportions[k] = min_samples_per_hospital
        
        # Adjust last proportion to ensure all samples are assigned
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        # If last proportion is negative, redistribute
        if proportions[-1] < 0:
            # Recalculate without minimum constraint
            proportions = np.random.dirichlet([alpha] * num_hospitals)
            proportions = (proportions * len(class_indices)).astype(int)
            proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        # Distribute samples
        start_idx = 0
        for k in range(num_hospitals):
            end_idx = start_idx + int(proportions[k])
            if end_idx > start_idx:  # Only add if there are samples
                hospital_indices[k].extend(class_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
    
    # Create hospital datasets
    hospitals = []
    skipped_count = 0
    for k in range(num_hospitals):
        indices = np.array(hospital_indices[k], dtype=np.int64)
        
        # Skip empty hospitals
        if len(indices) == 0:
            print(f"Hospital {k+1}: 0 samples - SKIPPED (empty)")
            skipped_count += 1
            continue
        
        np.random.shuffle(indices)
        
        X_k = X[indices]
        y_k = y[indices]
        
        # Calculate class distribution
        unique, counts = np.unique(y_k, return_counts=True)
        class_dist = dict(zip(unique, counts))
        class_ratios = {cls: counts[i] / len(y_k) for i, cls in enumerate(unique)}
        
        hospitals.append((X_k, y_k))
        
        print(f"Hospital {len(hospitals)}: {len(y_k)} samples, class dist: {class_dist}, ratios: {class_ratios}")
    
    if skipped_count > 0:
        print(f"\n[WARNING] Warning: {skipped_count} hospital(s) had 0 samples and were skipped")
        print(f"Active hospitals: {len(hospitals)} (requested: {num_hospitals})")
    print()
    return hospitals


def local_train_fedprox(
    X: np.ndarray,
    y: np.ndarray,
    w_global: np.ndarray,
    epochs: int,
    lr: float,
    mu: float
) -> Tuple[np.ndarray, list]:
    """
    Train logistic regression locally using FedProx (with proximal term).
    
    FedProx adds a proximal term to prevent local models from drifting too far:
    L_k(w) = cross_entropy(w) + (mu/2) * ||w - w_global||^2
    
    Gradient: grad_CE + mu * (w - w_global)
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        True labels of shape (n_samples,)
    w_global : np.ndarray
        Global weights from server (proximal center)
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    mu : float
        Proximal term coefficient (regularization strength)
        - mu = 0: Equivalent to FedAvg
        - mu > 0: Prevents drift from w_global
        
    Returns:
    --------
    tuple
        (w_final, loss_history)
        - w_final: Trained weights
        - loss_history: List of losses per epoch (including proximal term)
    """
    w = w_global.copy()
    loss_history = []
    
    for epoch in range(epochs):
        # Compute standard gradient
        grad_ce = compute_gradient(X, y, w)
        
        # Add proximal term gradient: mu * (w - w_global)
        grad_prox = mu * (w - w_global)
        
        # Total gradient
        grad = grad_ce + grad_prox
        
        # Update weights
        w = w - lr * grad
        
        # Compute loss (cross-entropy + proximal term)
        loss_ce = compute_loss(X, y, w)
        loss_prox = (mu / 2.0) * np.sum((w - w_global) ** 2)
        loss = loss_ce + loss_prox
        
        loss_history.append(loss)
    
    return w, loss_history


def fedprox_train(
    hospitals: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int,
    epochs: int,
    lr: float,
    mu: float,
    random_seed: int = 42,
    dp_enabled: bool = False,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clipping_norm: float = 1.0,
    dropout_rate: float = 0.0,
    bandwidth_mbps: float = 10.0,
    latency_ms: float = 50.0
) -> Dict:
    """
    Train federated model using FedProx algorithm with client dropouts and virtual latency tracking.
    
    Parameters:
    -----------
    hospitals : List[Tuple[np.ndarray, np.ndarray]]
        List of (X_k, y_k) for each hospital
    X_test : np.ndarray
        Test features for evaluation
    y_test : np.ndarray
        Test labels for evaluation
    rounds : int
        Number of communication rounds
    epochs : int
        Number of local training epochs per round
    lr : float
        Learning rate
    mu : float
        Proximal term coefficient
        - mu = 0: Equivalent to FedAvg
        - mu > 0: Prevents local drift
    random_seed : int
        Random seed for reproducibility
    dp_enabled : bool
        Whether to enable Differential Privacy
    epsilon : float
        Differential privacy budget epsilon
    delta : float
        Differential privacy parameter delta
    clipping_norm : float
        Differential privacy gradient clipping norm
    dropout_rate : float
        Rate of random client dropouts per round (0.0 to 1.0)
    bandwidth_mbps : float
        Network bandwidth in Mbps
    latency_ms : float
        Network latency in milliseconds
        
    Returns:
    --------
    Dict
        Dictionary containing:
        - w_global: Final global weights
        - round_losses: List of global losses per round
        - round_aucs: List of global AUCs per round
        - round_metrics: Detailed metrics per round
        - weight_drifts: L2 norm of weight change per round
        - cumulative_bytes: Cumulative bytes transferred
        - cumulative_time_s: Cumulative virtual time in seconds
    """
    # Get number of features from first hospital
    n_features = hospitals[0][0].shape[1]
    
    # Initialize global weights
    w_global = initialize_weights(n_features, random_seed)
    
    print(f"\n{'='*60}")
    print(f"FEDERATED LEARNING - FedProx")
    print(f"{'='*60}")
    print(f"Number of hospitals: {len(hospitals)}")
    print(f"Communication rounds: {rounds}")
    print(f"Local epochs per round: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Proximal coefficient (mu): {mu}")
    print(f"Client dropout rate: {dropout_rate}")
    print(f"Network: {bandwidth_mbps} Mbps, {latency_ms} ms latency")
    print(f"{'='*60}\n")
    
    round_losses = []
    round_aucs = []
    round_metrics = []
    weight_drifts = []
    
    # Trackers for virtual latency and communication
    cumulative_bytes = []
    cumulative_time_s = []
    total_bytes = 0
    total_time = 0.0
    
    # Pre-calculate network bandwidth and latency in standard units
    bandwidth_bps = bandwidth_mbps * 1e6
    latency_s = latency_ms / 1000.0
    
    for round_idx in range(rounds):
        w_global_prev = w_global.copy()
        
        # 1. Simulate client dropouts
        np.random.seed(random_seed + round_idx * 100)
        active_indices = []
        for k in range(len(hospitals)):
            if np.random.rand() >= dropout_rate:
                active_indices.append(k)
        
        # Ensure at least one client remains active to make progress
        if len(active_indices) == 0:
            active_indices = [np.random.randint(len(hospitals))]
            
        # Re-normalize aggregation weights among active clients
        active_samples = sum(len(hospitals[idx][1]) for idx in active_indices)
        active_weights = {idx: len(hospitals[idx][1]) / active_samples for idx in active_indices}
        
        # Store local weights from active hospitals
        local_weights = {}
        local_losses = []
        
        # 2. Local Training for active clients
        for k in active_indices:
            X_k, y_k = hospitals[k]
            # Local training with FedProx
            if dp_enabled:
                w_k, loss_history = local_train_fedprox_dp(
                    X_k, y_k, w_global, epochs, lr, mu,
                    epsilon=epsilon, delta=delta, clipping_norm=clipping_norm,
                    random_seed=random_seed + k + round_idx * 10
                )
            else:
                w_k, loss_history = local_train_fedprox(X_k, y_k, w_global, epochs, lr, mu)
            
            local_weights[k] = w_k
            local_losses.append(loss_history[-1])  # Last epoch loss
        
        # 3. Weighted Aggregation (FedProx/FedAvg over active clients)
        w_global_new = np.zeros_like(w_global)
        for k, w_k in local_weights.items():
            w_global_new += active_weights[k] * w_k
        w_global = w_global_new
        
        # Compute weight drift (L2 norm of change)
        drift = np.linalg.norm(w_global - w_global_prev)
        weight_drifts.append(drift)
        
        # 4. Compute Communication Costs & Virtual Latency
        # Bytes transferred: (download weight + upload weight) * active clients
        round_bytes = 2 * len(active_indices) * n_features * 8
        total_bytes += round_bytes
        cumulative_bytes.append(total_bytes)
        
        # Virtual round duration: local training time + download time + upload time + roundtrip latency
        local_train_time_s = epochs * 0.02  # Estimate 20ms per local epoch
        net_tx_time_s = (2 * n_features * 64) / bandwidth_bps  # 64 bits per weight element
        net_latency_s = 2 * latency_s  # one roundtrip
        round_time_s = local_train_time_s + net_tx_time_s + net_latency_s
        total_time += round_time_s
        cumulative_time_s.append(total_time)
        
        # Evaluate global model on test set
        test_loss = compute_loss(X_test, y_test, w_global)
        y_pred_proba = predict_proba(X_test, w_global)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        round_losses.append(test_loss)
        round_aucs.append(test_auc)
        
        # Store detailed metrics
        round_metrics.append({
            'round': round_idx + 1,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'avg_local_loss': np.mean(local_losses),
            'active_clients': len(active_indices),
            'weight_drift': drift,
            'round_bytes': round_bytes,
            'cumulative_bytes': total_bytes,
            'round_time_s': round_time_s,
            'cumulative_time_s': total_time
        })
        
        # Print progress every 10 rounds
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            print(f"Round {round_idx + 1}/{rounds} - Active Clients: {len(active_indices)} - Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Drift: {drift:.4f}, CumTime: {total_time:.2f}s")
    
    print(f"\n{'='*60}")
    print(f"FedProx Training Complete!")
    print(f"Final Test AUC: {round_aucs[-1]:.4f}")
    print(f"Average Weight Drift: {np.mean(weight_drifts):.4f}")
    print(f"Total Bytes Transferred: {total_bytes / (1024*1024):.2f} MB")
    print(f"Total Virtual Time: {total_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    return {
        'w_global': w_global,
        'round_losses': round_losses,
        'round_aucs': round_aucs,
        'round_metrics': round_metrics,
        'weight_drifts': weight_drifts,
        'cumulative_bytes': cumulative_bytes,
        'cumulative_time_s': cumulative_time_s
    }


def fedavg_cox_train(
    hospitals: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    times_test: np.ndarray,
    events_test: np.ndarray,
    rounds: int,
    epochs: int,
    lr: float,
    alpha: float = 0.1,
    random_seed: int = 42,
    dp_enabled: bool = False,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clipping_norm: float = 1.0,
    dropout_rate: float = 0.0,
    bandwidth_mbps: float = 10.0,
    latency_ms: float = 50.0
) -> Dict:
    """
    Train federated Cox Proportional Hazards model using FedAvg.
    """
    from .cox_numpy import local_cox_train, compute_concordance_index
    
    np.random.seed(random_seed)
    n_features = X_test.shape[1]
    w_global = np.zeros(n_features)
    
    round_c_indices = []
    cumulative_bytes = []
    cumulative_time_s = []
    
    total_bytes = 0
    total_time = 0.0
    
    # Pre-sort test data for C-index evaluation
    sort_idx = np.argsort(times_test)
    X_test_sorted = X_test[sort_idx]
    times_test_sorted = times_test[sort_idx]
    events_test_sorted = events_test[sort_idx]
    
    for round_idx in range(rounds):
        # Determine active clients
        active_clients_idx = []
        for idx in range(len(hospitals)):
            if np.random.rand() >= dropout_rate:
                active_clients_idx.append(idx)
                
        # If no clients are active, aggregate nothing (skip round)
        if len(active_clients_idx) == 0:
            round_c_indices.append(round_c_indices[-1] if round_c_indices else 0.5)
            cumulative_bytes.append(total_bytes)
            cumulative_time_s.append(total_time)
            continue
            
        client_weights = []
        client_sizes = []
        
        # Local training on active clients
        for idx in active_clients_idx:
            X_k, times_k, events_k = hospitals[idx]
            w_k = local_cox_train(
                X_k, times_k, events_k, epochs=epochs, lr=lr, alpha=alpha,
                w_global=w_global, mu=0.0,
                dp_enabled=dp_enabled, epsilon=epsilon, delta=delta, clipping_norm=clipping_norm
            )
            client_weights.append(w_k)
            client_sizes.append(len(X_k))
            
        # Re-normalized weighted average aggregation
        total_size = sum(client_sizes)
        w_global_new = np.zeros_like(w_global)
        for w_k, size_k in zip(client_weights, client_sizes):
            w_global_new += (size_k / total_size) * w_k
            
        w_global = w_global_new
        
        # Calculate C-index on global test set
        c_index = compute_concordance_index(w_global, X_test_sorted, times_test_sorted, events_test_sorted)
        round_c_indices.append(c_index)
        
        # Compute network overhead
        round_bytes = 2 * len(active_clients_idx) * n_features * 8
        total_bytes += round_bytes
        cumulative_bytes.append(total_bytes)
        
        # Compute virtual latency (slowest client download/upload + local epoch time)
        transmission_time = (2 * n_features * 64) / (bandwidth_mbps * 1e6) + 2 * (latency_ms / 1000.0)
        computation_time = epochs * 0.03  # Assume ~0.03 seconds per Cox epoch
        round_time = computation_time + transmission_time
        total_time += round_time
        cumulative_time_s.append(total_time)
        
    return {
        'w_global': w_global,
        'round_c_indices': round_c_indices,
        'cumulative_bytes': cumulative_bytes,
        'cumulative_time_s': cumulative_time_s
    }


def evaluate_personalized_fl(hospitals, w_global, epochs=2, lr=0.1, random_seed=42):
    """
    Evaluate Personalized Federated Learning (PFL).
    Each client receives the global consensus model weights, fine-tunes them for
    a few local epochs on their training split, and evaluates performance on their local test split.
    """
    from logistic_numpy import local_train, predict_proba
    from sklearn.metrics import roc_auc_score
    
    np.random.seed(random_seed)
    local_aucs_before = []
    local_aucs_after = []
    
    for idx, (X, y) in enumerate(hospitals):
        n = len(X)
        if n < 10:
            continue
            
        # Split local client data 80/20
        split_idx = int(0.8 * n)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(np.unique(y_test)) < 2:
            continue
            
        # Calculate local AUC before personalization
        probs_before = predict_proba(X_test, w_global)
        auc_before = roc_auc_score(y_test, probs_before)
        local_aucs_before.append(auc_before)
        
        # Fine-tune local model on client's local training data
        w_p, _ = local_train(X_train, y_train, w_init=w_global, epochs=epochs, lr=lr)
        
        # Calculate local AUC after personalization
        probs_after = predict_proba(X_test, w_p)
        auc_after = roc_auc_score(y_test, probs_after)
        local_aucs_after.append(auc_after)
        
    return {
        'local_aucs_before': local_aucs_before,
        'local_aucs_after': local_aucs_after,
        'mean_auc_before': np.mean(local_aucs_before) if local_aucs_before else 0.5,
        'mean_auc_after': np.mean(local_aucs_after) if local_aucs_after else 0.5
    }


def compute_rdp_privacy_budget(q: float, epsilon_local: float, T: int, delta: float = 1e-5) -> float:
    """
    Compute total composed privacy budget (Epsilon total) using RDP composition bounds.
    
    Parameters:
    -----------
    q : float
        Client sampling rate (active clients / total clients)
    epsilon_local : float
        DP budget epsilon per client update in each round
    T : int
        Total number of communication rounds
    delta : float, default=1e-5
        Failure probability
    """
    if T <= 0 or q <= 0:
        return 0.0
    epsilon_total = q * epsilon_local * np.sqrt(T * np.log(1.0 / delta))
    return float(epsilon_total)
