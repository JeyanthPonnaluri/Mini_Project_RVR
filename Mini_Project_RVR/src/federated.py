"""
Federated learning implementation using FedAvg algorithm.
Implements hospital data partitioning and federated averaging.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from logistic_numpy import (
    initialize_weights,
    local_train,
    compute_loss,
    predict_proba,
    compute_gradient
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
    random_seed: int = 42
) -> Dict:
    """
    Train federated model using FedAvg algorithm.
    
    FedAvg Algorithm:
    1. Initialize global weights w_global
    2. For each communication round:
       a. Send w_global to all hospitals
       b. Each hospital trains locally: w_k = local_train(X_k, y_k, w_global, epochs, lr)
       c. Aggregate: w_global = weighted_average(w_k) weighted by number of samples
    3. Return final w_global
    
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
        
    Returns:
    --------
    Dict
        Dictionary containing:
        - w_global: Final global weights
        - round_losses: List of global losses per round
        - round_aucs: List of global AUCs per round
        - round_metrics: Detailed metrics per round
    """
    # Get number of features from first hospital
    n_features = hospitals[0][0].shape[1]
    
    # Initialize global weights
    w_global = initialize_weights(n_features, random_seed)
    
    # Calculate total samples and weights per hospital
    total_samples = sum(len(y_k) for _, y_k in hospitals)
    hospital_weights = [len(y_k) / total_samples for _, y_k in hospitals]
    
    print(f"\n{'='*60}")
    print(f"FEDERATED LEARNING - FedAvg")
    print(f"{'='*60}")
    print(f"Number of hospitals: {len(hospitals)}")
    print(f"Total samples: {total_samples}")
    print(f"Communication rounds: {rounds}")
    print(f"Local epochs per round: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")
    
    round_losses = []
    round_aucs = []
    round_metrics = []
    
    for round_idx in range(rounds):
        # Store local weights from each hospital
        local_weights = []
        local_losses = []
        
        # Each hospital trains locally
        for k, (X_k, y_k) in enumerate(hospitals):
            # Local training
            w_k, loss_history = local_train(X_k, y_k, w_global, epochs, lr)
            
            local_weights.append(w_k)
            local_losses.append(loss_history[-1])  # Last epoch loss
        
        # Aggregate weights using weighted average (FedAvg)
        w_global = np.zeros_like(w_global)
        for k, w_k in enumerate(local_weights):
            w_global += hospital_weights[k] * w_k
        
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
            'avg_local_loss': np.mean(local_losses)
        })
        
        # Print progress every 10 rounds
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            print(f"Round {round_idx + 1}/{rounds} - Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"FedAvg Training Complete!")
    print(f"Final Test AUC: {round_aucs[-1]:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'w_global': w_global,
        'round_losses': round_losses,
        'round_aucs': round_aucs,
        'round_metrics': round_metrics
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
        
        # Assign samples to hospitals based on proportions
        proportions = (proportions * len(class_indices)).astype(int)
        
        # Adjust last proportion to ensure all samples are assigned
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        # Distribute samples
        start_idx = 0
        for k in range(num_hospitals):
            end_idx = start_idx + int(proportions[k])
            hospital_indices[k].extend(class_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
    
    # Create hospital datasets
    hospitals = []
    for k in range(num_hospitals):
        indices = np.array(hospital_indices[k], dtype=np.int64)
        np.random.shuffle(indices)
        
        X_k = X[indices]
        y_k = y[indices]
        
        # Calculate class distribution
        unique, counts = np.unique(y_k, return_counts=True)
        class_dist = dict(zip(unique, counts))
        class_ratios = {cls: counts[i] / len(y_k) for i, cls in enumerate(unique)}
        
        hospitals.append((X_k, y_k))
        
        print(f"Hospital {k+1}: {len(y_k)} samples, class dist: {class_dist}, ratios: {class_ratios}")
    
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
    random_seed: int = 42
) -> Dict:
    """
    Train federated model using FedProx algorithm.
    
    FedProx Algorithm:
    1. Initialize global weights w_global
    2. For each communication round:
       a. Send w_global to all hospitals
       b. Each hospital trains locally with proximal term:
          L_k(w) = cross_entropy + (mu/2) * ||w - w_global||^2
       c. Aggregate: w_global = weighted_average(w_k)
    3. Return final w_global
    
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
        
    Returns:
    --------
    Dict
        Dictionary containing:
        - w_global: Final global weights
        - round_losses: List of global losses per round
        - round_aucs: List of global AUCs per round
        - round_metrics: Detailed metrics per round
        - weight_drifts: L2 norm of weight change per round
    """
    # Get number of features from first hospital
    n_features = hospitals[0][0].shape[1]
    
    # Initialize global weights
    w_global = initialize_weights(n_features, random_seed)
    
    # Calculate total samples and weights per hospital
    total_samples = sum(len(y_k) for _, y_k in hospitals)
    hospital_weights = [len(y_k) / total_samples for _, y_k in hospitals]
    
    print(f"\n{'='*60}")
    print(f"FEDERATED LEARNING - FedProx")
    print(f"{'='*60}")
    print(f"Number of hospitals: {len(hospitals)}")
    print(f"Total samples: {total_samples}")
    print(f"Communication rounds: {rounds}")
    print(f"Local epochs per round: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Proximal coefficient (mu): {mu}")
    print(f"{'='*60}\n")
    
    round_losses = []
    round_aucs = []
    round_metrics = []
    weight_drifts = []
    
    for round_idx in range(rounds):
        w_global_prev = w_global.copy()
        
        # Store local weights from each hospital
        local_weights = []
        local_losses = []
        
        # Each hospital trains locally with proximal term
        for k, (X_k, y_k) in enumerate(hospitals):
            # Local training with FedProx
            w_k, loss_history = local_train_fedprox(X_k, y_k, w_global, epochs, lr, mu)
            
            local_weights.append(w_k)
            local_losses.append(loss_history[-1])  # Last epoch loss
        
        # Aggregate weights using weighted average (FedAvg aggregation)
        w_global = np.zeros_like(w_global)
        for k, w_k in enumerate(local_weights):
            w_global += hospital_weights[k] * w_k
        
        # Compute weight drift (L2 norm of change)
        drift = np.linalg.norm(w_global - w_global_prev)
        weight_drifts.append(drift)
        
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
            'weight_drift': drift
        })
        
        # Print progress every 10 rounds
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            print(f"Round {round_idx + 1}/{rounds} - Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Drift: {drift:.4f}")
    
    print(f"\n{'='*60}")
    print(f"FedProx Training Complete!")
    print(f"Final Test AUC: {round_aucs[-1]:.4f}")
    print(f"Average Weight Drift: {np.mean(weight_drifts):.4f}")
    print(f"{'='*60}\n")
    
    return {
        'w_global': w_global,
        'round_losses': round_losses,
        'round_aucs': round_aucs,
        'round_metrics': round_metrics,
        'weight_drifts': weight_drifts
    }
