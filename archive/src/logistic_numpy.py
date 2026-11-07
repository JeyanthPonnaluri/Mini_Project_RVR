"""
Manual logistic regression implementation using NumPy.
Used for federated learning to avoid sklearn's .fit() method.
"""

import numpy as np
from typing import Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid function.
    
    Parameters:
    -----------
    z : np.ndarray
        Input array
        
    Returns:
    --------
    np.ndarray
        Sigmoid of input: 1 / (1 + exp(-z))
    """
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def initialize_weights(n_features: int, random_seed: int = 42) -> np.ndarray:
    """
    Initialize weights for logistic regression.
    
    Parameters:
    -----------
    n_features : int
        Number of features (including bias if applicable)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Initialized weights of shape (n_features,)
    """
    np.random.seed(random_seed)
    # Xavier initialization
    return np.random.randn(n_features) * 0.01


def compute_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss.
    
    Loss = -(1/n) * sum(y * log(y_hat) + (1-y) * log(1-y_hat))
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        True labels of shape (n_samples,)
    w : np.ndarray
        Weights of shape (n_features,)
        
    Returns:
    --------
    float
        Binary cross-entropy loss
    """
    n = len(y)
    
    # Predictions
    z = X @ w
    y_hat = sigmoid(z)
    
    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    
    # Binary cross-entropy
    loss = -(1.0 / n) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    return loss


def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute gradient of binary cross-entropy loss.
    
    Gradient = (1/n) * X^T @ (y_hat - y)
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        True labels of shape (n_samples,)
    w : np.ndarray
        Weights of shape (n_features,)
        
    Returns:
    --------
    np.ndarray
        Gradient of shape (n_features,)
    """
    n = len(y)
    
    # Predictions
    z = X @ w
    y_hat = sigmoid(z)
    
    # Gradient
    gradient = (1.0 / n) * (X.T @ (y_hat - y))
    
    return gradient


def local_train(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    epochs: int,
    lr: float,
    l2_reg: float = 0.0
) -> Tuple[np.ndarray, list]:
    """
    Train logistic regression locally using gradient descent.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        True labels of shape (n_samples,)
    w_init : np.ndarray
        Initial weights of shape (n_features,)
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    l2_reg : float
        L2 regularization coefficient
        
    Returns:
    --------
    tuple
        (w_final, loss_history)
        - w_final: Trained weights
        - loss_history: List of losses per epoch
    """
    w = w_init.copy()
    loss_history = []
    
    for epoch in range(epochs):
        # Compute gradient
        grad = compute_gradient(X, y, w)
        if l2_reg > 0.0:
            grad = grad + l2_reg * w
        
        # Update weights
        w = w - lr * grad
        
        # Compute loss
        loss = compute_loss(X, y, w)
        loss_history.append(loss)
    
    return w, loss_history


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Predict probabilities using logistic regression.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    w : np.ndarray
        Weights of shape (n_features,)
        
    Returns:
    --------
    np.ndarray
        Predicted probabilities of shape (n_samples,)
    """
    z = X @ w
    return sigmoid(z)


def predict(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Predict class labels using logistic regression.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    w : np.ndarray
        Weights of shape (n_features,)
    threshold : float
        Classification threshold
        
    Returns:
    --------
    np.ndarray
        Predicted class labels of shape (n_samples,)
    """
    proba = predict_proba(X, w)
    return (proba >= threshold).astype(int)


def local_train_dp(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    epochs: int,
    lr: float,
    epsilon: float,
    delta: float,
    clipping_norm: float,
    random_seed: int = 42,
    noise_mult: float = None
) -> Tuple[np.ndarray, list]:
    """
    Train logistic regression locally using DP-SGD (clipping and noise addition).
    """
    np.random.seed(random_seed)
    w = w_init.copy()
    loss_history = []
    n_samples, n_features = X.shape
    
    if noise_mult is None:
        # Calculate noise multiplier using standard Gaussian mechanism approximation
        if epsilon <= 0:
            raise ValueError("Epsilon must be greater than zero.")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1).")
        noise_mult = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    
    for epoch in range(epochs):
        z = X @ w
        y_hat = sigmoid(z)
        
        # Per-sample gradients
        errors = (y_hat - y).reshape(-1, 1)
        per_sample_grads = X * errors
        
        # Clip per-sample gradients
        l2_norms = np.linalg.norm(per_sample_grads, axis=1, keepdims=True)
        l2_norms = np.clip(l2_norms, 1e-15, None)
        clip_factors = np.minimum(1.0, clipping_norm / l2_norms)
        clipped_grads = per_sample_grads * clip_factors
        
        # Sum and add noise
        sum_clipped_grad = np.sum(clipped_grads, axis=0)
        noise_std = noise_mult * clipping_norm
        noise = np.random.normal(0, noise_std, size=n_features)
        
        # Update
        dp_grad = (sum_clipped_grad + noise) / n_samples
        w = w - lr * dp_grad
        
        # Compute loss
        loss = compute_loss(X, y, w)
        loss_history.append(loss)
        
    return w, loss_history


def local_train_fedprox_dp(
    X: np.ndarray,
    y: np.ndarray,
    w_global: np.ndarray,
    epochs: int,
    lr: float,
    mu: float,
    epsilon: float,
    delta: float,
    clipping_norm: float,
    random_seed: int = 42,
    noise_mult: float = None
) -> Tuple[np.ndarray, list]:
    """
    Train logistic regression locally using DP-SGD with FedProx proximal term.
    """
    np.random.seed(random_seed)
    w = w_global.copy()
    loss_history = []
    n_samples, n_features = X.shape
    
    if noise_mult is None:
        # Calculate noise multiplier using standard Gaussian mechanism approximation
        if epsilon <= 0:
            raise ValueError("Epsilon must be greater than zero.")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1).")
        noise_mult = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    
    for epoch in range(epochs):
        z = X @ w
        y_hat = sigmoid(z)
        
        # Per-sample gradients of cross-entropy
        errors = (y_hat - y).reshape(-1, 1)
        per_sample_grads = X * errors
        
        # Clip per-sample gradients
        l2_norms = np.linalg.norm(per_sample_grads, axis=1, keepdims=True)
        l2_norms = np.clip(l2_norms, 1e-15, None)
        clip_factors = np.minimum(1.0, clipping_norm / l2_norms)
        clipped_grads = per_sample_grads * clip_factors
        
        # Sum and add noise
        sum_clipped_grad = np.sum(clipped_grads, axis=0)
        noise_std = noise_mult * clipping_norm
        noise = np.random.normal(0, noise_std, size=n_features)
        
        # Average DP gradient for BCE
        dp_grad_ce = (sum_clipped_grad + noise) / n_samples
        
        # Proximal term gradient
        grad_prox = mu * (w - w_global)
        
        # Update
        w = w - lr * (dp_grad_ce + grad_prox)
        
        # Compute loss (including proximal term)
        loss_ce = compute_loss(X, y, w)
        loss_prox = (mu / 2.0) * np.sum((w - w_global) ** 2)
        loss = loss_ce + loss_prox
        loss_history.append(loss)
        
    return w, loss_history

