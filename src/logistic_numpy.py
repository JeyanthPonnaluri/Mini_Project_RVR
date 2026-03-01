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
    lr: float
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
