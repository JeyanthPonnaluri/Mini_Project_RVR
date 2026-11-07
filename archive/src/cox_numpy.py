"""
Vectorized Cox Proportional Hazards model in NumPy.
Implements log-likelihood, gradient calculations, DP clipping, and FedProx constraints.
"""

import numpy as np


def compute_cox_likelihood_and_grad(w, X, times, events, alpha=0.1, w_global=None, mu=0.0):
    """
    Compute Cox partial log-likelihood and gradients.
    Assumes X, times, events are pre-sorted in ascending order of times.
    
    Parameters:
    -----------
    w : np.ndarray
        Model weights (d,)
    X : np.ndarray
        Sorted feature matrix (n, d)
    times : np.ndarray
        Sorted survival times (n,)
    events : np.ndarray
        Sorted event indicators (n,)
    alpha : float, default=0.1
        L2 regularization coefficient
    w_global : np.ndarray, optional
        Global weights from server (for FedProx)
    mu : float, default=0.0
        FedProx proximal coefficient
    """
    n, d = X.shape
    theta = X.dot(w)
    
    # Clip theta to avoid overflow in exp
    theta = np.clip(theta, -20.0, 20.0)
    exp_theta = np.exp(theta)
    
    # Reverse cumulative sums for risk set sums
    # risk_sums[i] = sum_{j=i}^n exp(theta_j)
    risk_sums = np.flip(np.cumsum(np.flip(exp_theta)))
    risk_sums = np.maximum(risk_sums, 1e-9)  # Avoid division by zero
    
    # Calculate negative partial log-likelihood
    log_likelihood = np.sum(events * (theta - np.log(risk_sums)))
    # Add L2 regularization
    loss = -log_likelihood + 0.5 * alpha * np.sum(w**2)
    
    # Calculate gradient of log-likelihood
    # sum_x_exp_theta[i] = sum_{j=i}^n X_j * exp(theta_j)
    x_exp_theta = X * exp_theta[:, np.newaxis]
    sum_x_exp_theta = np.flip(np.cumsum(np.flip(x_exp_theta, axis=0), axis=0), axis=0)
    
    risk_x_means = sum_x_exp_theta / risk_sums[:, np.newaxis]
    
    # Per-sample gradients: G[i] = events[i] * (X[i] - risk_x_means[i])
    G = events[:, np.newaxis] * (X - risk_x_means)
    
    # Total gradient of negative log-likelihood
    grad = -np.sum(G, axis=0) + alpha * w
    
    # Add FedProx proximal penalty
    if w_global is not None and mu > 0:
        prox_grad = mu * (w - w_global)
        grad += prox_grad
        loss += 0.5 * mu * np.sum((w - w_global)**2)
        
    return loss, grad, G


def local_cox_train(X, times, events, epochs=5, lr=0.01, alpha=0.1, w_global=None, mu=0.0,
                    dp_enabled=False, epsilon=1.0, delta=1e-5, clipping_norm=1.0, noise_mult=None):
    """
    Local training loop using gradient descent on the Cox model.
    """
    n, d = X.shape
    w = np.copy(w_global) if w_global is not None else np.zeros(d)
    
    # Sort samples by time (ascending) to prepare for Cox likelihood
    sort_idx = np.argsort(times)
    X_sorted = X[sort_idx]
    times_sorted = times[sort_idx]
    events_sorted = events[sort_idx]
    
    # Calibration of Gaussian noise for DP
    if dp_enabled:
        if noise_mult is None:
            sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        else:
            sigma = noise_mult
        noise_std = (clipping_norm * sigma) / n
    else:
        noise_std = 0.0
        
    for epoch in range(epochs):
        loss, grad, G = compute_cox_likelihood_and_grad(
            w, X_sorted, times_sorted, events_sorted, alpha, w_global, mu
        )
        
        if dp_enabled:
            # Per-sample gradient clipping (G is from sorted data)
            norms = np.linalg.norm(G, axis=1)
            clip_factors = np.minimum(1.0, clipping_norm / (norms + 1e-9))
            G_clipped = G * clip_factors[:, np.newaxis]
            
            # Reconstruct gradient with noise
            # Note: Negative sign since loss gradient was -G
            grad = -np.sum(G_clipped, axis=0) + alpha * w
            if w_global is not None and mu > 0:
                grad += mu * (w - w_global)
                
            # Add calibrated noise
            noise = np.random.normal(0, noise_std, size=d)
            grad += noise
            
        w -= lr * grad
        
    return w


def compute_concordance_index(w, X, times, events):
    """
    Compute the Harrell's Concordance Index (C-index) for survival prediction.
    A value of 0.5 represents random prediction, and 1.0 represents perfect prediction.
    """
    n = len(times)
    concordant = 0
    permissible = 0
    tied_risk = 0
    
    # Predict risk scores: risk = exp(Xw) -> monotonic to risk_score = Xw
    risk_scores = X.dot(w)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if pair is permissible
            # Case 1: both experienced the event, times are different
            # Case 2: one experienced event at t_i, other censored at t_j > t_i
            if times[i] < times[j]:
                if events[i] == 1:
                    permissible += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        tied_risk += 0.5
            elif times[j] < times[i]:
                if events[j] == 1:
                    permissible += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] == risk_scores[i]:
                        tied_risk += 0.5
                        
    if permissible == 0:
        return 0.5
        
    return (concordant + tied_risk) / permissible
