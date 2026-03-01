"""
Machine learning model module for clinical stage classification.
Implements logistic regression training and prediction.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, random_seed=42):
    """
    Train a logistic regression model with balanced class weights.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray or pd.Series
        Training labels (binary: 0 or 1)
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    LogisticRegression
        Trained logistic regression model
    """
    print(f"Training logistic regression on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    print(f"Training set class distribution: {dict(pd.Series(y_train).value_counts().sort_index())}")
    
    model = LogisticRegression(
        random_state=random_seed,
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    print(f"Training complete - Training accuracy: {train_score:.4f}")
    
    return model


def predict_model(model, X_test):
    """
    Make predictions using a trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with predict and predict_proba methods
    X_test : np.ndarray
        Test features
        
    Returns:
    --------
    tuple
        (y_pred, y_pred_proba)
        - y_pred: np.ndarray of predicted class labels
        - y_pred_proba: np.ndarray of predicted probabilities for positive class
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Predictions generated for {X_test.shape[0]} samples")
    
    return y_pred, y_pred_proba


# Import pandas for class distribution display
import pandas as pd
