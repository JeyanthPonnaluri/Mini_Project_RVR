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


def train_regularized_model(X_train, y_train, penalty='l2', C=1.0, random_seed=42):
    """
    Train a logistic regression model with L1 or L2 regularization and balanced weights.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray or pd.Series
        Training labels
    penalty : str, default='l2'
        Type of regularization: 'l1' (Lasso) or 'l2' (Ridge)
    C : float, default=1.0
        Inverse of regularization strength (smaller specifies stronger regularization)
    random_seed : int, default=42
        Random seed
        
    Returns:
    --------
    LogisticRegression
        Trained regularized model
    """
    print(f"Training Scikit-learn {penalty.upper()} regularized logistic regression (C={C}) on {X_train.shape[0]} samples...")
    
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        random_state=random_seed,
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    print(f"Training complete - Training accuracy: {train_score:.4f}")
    
    return model


def train_non_linear_model(X_train, y_train, model_type='rf', random_seed=42):
    """
    Train a non-linear model (Random Forest 'rf' or Multi-Layer Perceptron 'mlp') with balanced weights.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray or pd.Series
        Training labels
    model_type : str, default='rf'
        'rf' for Random Forest, 'mlp' for Multi-Layer Perceptron
    random_seed : int, default=42
        Random seed
        
    Returns:
    --------
    Classifier model
        Trained non-linear model
    """
    if model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        print(f"Training Random Forest on {X_train.shape[0]} samples...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_seed,
            class_weight='balanced'
        )
    elif model_type == 'mlp':
        from sklearn.neural_network import MLPClassifier
        print(f"Training Multi-Layer Perceptron (MLP) on {X_train.shape[0]} samples...")
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=random_seed,
            early_stopping=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    print(f"Non-linear baseline {model_type.upper()} training complete - Accuracy: {train_score:.4f}")
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
