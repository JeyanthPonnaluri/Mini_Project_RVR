import os
import sys
import numpy as np
import pandas as pd

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from logistic_numpy import initialize_weights, local_train_dp, local_train_fedprox_dp
from federated import partition_equal, fedavg_train, fedprox_train
from shapley import compute_federated_shapley_values
from evaluation import bootstrap_auc_ci, plot_roc_curve

def generate_mock_data(n_samples=100, n_features=10):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Generate binary target
    w_true = np.random.randn(n_features)
    z = X @ w_true
    p = 1 / (1 + np.exp(-z))
    y = (p >= 0.5).astype(int)
    return X, y

def test_differential_privacy():
    print("\n--- Testing Differential Privacy ---")
    X, y = generate_mock_data(50, 5)
    w_init = initialize_weights(5, random_seed=42)
    
    # Run with DP
    w_dp, loss_dp = local_train_dp(
        X, y, w_init, epochs=5, lr=0.1,
        epsilon=1.0, delta=1e-3, clipping_norm=1.0,
        random_seed=42
    )
    print(f"DP Weights shape: {w_dp.shape}")
    print(f"DP Final Loss: {loss_dp[-1]:.4f}")
    
    # Run with FedProx DP
    w_prox_dp, loss_prox_dp = local_train_fedprox_dp(
        X, y, w_init, epochs=5, lr=0.1, mu=0.1,
        epsilon=1.0, delta=1e-3, clipping_norm=1.0,
        random_seed=42
    )
    print(f"FedProx-DP Weights shape: {w_prox_dp.shape}")
    print(f"FedProx-DP Final Loss: {loss_prox_dp[-1]:.4f}")
    print("DP Local training tests passed!")

def test_federated_train_with_dp():
    print("\n--- Testing Federated Training with DP ---")
    X, y = generate_mock_data(120, 8)
    X_test, y_test = generate_mock_data(30, 8)
    
    hospitals = partition_equal(X, y, num_hospitals=3, random_seed=42)
    
    # Run FedAvg with DP
    fedavg_res = fedavg_train(
        hospitals, X_test, y_test, rounds=3, epochs=2, lr=0.1,
        random_seed=42, dp_enabled=True, epsilon=2.0, delta=1e-4, clipping_norm=1.2
    )
    print(f"FedAvg-DP Final AUC: {fedavg_res['round_aucs'][-1]:.4f}")
    
    # Run FedProx with DP
    fedprox_res = fedprox_train(
        hospitals, X_test, y_test, rounds=3, epochs=2, lr=0.1, mu=0.2,
        random_seed=42, dp_enabled=True, epsilon=2.0, delta=1e-4, clipping_norm=1.2
    )
    print(f"FedProx-DP Final AUC: {fedprox_res['round_aucs'][-1]:.4f}")
    print("Federated training with DP tests passed!")

def test_shapley_values():
    print("\n--- Testing Federated Shapley Value ---")
    X, y = generate_mock_data(80, 6)
    X_test, y_test = generate_mock_data(20, 6)
    hospitals = partition_equal(X, y, num_hospitals=3, random_seed=42)
    
    # Compute SV
    shapley_df = compute_federated_shapley_values(
        hospitals, X_test, y_test, rounds=2, epochs=2, lr=0.1,
        algorithm='fedavg', mu=0.1, n_permutations=6, random_seed=42,
        dp_enabled=True, epsilon=1.5, delta=1e-4, clipping_norm=1.0
    )
    print("Shapley Value Table:")
    print(shapley_df)
    assert len(shapley_df) == 3, "Should compute Shapley values for 3 hospitals"
    print("Shapley Value tests passed!")

def test_bootstrapped_ci():
    print("\n--- Testing Bootstrapped AUC CIs ---")
    np.random.seed(42)
    y_true = np.array([0]*30 + [1]*30)
    y_pred_proba = np.random.uniform(0.1, 0.9, size=60)
    # Add correlation to ensure AUC > 0.5
    y_pred_proba[30:] += 0.2
    y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)
    
    mean_auc, lower, upper = bootstrap_auc_ci(y_true, y_pred_proba, n_bootstraps=100)
    print(f"Mean AUC: {mean_auc:.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")
    assert lower <= mean_auc <= upper, "Mean AUC must lie in the confidence interval"
    print("Bootstrapped CIs tests passed!")

if __name__ == "__main__":
    print("==================================================")
    print("RUNNING VERIFICATION OF JOURNAL UPGRADE COMPONENTS")
    print("==================================================")
    try:
        test_differential_privacy()
        test_federated_train_with_dp()
        test_shapley_values()
        test_bootstrapped_ci()
        print("\n==================================================")
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("==================================================")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
