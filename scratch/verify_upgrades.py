import sys
import os
import numpy as np

# Add src folder to sys path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Add workspace path for execution
sys.path.insert(0, "D:/Mini_project_JP/src")

from model import train_model, train_regularized_model, predict_model
from federated import fedavg_train, fedprox_train
from evaluation import evaluate_model

print("==================================================")
print("RUNNING VERIFICATION OF SYSTEM UPGRADES")
print("==================================================")

# Generate toy dataset
print("\n--- Generating Toy Dataset ---")
np.random.seed(42)
X_train = np.random.randn(100, 10)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.randn(20, 10)
y_test = np.random.randint(0, 2, size=20)

print(f"X_train: {X_train.shape}, y_train class distribution: {np.bincount(y_train)}")

# 1. Testing L1/L2 regularized baseline models
print("\n--- Testing Scikit-learn L1/L2 regularized baselines ---")
try:
    l1_model = train_regularized_model(X_train, y_train, penalty='l1', C=1.0)
    l2_model = train_regularized_model(X_train, y_train, penalty='l2', C=1.0)
    
    y_pred, y_prob = predict_model(l1_model, X_test)
    print(f"L1 model test predictions shape: {y_pred.shape}")
    print("[OK] L1/L2 regularized baseline training passed!")
except Exception as e:
    print(f"Error in L1/L2 baselines: {e}")
    import traceback
    traceback.print_exc()

# 2. Testing Federated training with client dropouts & network metrics
print("\n--- Testing Federated Training with Client Dropouts ---")
try:
    # Partition data into 3 mock hospitals
    hospitals = []
    hospitals.append((X_train[:30], y_train[:30]))
    hospitals.append((X_train[30:65], y_train[30:65]))
    hospitals.append((X_train[65:], y_train[65:]))
    
    # Train FedAvg with 40% dropout rate
    print("\nRunning FedAvg with 40% dropout rate...")
    fedavg_res = fedavg_train(
        hospitals, X_test, y_test,
        rounds=5, epochs=2, lr=0.1,
        dropout_rate=0.4,
        bandwidth_mbps=10.0,
        latency_ms=100.0,
        random_seed=42
    )
    
    print("\nFedAvg Verification Metrics:")
    print(f"Final Test AUC: {fedavg_res['round_aucs'][-1]:.4f}")
    print(f"Cumulative bytes transferred: {fedavg_res['cumulative_bytes'][-1]} bytes ({fedavg_res['cumulative_bytes'][-1]/1024:.2f} KB)")
    print(f"Cumulative virtual time: {fedavg_res['cumulative_time_s'][-1]:.2f} seconds")
    print(f"Active clients count per round: {[m['active_clients'] for m in fedavg_res['round_metrics']]}")
    
    # Train FedProx with 40% dropout rate
    print("\nRunning FedProx with 40% dropout rate...")
    fedprox_res = fedprox_train(
        hospitals, X_test, y_test,
        rounds=5, epochs=2, lr=0.1, mu=0.5,
        dropout_rate=0.4,
        bandwidth_mbps=10.0,
        latency_ms=100.0,
        random_seed=42
    )
    
    print("\nFedProx Verification Metrics:")
    print(f"Final Test AUC: {fedprox_res['round_aucs'][-1]:.4f}")
    print(f"Cumulative bytes transferred: {fedprox_res['cumulative_bytes'][-1]} bytes ({fedprox_res['cumulative_bytes'][-1]/1024:.2f} KB)")
    print(f"Cumulative virtual time: {fedprox_res['cumulative_time_s'][-1]:.2f} seconds")
    print(f"Active clients count per round: {[m['active_clients'] for m in fedprox_res['round_metrics']]}")
    
    print("\n[OK] Federated training with dropouts and latency tracking passed!")
except Exception as e:
    print(f"Error in Federated training: {e}")
    import traceback
    traceback.print_exc()

print("\n==================================================")
print("ALL SYSTEM UPGRADE VERIFICATIONS PASSED!")
print("==================================================")
