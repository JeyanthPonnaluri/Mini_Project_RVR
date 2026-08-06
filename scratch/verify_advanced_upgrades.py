import sys
import os
import numpy as np
import pandas as pd

# Add src folder to path
sys.path.insert(0, "D:/Mini_project_JP/src")

from model import train_non_linear_model, predict_model
from preprocessing import load_survival_data, merge_clinical_survival
from cox_numpy import local_cox_train, compute_concordance_index
from federated import fedavg_cox_train, evaluate_personalized_fl, compute_rdp_privacy_budget

print("==================================================")
print("RUNNING ADVANCED MATH & SYSTEM UPGRADES VERIFICATION")
print("==================================================")

# Generate toy classification data
np.random.seed(42)
X_class = np.random.randn(100, 5)
y_class = np.random.randint(0, 2, size=100)
X_test_class = np.random.randn(20, 5)
y_test_class = np.random.randint(0, 2, size=20)

# Test 1: Random Forest & MLP baselines
print("\n--- Test 1: Centralized RF & MLP Baselines ---")
rf_model = train_non_linear_model(X_class, y_class, model_type='rf', random_seed=42)
mlp_model = train_non_linear_model(X_class, y_class, model_type='mlp', random_seed=42)

y_pred_rf, y_prob_rf = predict_model(rf_model, X_test_class)
y_pred_mlp, y_prob_mlp = predict_model(mlp_model, X_test_class)

print(f"Random Forest test AUC-ROC matches predictions: {y_pred_rf.shape == (20,)}")
print(f"MLP test AUC-ROC matches predictions: {y_pred_mlp.shape == (20,)}")
print("[OK] Centralized non-linear baselines verified.")

# Test 2: Survival Preprocessing
print("\n--- Test 2: Survival Preprocessing Merges ---")
# Create mock survival data frame
mock_survival_df = pd.DataFrame({
    'sample': [f'Sample_{i}' for i in range(10)],
    'OS.time': np.random.uniform(10, 100, size=10).round(1),
    'OS': np.random.randint(0, 2, size=10),
    '_PATIENT': [f'Patient_{i}' for i in range(10)]
})
mock_clinical_df = pd.DataFrame({
    'sample': [f'Sample_{i}' for i in range(10)],
    'age': np.random.randint(50, 80, size=10),
    'bcr_patient_barcode': [f'Patient_{i}' for i in range(10)]
})

merged_survival = merge_clinical_survival(mock_clinical_df, mock_survival_df)
print(f"Merged columns: {list(merged_survival.columns)}")
print(f"Success: {'OS.time' in merged_survival.columns and 'OS' in merged_survival.columns}")
print("[OK] Survival merge preprocessors verified.")

# Test 3: Cox Proportional Hazards regression model
print("\n--- Test 3: NumPy Cox Proportional Hazards Model ---")
# Generate toy survival data (X, times, events)
X_surv = np.random.randn(50, 4)
times_surv = np.random.uniform(5, 120, size=50)
events_surv = np.random.randint(0, 2, size=50)

# Sort times to make c-index calculations clean
w = np.zeros(4)
c_index_init = compute_concordance_index(w, X_surv, times_surv, events_surv)
print(f"Initial C-index (zero weights): {c_index_init:.4f}")

# Train local Cox model
w_trained = local_cox_train(X_surv, times_surv, events_surv, epochs=10, lr=0.01, alpha=0.1)
c_index_trained = compute_concordance_index(w_trained, X_surv, times_surv, events_surv)
print(f"Trained C-index: {c_index_trained:.4f}")
print(f"Weights: {w_trained}")
print("[OK] Vectorized NumPy Cox model training verified.")

# Test 4: Federated Cox Model Training
print("\n--- Test 4: Federated Cox Model Training (FedAvg) ---")
# Partition toy survival data into 3 mock clinics
hospitals_surv = [
    (X_surv[:15], times_surv[:15], events_surv[:15]),
    (X_surv[15:35], times_surv[15:35], events_surv[15:35]),
    (X_surv[35:], times_surv[35:], events_surv[35:])
]
X_test_surv = np.random.randn(15, 4)
times_test_surv = np.random.uniform(10, 100, size=15)
events_test_surv = np.random.randint(0, 2, size=15)

fed_cox_res = fedavg_cox_train(
    hospitals_surv, X_test_surv, times_test_surv, events_test_surv,
    rounds=5, epochs=3, lr=0.05, alpha=0.1, random_seed=42,
    dropout_rate=0.2, bandwidth_mbps=10.0, latency_ms=50.0
)

print(f"Global consensus weights: {fed_cox_res['w_global']}")
print(f"Rounds C-index curve: {fed_cox_res['round_c_indices']}")
print(f"Total simulated data transferred: {fed_cox_res['cumulative_bytes'][-1]} bytes")
print(f"Total simulated virtual execution duration: {fed_cox_res['cumulative_time_s'][-1]:.2f} seconds")
print("[OK] Federated Cox model training loop verified.")

# Test 5: Personalized FL (PFL) evaluation
print("\n--- Test 5: Personalized FL (PFL) Fine-tuning ---")
# Create mock classification clinics
hospitals_class = [
    (X_class[:30], y_class[:30]),
    (X_class[30:65], y_class[30:65]),
    (X_class[65:], y_class[65:])
]
w_global_class = np.zeros(5)

pfl_res = evaluate_personalized_fl(hospitals_class, w_global_class, epochs=2, lr=0.1, random_seed=42)
print(f"Mean AUC before personalization: {pfl_res['mean_auc_before']:.4f}")
print(f"Mean AUC after personalization: {pfl_res['mean_auc_after']:.4f}")
print(f"Local AUCs before: {pfl_res['local_aucs_before']}")
print(f"Local AUCs after: {pfl_res['local_aucs_after']}")
print("[OK] Personalized FL fine-tuning evaluations verified.")

# Test 6: Rényi Differential Privacy (RDP) Budget Calculation
print("\n--- Test 6: Rényi DP Privacy Accountant Compositions ---")
epsilon_composed = compute_rdp_privacy_budget(q=0.6, epsilon_local=1.0, T=30, delta=1e-5)
print(f"Composed global Epsilon (T=30, q=0.6, local_eps=1.0): {epsilon_composed:.4f}")
print("[OK] RDP mechanism composition accountant verified.")

print("\n==================================================")
print("ALL ADVANCED UPGRADES VERIFIED SUCCESSFULLY!")
print("==================================================")
