import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocessing import (
    load_clinical,
    load_protein,
    merge_clinical_protein,
    create_target,
    preprocess_features,
    preprocess_protein,
    apply_pca
)
from logistic_numpy import initialize_weights, local_train, predict_proba
from federated import (
    partition_dirichlet,
    fedavg_train,
    fedprox_train
)

OUTPUT_DIR = "reports/boundary_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

def load_data():
    clinical_path = "D:/Mini_project_JP/datasets/TCGA-PRAD.clinical.tsv/TCGA-PRAD.clinical.tsv"
    protein_path = "D:/Mini_project_JP/datasets/TCGA-PRAD.protein.tsv/TCGA-PRAD.protein.tsv"
    
    clinical_df = load_clinical(clinical_path)
    protein_df = load_protein(protein_path)
    
    merged_df = merge_clinical_protein(clinical_df, protein_df)
    df_filtered, target = create_target(merged_df)
    
    protein_cols = [c for c in protein_df.columns if c not in ['sample', 'case_id', 'patient_id', 'submitter_id', 'bcr_patient_barcode']]
    clinical_cols = [c for c in df_filtered.columns if c not in protein_cols]
    
    X_clin, _, _ = preprocess_features(df_filtered[clinical_cols])
    X_prot, _ = preprocess_protein(df_filtered[['sample'] + protein_cols])
    X_prot_pca, _, _ = apply_pca(X_prot, variance_threshold=0.95)
    
    X = np.hstack([X_clin, X_prot_pca])
    y = np.array(target)
    
    return X, y

def test_boundary_1_hospital_scaling(X, y):
    print("\n==================================================")
    # Changed heading to standard characters to avoid Windows encoding issues
    print("BOUNDARY TEST 1: HOSPITAL SCALING EFFECT (1 TO 10)")
    print("==================================================")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Partition into 10 hospitals (Dirichlet alpha=0.5)
    ten_hospitals = partition_dirichlet(X_train, y_train, num_hospitals=10, alpha=0.5, random_seed=RANDOM_SEED)
    
    # We will vary the number of contributing hospitals N from 1 to 10
    n_values = [1, 2, 3, 5, 8, 10]
    scaling_aucs = []
    
    for n in n_values:
        sub_hospitals = ten_hospitals[:n]
        print(f"Training with N = {n} hospitals contributing...")
        res = fedavg_train(
            sub_hospitals, X_test, y_test, rounds=15, epochs=5, lr=0.1, random_seed=RANDOM_SEED
        )
        scaling_aucs.append(res['round_aucs'][-1])
        
    print("\nHospital Scaling Results:")
    for n, auc_val in zip(n_values, scaling_aucs):
        print(f"  N = {n:2d} Contributing Hospitals: AUC = {auc_val:.4f}")
        
    # Plot scaling curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, scaling_aucs, marker='o', color='#2CA02C', lw=2)
    ax.set_xlabel('Number of Contributing Hospitals (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Model Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('Global Model Utility vs. Consortium Size', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    
    plot_path = os.path.join(OUTPUT_DIR, "bound1_hospital_scaling.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Scaling plot to {plot_path}")
    return n_values, scaling_aucs

def test_boundary_2_heterogeneity_limit(X, y):
    print("\n==================================================")
    print("BOUNDARY TEST 2: EXTREME HETEROGENEITY LIMITS")
    print("==================================================")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Alpha values from extreme IID (10.0) to extreme non-IID (0.02)
    alphas = [10.0, 1.0, 0.1, 0.02]
    avg_aucs = []
    prox_aucs = []
    
    for alpha in alphas:
        print(f"\nEvaluating partitions with Dirichlet alpha = {alpha}...")
        hospitals = partition_dirichlet(X_train, y_train, num_hospitals=5, alpha=alpha, random_seed=RANDOM_SEED)
        
        # FedAvg (mu=0.0)
        res_avg = fedavg_train(hospitals, X_test, y_test, rounds=20, epochs=5, lr=0.1, random_seed=RANDOM_SEED)
        avg_aucs.append(res_avg['round_aucs'][-1])
        
        # FedProx (mu=1.0)
        res_prox = fedprox_train(hospitals, X_test, y_test, rounds=20, epochs=5, lr=0.1, mu=1.0, random_seed=RANDOM_SEED)
        prox_aucs.append(res_prox['round_aucs'][-1])
        
    print("\nHeterogeneity Limit Results:")
    for idx, alpha in enumerate(alphas):
        diff = prox_aucs[idx] - avg_aucs[idx]
        print(f"  Alpha = {alpha:5.2f} | FedAvg AUC: {avg_aucs[idx]:.4f} | FedProx AUC: {prox_aucs[idx]:.4f} | Delta: {diff:+.4f}")
        
    # Plot bar comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(alphas))
    width = 0.35
    
    ax.bar(x - width/2, avg_aucs, width, label='FedAvg (mu=0.0)', color='#A23B72')
    ax.bar(x + width/2, prox_aucs, width, label='FedProx (mu=1.0)', color='#2E86AB')
    
    ax.set_ylabel('Global Model Test AUC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dirichlet Skew Parameter (Alpha) - Lower is Skewed', fontsize=12, fontweight='bold')
    ax.set_title('FedAvg vs. FedProx across Heterogeneity Bounds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, linestyle='--')
    
    plot_path = os.path.join(OUTPUT_DIR, "bound2_heterogeneity_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Heterogeneity plot to {plot_path}")

def test_boundary_3_cooperation_incentive(X, y):
    print("\n==================================================")
    print("BOUNDARY TEST 3: COOPERATION INCENTIVE BOUNDS")
    print("==================================================")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # 5 Hospitals with highly unequal sizes: H1(70%), H2(15%), H3(7.5%), H4(5%), H5(2.5%)
    np.random.seed(RANDOM_SEED)
    total_samples = len(y_train)
    indices = np.random.permutation(total_samples)
    
    splits = [0.70, 0.15, 0.075, 0.05, 0.025]
    hospitals = []
    curr_idx = 0
    for s in splits:
        size = int(s * total_samples)
        idx = indices[curr_idx:curr_idx+size]
        hospitals.append((X_train[idx], y_train[idx]))
        curr_idx += size
        
    print(f"Hospital sample sizes: H1: {len(hospitals[0][0])}, H2: {len(hospitals[1][0])}, H3: {len(hospitals[2][0])}, H4: {len(hospitals[3][0])}, H5: {len(hospitals[4][0])}")
    
    # 1. Train local models for each hospital
    local_aucs = []
    for idx, (X_h, y_h) in enumerate(hospitals):
        if len(np.unique(y_h)) < 2:
            print(f"  Hospital {idx+1} has single class labels, setting Local AUC = 0.5")
            local_aucs.append(0.5)
            continue
        w_init = initialize_weights(X_h.shape[1], RANDOM_SEED)
        w_final, _ = local_train(X_h, y_h, w_init, epochs=25, lr=0.1)
        y_pred = predict_proba(X_test, w_final)
        local_aucs.append(roc_auc_score(y_test, y_pred))
        
    # 2. Train global federated model
    res_fed = fedavg_train(hospitals, X_test, y_test, rounds=15, epochs=5, lr=0.1, random_seed=RANDOM_SEED)
    global_auc = res_fed['round_aucs'][-1]
    
    print("\nIncentive Evaluation:")
    for idx in range(5):
        gain = global_auc - local_aucs[idx]
        print(f"  Hospital {idx+1} (Size={len(hospitals[idx][0]):3d}): Local AUC = {local_aucs[idx]:.4f} | Global AUC = {global_auc:.4f} | Gain = {gain:+.4f}")
        
    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    h_labels = [f"H1 (70%)", f"H2 (15%)", f"H3 (7.5%)", f"H4 (5%)", f"H5 (2.5%)"]
    
    ax.bar(h_labels, local_aucs, width=0.4, label='Local Model AUC', color='#A23B72', alpha=0.8)
    ax.axhline(global_auc, color='#2E86AB', linestyle='--', lw=2, label=f'Global Model AUC ({global_auc:.4f})')
    
    ax.set_ylabel('Model Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('Hospital Incentive Analysis: Local vs. Federated AUC', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, linestyle='--')
    
    plot_path = os.path.join(OUTPUT_DIR, "bound3_incentive_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Incentive plot to {plot_path}")

def test_boundary_4_dp_bounds(X, y):
    print("\n==================================================")
    print("BOUNDARY TEST 4: DIFFERENTIAL PRIVACY BOUNDS")
    print("==================================================")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    hospitals = partition_dirichlet(X_train, y_train, num_hospitals=5, alpha=0.5, random_seed=RANDOM_SEED)
    
    # Extreme privacy bounds:
    # 1. epsilon = 0.01 (extremely noisy/private)
    # 2. epsilon = 100.0 (virtually no noise added)
    dp_bounds = [0.01, 100.0]
    dp_aucs = []
    
    for eps in dp_bounds:
        print(f"Running federated loop with Epsilon = {eps}...")
        res = fedavg_train(
            hospitals, X_test, y_test, rounds=15, epochs=5, lr=0.1,
            random_seed=RANDOM_SEED, dp_enabled=True, epsilon=eps, delta=1e-5, clipping_norm=1.0
        )
        dp_aucs.append(res['round_aucs'][-1])
        
    print("\nPrivacy Boundary Results:")
    print(f"  Strict Privacy (Epsilon = 0.01): AUC = {dp_aucs[0]:.4f} (Destructive noise regime)")
    print(f"  Loose Privacy  (Epsilon = 100.0): AUC = {dp_aucs[1]:.4f} (Non-destructive noise regime)")
    
    return dp_bounds, dp_aucs

if __name__ == "__main__":
    X, y = load_data()
    test_boundary_1_hospital_scaling(X, y)
    test_boundary_2_heterogeneity_limit(X, y)
    test_boundary_3_cooperation_incentive(X, y)
    test_boundary_4_dp_bounds(X, y)
    print("\nAll boundary tests successfully executed.")
