import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Add src directory to path
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
    partition_equal,
    partition_imbalanced,
    partition_dirichlet,
    fedavg_train,
    fedprox_train,
    generate_imbalanced_distribution
)
from shapley import compute_federated_shapley_values, plot_shapley_comparison
from evaluation import bootstrap_auc_ci, plot_roc_curve

# Output folders
REPORTS_DIR = "reports/journal_experiments"
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_SEED = 42

def load_and_preprocess_data():
    clinical_path = "D:/Mini_project_JP/datasets/TCGA-PRAD.clinical.tsv/TCGA-PRAD.clinical.tsv"
    protein_path = "D:/Mini_project_JP/datasets/TCGA-PRAD.protein.tsv/TCGA-PRAD.protein.tsv"
    
    print("Loading datasets...")
    clinical_df = load_clinical(clinical_path)
    protein_df = load_protein(protein_path)
    
    # 1. Clinical-only preprocessing
    print("Processing clinical-only cohort...")
    clin_filtered, target_clin = create_target(clinical_df)
    X_clin_only, _, _ = preprocess_features(clin_filtered)
    
    # 2. Multi-modal preprocessing
    print("Processing multi-modal cohort...")
    merged_df = merge_clinical_protein(clinical_df, protein_df)
    multi_filtered, target_multi = create_target(merged_df)
    
    # Preprocess clinical part of merged
    protein_cols = [c for c in protein_df.columns if c not in ['sample', 'case_id', 'patient_id', 'submitter_id', 'bcr_patient_barcode']]
    clinical_cols = [c for c in multi_filtered.columns if c not in protein_cols]
    X_clin, _, _ = preprocess_features(multi_filtered[clinical_cols])
    
    # Preprocess protein part
    protein_part = multi_filtered[['sample'] + protein_cols]
    X_prot, _ = preprocess_protein(protein_part)
    X_prot_pca, _, n_components = apply_pca(X_prot, variance_threshold=0.95)
    
    # Stack clinical + protein PCA
    X_multi = np.hstack([X_clin, X_prot_pca])
    
    return X_clin_only, target_clin, X_multi, target_multi

def run_experiment_1(X_clin_only, target_clin, X_multi, target_multi):
    print("\n==================================================")
    print("EXPERIMENT 1: MULTI-MODAL BASELINE COMPARISON")
    print("==================================================")
    
    # Split clinical-only
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clin_only, target_clin, test_size=0.2, random_state=RANDOM_SEED, stratify=target_clin
    )
    y_train_c, y_test_c = np.array(y_train_c), np.array(y_test_c)
    
    # Split multi-modal
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, target_multi, test_size=0.2, random_state=RANDOM_SEED, stratify=target_multi
    )
    y_train_m, y_test_m = np.array(y_train_m), np.array(y_test_m)
    
    # Train Centralized Clinical-Only Model
    print("Training Clinical-Only model...")
    w_clin_init = initialize_weights(X_train_c.shape[1], RANDOM_SEED)
    w_clin, _ = local_train(X_train_c, y_train_c, w_clin_init, epochs=50, lr=0.1)
    y_pred_clin = predict_proba(X_test_c, w_clin)
    clin_auc = roc_auc_score(y_test_c, y_pred_clin)
    mean_auc_c, lower_c, upper_c = bootstrap_auc_ci(y_test_c, y_pred_clin, n_bootstraps=500)
    
    # Train Centralized Multi-Modal Model (Unregularized)
    print("Training Multi-Modal (Unregularized) model...")
    w_multi_init = initialize_weights(X_train_m.shape[1], RANDOM_SEED)
    w_multi, _ = local_train(X_train_m, y_train_m, w_multi_init, epochs=50, lr=0.1)
    y_pred_multi = predict_proba(X_test_m, w_multi)
    multi_auc = roc_auc_score(y_test_m, y_pred_multi)
    mean_auc_m, lower_m, upper_m = bootstrap_auc_ci(y_test_m, y_pred_multi, n_bootstraps=500)

    # Train Centralized Multi-Modal Model (L2 Regularized)
    print("Training Multi-Modal (L2 Regularized) model...")
    w_multi_reg_init = initialize_weights(X_train_m.shape[1], RANDOM_SEED)
    w_multi_reg, _ = local_train(X_train_m, y_train_m, w_multi_reg_init, epochs=50, lr=0.1, l2_reg=0.1)
    y_pred_multi_reg = predict_proba(X_test_m, w_multi_reg)
    multi_reg_auc = roc_auc_score(y_test_m, y_pred_multi_reg)
    mean_auc_mr, lower_mr, upper_mr = bootstrap_auc_ci(y_test_m, y_pred_multi_reg, n_bootstraps=500)
    
    delta_auc = multi_reg_auc - clin_auc
    
    print(f"Clinical-Only AUC:            {clin_auc:.4f} (95% CI [{lower_c:.4f}, {upper_c:.4f}])")
    print(f"Multi-Modal (Unregularized):  {multi_auc:.4f} (95% CI [{lower_m:.4f}, {upper_m:.4f}])")
    print(f"Multi-Modal (L2 Regularized): {multi_reg_auc:.4f} (95% CI [{lower_mr:.4f}, {upper_mr:.4f}])")
    print(f"Delta AUC (Reg MM vs Clin):   {delta_auc:.4f}")
    
    # Plot comparative ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Clinical ROC and CI Band
    fpr_c, tpr_c, _ = roc_curve(y_test_c, y_pred_clin)
    ax.plot(fpr_c, tpr_c, color='#A23B72', lw=2, label=f'Clinical-Only (AUC = {clin_auc:.4f}, 95% CI [{lower_c:.4f}, {upper_c:.4f}])')
    
    # Multi-modal Unregularized ROC and CI Band
    fpr_m, tpr_m, _ = roc_curve(y_test_m, y_pred_multi)
    ax.plot(fpr_m, tpr_m, color='#2E86AB', lw=1.8, linestyle='--', label=f'Multi-Modal Unreg (AUC = {multi_auc:.4f}, 95% CI [{lower_m:.4f}, {upper_m:.4f}])')

    # Multi-modal L2 Regularized ROC and CI Band
    fpr_mr, tpr_mr, _ = roc_curve(y_test_m, y_pred_multi_reg)
    ax.plot(fpr_mr, tpr_mr, color='#2CA02C', lw=2.5, label=f'Multi-Modal L2 Reg (AUC = {multi_reg_auc:.4f}, 95% CI [{lower_mr:.4f}, {upper_mr:.4f}])')
    
    # Add bootstrap CI bands
    grid_fpr = np.linspace(0, 1, 100)
    for y_test_t, y_pred_t, color in [(y_test_c, y_pred_clin, '#A23B72'), (y_test_m, y_pred_multi, '#2E86AB'), (y_test_m, y_pred_multi_reg, '#2CA02C')]:
        y_test_arr = np.array(y_test_t)
        y_pred_arr = np.array(y_pred_t)
        tprs = []
        for _ in range(200):
            idx = np.random.choice(len(y_test_arr), len(y_test_arr), replace=True)
            if len(np.unique(y_test_arr[idx])) < 2:
                continue
            f, t, _ = roc_curve(y_test_arr[idx], y_pred_arr[idx])
            tprs.append(np.interp(grid_fpr, f, t))
        if tprs:
            tprs_lower = np.percentile(tprs, 2.5, axis=0)
            tprs_upper = np.percentile(tprs, 97.5, axis=0)
            ax.fill_between(grid_fpr, tprs_lower, tprs_upper, color=color, alpha=0.1)

    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison: Clinical vs Multi-Modal Staging', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "exp1_multimodal_roc.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved ROC curve plot to {plot_path}")
    
    return {
        'clin_auc': clin_auc, 'clin_ci': (lower_c, upper_c),
        'multi_auc': multi_auc, 'multi_ci': (lower_m, upper_m),
        'multi_reg_auc': multi_reg_auc, 'multi_reg_ci': (lower_mr, upper_mr),
        'delta_auc': delta_auc
    }

def run_experiment_2(X_multi, target_multi):
    print("\n==================================================")
    print("EXPERIMENT 2: PRIVACY-UTILITY TRADE-OFF STUDY")
    print("==================================================")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, target_multi, test_size=0.2, random_state=RANDOM_SEED, stratify=target_multi
    )
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    # Partition into 5 hospitals
    hospitals = partition_dirichlet(X_train, y_train, num_hospitals=5, alpha=0.5, random_seed=RANDOM_SEED)
    
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    aucs = []
    
    # 1. Run Non-DP baseline
    print("Running Non-Private Federated Learning...")
    non_dp_res = fedavg_train(
        hospitals, X_test, y_test, rounds=20, epochs=5, lr=0.1,
        random_seed=RANDOM_SEED, dp_enabled=False
    )
    non_dp_auc = non_dp_res['round_aucs'][-1]
    
    # 2. Run DP for each epsilon
    for eps in epsilons:
        print(f"Running DP Federated Learning with epsilon = {eps}...")
        dp_res = fedavg_train(
            hospitals, X_test, y_test, rounds=20, epochs=5, lr=0.1,
            random_seed=RANDOM_SEED, dp_enabled=True, epsilon=eps, delta=1e-5, clipping_norm=1.0
        )
        aucs.append(dp_res['round_aucs'][-1])
        
    print("\nPrivacy-Utility Results:")
    print(f"Non-DP Baseline AUC: {non_dp_auc:.4f}")
    for eps, auc_val in zip(epsilons, aucs):
         print(f"Epsilon = {eps:4.1f} AUC:  {auc_val:.4f} (Utility loss: {non_dp_auc - auc_val:.4f})")
         
    # Plot tradeoff
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epsilons, aucs, marker='o', color='#2E86AB', lw=2, label='DP-FedAvg')
    ax.axhline(non_dp_auc, color='red', linestyle='--', label='Non-DP Baseline')
    
    # Add annotations
    for i, txt in enumerate(aucs):
        ax.annotate(f"{txt:.4f}", (epsilons[i], aucs[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    ax.set_xscale('log')
    ax.set_xticks(epsilons)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Privacy Budget (Epsilon ε) - Log Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Model Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('Differential Privacy Privacy-Utility Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "exp2_privacy_utility.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved Privacy-Utility plot to {plot_path}")
    
    return {
        'epsilons': epsilons, 'aucs': aucs, 'non_dp_auc': non_dp_auc
    }

def run_experiment_3(X_multi, target_multi):
    print("\n==================================================")
    print("EXPERIMENT 3: NON-IID CONVERGENCE STABILITY STUDY")
    print("==================================================")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, target_multi, test_size=0.2, random_state=RANDOM_SEED, stratify=target_multi
    )
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    # Strong Non-IID Dirichlet Partition (alpha = 0.1)
    hospitals = partition_dirichlet(X_train, y_train, num_hospitals=5, alpha=0.1, random_seed=RANDOM_SEED)
    
    # 1. Run FedAvg (which matches FedProx with mu=0.0)
    print("Running FedAvg under strong non-IID partitions...")
    fedavg_res = fedavg_train(
        hospitals, X_test, y_test, rounds=25, epochs=5, lr=0.1, random_seed=RANDOM_SEED
    )
    fedavg_aucs = fedavg_res['round_aucs']
    fedavg_std = np.std(fedavg_aucs[-5:])
    
    mu_values = [0.01, 0.1, 0.5, 1.0]
    prox_results = {}
    
    for mu in mu_values:
        print(f"Running FedProx with mu = {mu} under strong non-IID partitions...")
        fedprox_res = fedprox_train(
            hospitals, X_test, y_test, rounds=25, epochs=5, lr=0.1, mu=mu, random_seed=RANDOM_SEED
        )
        prox_results[mu] = {
            'aucs': fedprox_res['round_aucs'],
            'std': np.std(fedprox_res['round_aucs'][-5:])
        }
        
    print("\nConvergence Stability (AUC Standard Deviation in last 5 rounds):")
    print(f"FedAvg (mu = 0.0):  {fedavg_std:.4f} (Final AUC: {fedavg_aucs[-1]:.4f})")
    for mu in mu_values:
        print(f"FedProx (mu = {mu:4.2f}): {prox_results[mu]['std']:.4f} (Final AUC: {prox_results[mu]['aucs'][-1]:.4f})")
        
    # Plot curves
    fig, ax = plt.subplots(figsize=(9, 6))
    rounds_range = range(1, 26)
    
    ax.plot(rounds_range, fedavg_aucs, label=f'FedAvg (μ=0.0, std={fedavg_std:.4f})', color='#A23B72', lw=2)
    colors = ['#F18F01', '#2E86AB', '#C73E1D', '#3B1F2B']
    
    for idx, mu in enumerate(mu_values):
        ax.plot(rounds_range, prox_results[mu]['aucs'], 
                label=f'FedProx (μ={mu}, std={prox_results[mu]["std"]:.4f})', 
                color=colors[idx], lw=1.8, linestyle='--' if mu < 0.1 else '-')
        
    ax.set_xlabel('Communication Rounds', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Model Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Stability in Non-IID Dirichlet Skew (α = 0.1)', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "exp3_noniid_convergence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved Non-IID convergence plot to {plot_path}")
    
    return {
        'fedavg_aucs': fedavg_aucs, 'fedavg_std': fedavg_std,
        'prox_results': prox_results
    }

def run_experiment_4(X_multi, target_multi):
    print("\n==================================================")
    print("EXPERIMENT 4: CLIENT VALUATION DISCREPANCY STUDY")
    print("==================================================")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, target_multi, test_size=0.2, random_state=RANDOM_SEED, stratify=target_multi
    )
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    # Generate 3 hospitals with high size imbalance (e.g. 70%, 20%, 10%)
    np.random.seed(RANDOM_SEED)
    total_samples = len(y_train)
    indices = np.random.permutation(total_samples)
    
    idx_h1 = indices[:int(0.7 * total_samples)]
    idx_h2 = indices[int(0.7 * total_samples):int(0.9 * total_samples)]
    idx_h3 = indices[int(0.9 * total_samples):]
    
    hospitals = [
        (X_train[idx_h1], y_train[idx_h1]),
        (X_train[idx_h2], y_train[idx_h2]),
        (X_train[idx_h3], y_train[idx_h3])
    ]
    
    print(f"Hospitals sizes: H1: {len(idx_h1)}, H2: {len(idx_h2)}, H3: {len(idx_h3)}")
    
    # 1. Run Leave-One-Out (LOO) contribution
    # We do a simplified custom LOO here
    from contribution import measure_hospital_contribution
    contribution_df = measure_hospital_contribution(
        hospitals, X_test, y_test, rounds=15, epochs=5, lr=0.1,
        algorithm='fedavg', random_seed=RANDOM_SEED
    )
    
    # 2. Run Federated Shapley Value (SV) calculation
    shapley_df = compute_federated_shapley_values(
        hospitals, X_test, y_test, rounds=15, epochs=5, lr=0.1,
        algorithm='fedavg', n_permutations=20, random_seed=RANDOM_SEED
    )
    
    # Print comparison
    merged = pd.merge(
        shapley_df[['hospital_id', 'num_samples', 'shapley_value']],
        contribution_df[['hospital_id', 'contribution']],
        on='hospital_id'
    )
    print("\nClient Valuation Comparison Table:")
    print(merged)
    
    # Plot comparison using the custom plotting function
    plot_path = os.path.join(REPORTS_DIR, "exp4_valuation_comparison.png")
    fig = plot_shapley_comparison(shapley_df, contribution_df, save_path=plot_path)
    plt.close(fig)
    
    return merged

if __name__ == "__main__":
    X_clin, target_clin, X_multi, target_multi = load_and_preprocess_data()
    
    exp1_res = run_experiment_1(X_clin, target_clin, X_multi, target_multi)
    exp2_res = run_experiment_2(X_multi, target_multi)
    exp3_res = run_experiment_3(X_multi, target_multi)
    exp4_res = run_experiment_4(X_multi, target_multi)
    
    print("\nAll experiments successfully executed and plots generated in reports/journal_experiments/")
