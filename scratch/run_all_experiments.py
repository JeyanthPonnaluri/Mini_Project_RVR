import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import from src
from preprocessing import (
    load_clinical,
    load_protein,
    merge_clinical_protein,
    create_target,
    preprocess_features,
    preprocess_protein,
    apply_pca,
    generate_domain_shifted_cohort
)
from federated import (
    partition_equal,
    partition_dirichlet,
    fedavg_train,
    fedprox_train,
    fedavg_cox_train
)
from model import (
    train_regularized_model,
    train_non_linear_model
)
from logistic_numpy import (
    local_train,
    predict_proba
)
from experiments import (
    centralized_train_numpy
)
from statistical_analysis import (
    compute_bootstrap_ci_auc,
    compute_bootstrap_ci_cindex,
    run_shapley_stability_analysis,
    run_dirichlet_heterogeneity_sweep
)

def main():
    print("==================================================")
    # 0. Load Datasets
    print("Step 0: Loading clinical and protein expression data...")
    clinical_path = "datasets/TCGA-PRAD.clinical.tsv/TCGA-PRAD.clinical.tsv"
    protein_path = "datasets/TCGA-PRAD.protein.tsv/TCGA-PRAD.protein.tsv"
    survival_path = "datasets/TCGA-PRAD.survival.tsv/TCGA-PRAD.survival.tsv"
    
    RANDOM_SEED = 42
    
    from preprocessing import load_clinical, load_protein
    clinical_df = load_clinical(clinical_path)
    protein_df = load_protein(protein_path)
    
    merged_df = merge_clinical_protein(clinical_df, protein_df)
    df_filtered, target = create_target(merged_df)
    
    # 1. Zero-Leakage train-test split and preprocessing
    print("Step 1: Splitting and preprocessing clinical + protein data...")
    df_train, df_test, y_train, y_test = train_test_split(
        df_filtered, target, test_size=0.2, random_state=RANDOM_SEED, stratify=target
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    protein_cols = [c for c in protein_df.columns if c not in ['sample', 'case_id', 'patient_id', 'submitter_id', 'bcr_patient_barcode']]
    clinical_cols = [c for c in df_filtered.columns if c not in protein_cols]
    
    # 1. Preprocess Clinical
    X_train_clin, feature_names_clin, preprocessor_clin = preprocess_features(df_train[clinical_cols])
    X_test_clin, _, _ = preprocess_features(df_test[clinical_cols], preprocessor=preprocessor_clin)
    
    # 2. Preprocess Protein
    protein_part_train = df_train[['sample'] + protein_cols]
    X_train_prot, feature_names_prot, preprocessor_prot = preprocess_protein(protein_part_train)
    protein_part_test = df_test[['sample'] + protein_cols]
    X_test_prot, _, _ = preprocess_protein(protein_part_test, preprocessor=preprocessor_prot)
    
    # 3. Apply PCA
    X_train_prot_pca, pca_model, n_components = apply_pca(X_train_prot, variance_threshold=0.95)
    X_test_prot_pca, _, _ = apply_pca(X_test_prot, pca_model=pca_model)
    
    # 4. Concat
    X_train = np.hstack([X_train_clin, X_train_prot_pca])
    X_test = np.hstack([X_test_clin, X_test_prot_pca])
    
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    results = {}
    
    # --- Experiment 1: Centralized Baselines ---
    print("\nStep 2: Training Centralized Baselines...")
    # Scikit-learn Lasso (L1)
    model_l1 = train_regularized_model(X_train, y_train, penalty='l1', C=1.0)
    y_prob_l1 = model_l1.predict_proba(X_test)[:, 1]
    auc_l1 = roc_auc_score(y_test, y_prob_l1)
    
    # Scikit-learn Ridge (L2)
    model_l2 = train_regularized_model(X_train, y_train, penalty='l2', C=1.0)
    y_prob_l2 = model_l2.predict_proba(X_test)[:, 1]
    auc_l2 = roc_auc_score(y_test, y_prob_l2)
    
    # Neural Network MLP
    model_mlp = train_non_linear_model(X_train, y_train, model_type='mlp', random_seed=RANDOM_SEED)
    y_prob_mlp = model_mlp.predict_proba(X_test)[:, 1]
    auc_mlp = roc_auc_score(y_test, y_prob_mlp)
    
    # Custom NumPy Model
    numpy_model = centralized_train_numpy(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, random_seed=RANDOM_SEED)
    y_prob_np = predict_proba(X_test, numpy_model['w'])
    auc_np = roc_auc_score(y_test, y_prob_np)
    
    # Bootstrap CI for custom NumPy
    low_np, high_np, _ = compute_bootstrap_ci_auc(y_test, y_prob_np, n_bootstraps=200, random_seed=RANDOM_SEED)
    
    print(f"  Lasso AUC: {auc_l1:.4f}")
    print(f"  Ridge AUC: {auc_l2:.4f}")
    print(f"  MLP NN AUC: {auc_mlp:.4f}")
    print(f"  Custom NumPy AUC: {auc_np:.4f} [95% CI: {low_np:.4f}-{high_np:.4f}]")
    
    results['centralized_baselines'] = {
        'lasso_auc': auc_l1,
        'ridge_auc': auc_l2,
        'mlp_auc': auc_mlp,
        'numpy_auc': auc_np,
        'numpy_auc_ci_lower': low_np,
        'numpy_auc_ci_upper': high_np
    }
    
    # --- Experiment 2: Federated Optimization & Heterogeneity Sweep ---
    print("\nStep 3: Running Dirichlet sweeps & FedProx weight-drift analysis...")
    sweep_df = run_dirichlet_heterogeneity_sweep(
        X_train, y_train, X_test, y_test,
        num_hospitals=3, alphas=[10.0, 1.0, 0.5, 0.1],
        rounds=15, epochs=3, lr=0.1, mu=0.5, random_seed=RANDOM_SEED
    )
    print("  Sweep results:")
    print(sweep_df.to_string(index=False))
    
    results['heterogeneity_sweep'] = sweep_df.to_dict(orient='records')
    
    # --- Experiment 3: Privacy-Utility Curves ---
    print("\nStep 4: Running Privacy-Utility Sweeps (Epsilon vs AUC)...")
    hospitals_equal = partition_equal(X_train, y_train, num_hospitals=3, random_seed=RANDOM_SEED)
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    dp_aucs = []
    
    for eps in epsilons:
        res = fedavg_train(
            hospitals_equal, X_test, y_test, rounds=15, epochs=3, lr=0.1,
            dp_enabled=True, epsilon=eps, delta=1e-5, clipping_norm=1.0, random_seed=RANDOM_SEED
        )
        dp_aucs.append(res['round_aucs'][-1])
        
    print("  Privacy curve:")
    for eps, auc in zip(epsilons, dp_aucs):
        print(f"  Epsilon: {eps:.1f} -> Test AUC: {auc:.4f}")
        
    results['privacy_utility_sweep'] = {
        'epsilons': epsilons,
        'aucs': dp_aucs
    }
    
    # --- Experiment 4: Survival modeling Cox baselines ---
    print("\nStep 5: Running Cox Proportional Hazards modeling...")
    if os.path.exists(survival_path):
        survival_df = pd.read_csv(survival_path, sep='\t')
        
        # Keep samples in clinical
        survival_df = survival_df[survival_df['sample'].isin(df_filtered['sample'])]
        # Extract features and targets
        df_surv_feat = df_filtered[df_filtered['sample'].isin(survival_df['sample'])].copy()
        
        # Split survival
        df_tr_s, df_te_s, times_tr_s, times_te_s = train_test_split(
            df_surv_feat, survival_df['OS.time'].values, test_size=0.2, random_state=RANDOM_SEED
        )
        _, _, events_tr_s, events_te_s = train_test_split(
            df_surv_feat, survival_df['OS'].values, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Preprocess survival clinical
        X_tr_s_clin, _, preprocessor_s_clin = preprocess_features(df_tr_s[clinical_cols])
        X_te_s_clin, _, _ = preprocess_features(df_te_s[clinical_cols], preprocessor=preprocessor_s_clin)
        
        # Preprocess survival protein
        protein_part_tr_s = df_tr_s[['sample'] + protein_cols]
        X_tr_s_prot, _, preprocessor_s_prot = preprocess_protein(protein_part_tr_s)
        protein_part_te_s = df_te_s[['sample'] + protein_cols]
        X_te_s_prot, _, _ = preprocess_protein(protein_part_te_s, preprocessor=preprocessor_s_prot)
        
        # PCA survival
        X_tr_s_prot_pca, pca_s_model, n_comp_s = apply_pca(X_tr_s_prot, variance_threshold=0.95)
        X_te_s_prot_pca, _, _ = apply_pca(X_te_s_prot, pca_model=pca_s_model)
        
        # Concat survival features
        X_tr_s = np.hstack([X_tr_s_clin, X_tr_s_prot_pca])
        X_te_s = np.hstack([X_te_s_clin, X_te_s_prot_pca])
        
        # Partition to 3 clinics
        hospitals_surv = [
            (X_tr_s[:len(X_tr_s)//3], times_tr_s[:len(X_tr_s)//3], events_tr_s[:len(X_tr_s)//3]),
            (X_tr_s[len(X_tr_s)//3:2*len(X_tr_s)//3], times_tr_s[len(X_tr_s)//3:2*len(X_tr_s)//3], events_tr_s[len(X_tr_s)//3:2*len(X_tr_s)//3]),
            (X_tr_s[2*len(X_tr_s)//3:], times_tr_s[2*len(X_tr_s)//3:], events_tr_s[2*len(X_tr_s)//3:])
        ]
        
        # Train Federated Cox survival model
        cox_res = fedavg_cox_train(
            hospitals_surv, X_te_s, times_te_s, events_te_s,
            rounds=20, epochs=5, lr=0.01, random_seed=RANDOM_SEED
        )
        
        # Compute bootstrap CI
        low_ci_c, upper_ci_c, _ = compute_bootstrap_ci_cindex(
            cox_res['w_global'], X_te_s, times_te_s, events_te_s, n_bootstraps=100, random_seed=RANDOM_SEED
        )
        
        print(f"  Federated Cox model final C-index: {cox_res['round_c_indices'][-1]:.4f}")
        print(f"  95% Bootstrap Confidence Interval: [{low_ci_c:.4f}, {upper_ci_c:.4f}]")
        
        results['survival_analysis'] = {
            'c_index': cox_res['round_c_indices'][-1],
            'c_index_ci_lower': low_ci_c,
            'c_index_ci_upper': upper_ci_c
        }
    else:
        print("  [WARNING] Survival dataset not found, skipping Cox experiments.")
        
    # --- Experiment 5: Shapley Value Stability Audit ---
    print("\nStep 6: Running Shapley value seed stability analysis...")
    hospitals_shap = partition_dirichlet(X_train, y_train, num_hospitals=3, alpha=0.5, random_seed=RANDOM_SEED)
    stability_df = run_shapley_stability_analysis(
        hospitals_shap, X_test, y_test, rounds=5, epochs=2, lr=0.1, n_seeds=5
    )
    print("  Shapley Seed Stability Summary:")
    print(stability_df.to_string(index=False))
    
    results['shapley_stability'] = stability_df.to_dict(orient='records')
    
    # --- Experiment 6: Controlled Domain Shift generalizability ---
    print("\nStep 7: Evaluating generalizability under covariate and concept shifts...")
    severities = [0.0, 0.5, 1.0, 1.5, 2.0]
    covariate_aucs = []
    concept_aucs = []
    
    # Train consensus federated model (15 rounds)
    hospitals_fed = partition_dirichlet(X_train, y_train, num_hospitals=3, alpha=0.5, random_seed=RANDOM_SEED)
    res_fed = fedavg_train(hospitals_fed, X_test, y_test, rounds=15, epochs=3, lr=0.1, random_seed=RANDOM_SEED)
    w_global = res_fed['w_global']
    
    for sev in severities:
        # Covariate shift P(X) changes, P(Y|X) preserved
        X_shifted_cov, y_shifted_cov, _, _ = generate_domain_shifted_cohort(
            X_test, y_test, None, None, shift_type='covariate', severity=sev, random_seed=RANDOM_SEED
        )
        y_prob_cov = predict_proba(X_shifted_cov, w_global)
        auc_cov = roc_auc_score(y_shifted_cov, y_prob_cov)
        covariate_aucs.append(auc_cov)
        
        # Concept shift P(Y|X) changes
        X_shifted_con, y_shifted_con, _, _ = generate_domain_shifted_cohort(
            X_test, y_test, None, None, shift_type='concept', severity=sev, random_seed=RANDOM_SEED
        )
        y_prob_con = predict_proba(X_shifted_con, w_global)
        auc_con = roc_auc_score(y_shifted_con, y_prob_con)
        concept_aucs.append(auc_con)
        
    print("  Domain Shift Results:")
    for sev, cov_auc, con_auc in zip(severities, covariate_aucs, concept_aucs):
        print(f"  Severity: {sev:.1f} -> Covariate Shift AUC: {cov_auc:.4f}, Concept Shift AUC: {con_auc:.4f}")
        
    results['domain_shift'] = {
        'severities': severities,
        'covariate_aucs': covariate_aucs,
        'concept_aucs': concept_aucs
    }
    
    # Save results to file
    os.makedirs("reports", exist_ok=True)
    with open("reports/empirical_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n[SUCCESS] ALL EXPERIMENTAL SWEEPS RUN AND LOGGED TO reports/empirical_results.json")
    print("==================================================")

if __name__ == "__main__":
    main()
