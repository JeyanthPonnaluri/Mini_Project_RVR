import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
    partition_dirichlet,
    partition_equal,
    fedavg_train,
    fedprox_train,
    calibrate_noise_multiplier,
    local_train_fedprox
)
from logistic_numpy import (
    local_train,
    local_train_dp,
    local_train_fedprox_dp,
    predict_proba,
    compute_loss
)
from shapley import compute_federated_shapley_values
from contribution import measure_hospital_contribution
from statistical_analysis import compute_bootstrap_ci_auc

def load_and_preprocess_prad(random_seed=42):
    clinical_path = "datasets/TCGA-PRAD.clinical.tsv/TCGA-PRAD.clinical.tsv"
    protein_path = "datasets/TCGA-PRAD.protein.tsv/TCGA-PRAD.protein.tsv"
    
    clinical_df = load_clinical(clinical_path)
    protein_df = load_protein(protein_path)
    
    merged_df = merge_clinical_protein(clinical_df, protein_df)
    df_filtered, target = create_target(merged_df)
    
    df_train, df_test, y_train, y_test = train_test_split(
        df_filtered, target, test_size=0.2, random_state=random_seed, stratify=target
    )
    
    protein_cols = [c for c in protein_df.columns if c not in ['sample', 'case_id', 'patient_id', 'submitter_id', 'bcr_patient_barcode']]
    clinical_cols = [c for c in df_filtered.columns if c not in protein_cols]
    
    X_train_clin, feature_names_clin, preprocessor_clin = preprocess_features(df_train[clinical_cols])
    X_test_clin, _, _ = preprocess_features(df_test[clinical_cols], preprocessor=preprocessor_clin)
    
    protein_part_train = df_train[['sample'] + protein_cols]
    X_train_prot, feature_names_prot, preprocessor_prot = preprocess_protein(protein_part_train)
    protein_part_test = df_test[['sample'] + protein_cols]
    X_test_prot, _, _ = preprocess_protein(protein_part_test, preprocessor=preprocessor_prot)
    
    X_train_prot_pca, pca_model, n_components = apply_pca(X_train_prot, variance_threshold=0.95)
    X_test_prot_pca, _, _ = apply_pca(X_test_prot, pca_model=pca_model)
    
    X_train = np.hstack([X_train_clin, X_train_prot_pca])
    X_test = np.hstack([X_test_clin, X_test_prot_pca])
    
    return X_train, np.array(y_train), X_test, np.array(y_test), df_filtered

def run_experiment_c_2d():
    print("\n==================================================")
    print("EXPERIMENT C: 2D Privacy x Heterogeneity Interaction Matrix (5 Seeds)")
    print("==================================================")
    
    alphas = [10.0, 1.0, 0.5, 0.1]
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    rounds = 15
    epochs = 3
    lr = 0.1
    mu = 0.5
    delta = 1e-5
    
    X_train, y_train, X_test, y_test, _ = load_and_preprocess_prad(random_seed=42)
    
    results = []
    
    for alpha in alphas:
        for eps in epsilons:
            aucs_avg = []
            aucs_prox = []
            benefits = []
            
            for seed_idx in range(5):
                seed = 42 + seed_idx
                hospitals = partition_dirichlet(X_train, y_train, num_hospitals=3, alpha=alpha, random_seed=seed)
                
                # 1. FedAvg + DP
                res_avg = fedavg_train(
                    hospitals, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr,
                    dp_enabled=True, epsilon=eps, delta=delta, clipping_norm=1.0, random_seed=seed
                )
                auc_avg = res_avg['round_aucs'][-1]
                aucs_avg.append(auc_avg)
                
                # 2. FedProx + DP
                res_prox = fedprox_train(
                    hospitals, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr, mu=mu,
                    dp_enabled=True, epsilon=eps, delta=delta, clipping_norm=1.0, random_seed=seed
                )
                auc_prox = res_prox['round_aucs'][-1]
                aucs_prox.append(auc_prox)
                
                benefits.append(auc_prox - auc_avg)
                
            mean_avg = np.mean(aucs_avg)
            std_avg = np.std(aucs_avg)
            mean_prox = np.mean(aucs_prox)
            std_prox = np.std(aucs_prox)
            mean_benefit = np.mean(benefits)
            std_benefit = np.std(benefits)
            
            # Simple 95% Confidence Interval for the benefit
            ci_half = 1.96 * std_benefit / np.sqrt(5)
            ci_lower = mean_benefit - ci_half
            ci_upper = mean_benefit + ci_half
            
            results.append({
                'Alpha': alpha,
                'Epsilon': eps,
                'FedAvg_AUC_mean': mean_avg,
                'FedAvg_AUC_std': std_avg,
                'FedProx_AUC_mean': mean_prox,
                'FedProx_AUC_std': std_prox,
                'Benefit_mean': mean_benefit,
                'Benefit_std': std_benefit,
                'Benefit_CI_lower': ci_lower,
                'Benefit_CI_upper': ci_upper
            })
            
            print(f"  Alpha: {alpha:4.1f} | Eps: {eps:4.1f} | Avg: {mean_avg:.4f}±{std_avg:.4f} | Prox: {mean_prox:.4f}±{std_prox:.4f} | Benefit: {mean_benefit:+.4f}±{std_benefit:.4f} [95% CI: {ci_lower:+.4f}, {ci_upper:+.4f}]")
            
    return results

def run_experiment_d_personalization():
    print("\n==================================================")
    print("EXPERIMENT D: Personalization Baseline Comparison")
    print("==================================================")
    
    rounds = 15
    epochs = 3
    lr = 0.1
    mu = 0.5
    
    X_train, y_train, X_test, y_test, _ = load_and_preprocess_prad(random_seed=42)
    
    # Partition training and test datasets using same Dirichlet distributions to represent local clients
    hospitals_train = partition_dirichlet(X_train, y_train, num_hospitals=3, alpha=0.5, random_seed=42)
    hospitals_test = partition_dirichlet(X_test, y_test, num_hospitals=3, alpha=0.5, random_seed=42)
    
    # Train Global Models
    res_avg = fedavg_train(hospitals_train, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr, random_seed=42)
    w_avg = res_avg['w_global']
    
    res_prox = fedprox_train(hospitals_train, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr, mu=mu, random_seed=42)
    w_prox = res_prox['w_global']
    
    results = []
    
    for k in range(3):
        X_tr_k, y_tr_k = hospitals_train[k]
        X_te_k, y_te_k = hospitals_test[k]
        
        # 1. Local-Only Model
        w_local, _ = local_train(X_tr_k, y_tr_k, w_init=np.zeros(X_train.shape[1]), epochs=epochs*rounds, lr=lr)
        auc_local = roc_auc_score(y_te_k, predict_proba(X_te_k, w_local))
        
        # 2. Global FedAvg evaluated on Client k
        auc_avg_k = roc_auc_score(y_te_k, predict_proba(X_te_k, w_avg))
        
        # 3. Global FedProx evaluated on Client k
        auc_prox_k = roc_auc_score(y_te_k, predict_proba(X_te_k, w_prox))
        
        # 4. Personalized FL (PFL) - fine-tune starting from Global FedProx
        w_pfl, _ = local_train(X_tr_k, y_tr_k, w_init=w_prox, epochs=2, lr=0.05)
        auc_pfl = roc_auc_score(y_te_k, predict_proba(X_te_k, w_pfl))
        
        results.append({
            'Hospital': f"Hospital {k+1}",
            'Local_AUC': auc_local,
            'FedAvg_AUC': auc_avg_k,
            'FedProx_AUC': auc_prox_k,
            'PFL_AUC': auc_pfl,
            'Delta_PFL': auc_pfl - auc_prox_k,
            'Delta_PFL_Local': auc_pfl - auc_local
        })
        
        print(f"  Hospital {k+1}: Local={auc_local:.4f} | FedAvg={auc_avg_k:.4f} | FedProx={auc_prox_k:.4f} | PFL={auc_pfl:.4f} | Gain={auc_pfl - auc_prox_k:+.4f}")
        
    return results

def run_experiment_f_shapley_privacy():
    print("\n==================================================")
    print("EXPERIMENT F: Shapley Valuation under Privacy Noise")
    print("==================================================")
    
    rounds = 5
    epochs = 2
    lr = 0.1
    delta = 1e-5
    
    X_train, y_train, X_test, y_test, _ = load_and_preprocess_prad(random_seed=42)
    hospitals = partition_dirichlet(X_train, y_train, num_hospitals=3, alpha=0.5, random_seed=42)
    
    epsilons = [None, 10.0, 5.0, 2.0, 1.0, 0.5]
    results = []
    
    for eps in epsilons:
        dp_enabled = eps is not None
        eps_label = str(eps) if dp_enabled else "No DP"
        
        shap_df = compute_federated_shapley_values(
            hospitals, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr,
            algorithm='fedavg', n_permutations=10, random_seed=42,
            dp_enabled=dp_enabled, epsilon=eps if dp_enabled else 1.0, delta=delta
        )
        
        scores = shap_df['shapley_value'].values
        # Determine ranks (1st, 2nd, 3rd)
        ranks = np.argsort(np.argsort(-scores)) + 1
        
        results.append({
            'Privacy': eps_label,
            'H1_Score': scores[0],
            'H2_Score': scores[1],
            'H3_Score': scores[2],
            'H1_Rank': int(ranks[0]),
            'H2_Rank': int(ranks[1]),
            'H3_Rank': int(ranks[2])
        })
        
        print(f"  Privacy: {eps_label:6s} | Shapley: H1={scores[0]:.4f} (Rank {ranks[0]}), H2={scores[1]:.4f} (Rank {ranks[1]}), H3={scores[2]:.4f} (Rank {ranks[2]})")
        
    return results

def run_experiment_g_consistency():
    print("\n==================================================")
    print("EXPERIMENT G: Contribution-Utility Consistency Test")
    print("==================================================")
    
    rounds = 10
    epochs = 3
    lr = 0.1
    
    X_train, y_train, X_test, y_test, _ = load_and_preprocess_prad(random_seed=42)
    hospitals = partition_dirichlet(X_train, y_train, num_hospitals=3, alpha=0.5, random_seed=42)
    
    # 1. Get baseline AUC with all hospitals
    res_all = fedavg_train(hospitals, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr, random_seed=42)
    baseline_auc = res_all['round_aucs'][-1]
    
    # 2. Compute Shapley values
    shap_df = compute_federated_shapley_values(
        hospitals, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr,
        algorithm='fedavg', n_permutations=10, random_seed=42
    )
    shap_scores = shap_df['shapley_value'].values
    
    # 3. Leave each hospital out and compute performance degradation
    degradations = []
    for k in range(3):
        hospitals_without_k = [hospitals[i] for i in range(3) if i != k]
        res_without = fedavg_train(hospitals_without_k, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr, random_seed=42)
        without_auc = res_without['round_aucs'][-1]
        degradation = baseline_auc - without_auc
        degradations.append(degradation)
        
        print(f"  Removed Hospital {k+1} | AUC: {without_auc:.4f} | Degradation: {degradation:+.4f} | Shapley Value: {shap_scores[k]:.4f}")
        
    # Compute correlation
    corr, _ = spearmanr(shap_scores, degradations)
    print(f"  Spearman Correlation between Shapley values and empirical degradation: {corr:.4f}")
    
    return {
        'shap_scores': list(shap_scores),
        'degradations': degradations,
        'correlation': corr
    }

def run_experiment_i_privacy_attack():
    print("\n==================================================")
    print("EXPERIMENT I: Empirical Privacy Attack (MIA)")
    print("==================================================")
    
    rounds = 15
    epochs = 3
    lr = 0.1
    delta = 1e-5
    
    X_train, y_train, X_test, y_test, _ = load_and_preprocess_prad(random_seed=42)
    hospitals = partition_equal(X_train, y_train, num_hospitals=3, random_seed=42)
    
    epsilons = [None, 10.0, 5.0, 2.0, 1.0, 0.5]
    results = []
    
    for eps in epsilons:
        dp_enabled = eps is not None
        eps_label = str(eps) if dp_enabled else "No DP"
        
        res = fedavg_train(
            hospitals, X_test, y_test, rounds=rounds, epochs=epochs, lr=lr,
            dp_enabled=dp_enabled, epsilon=eps if dp_enabled else 1.0, delta=delta, clipping_norm=1.0, random_seed=42
        )
        w_global = res['w_global']
        
        # Membership Inference Attack based on prediction confidence
        # Trainees (members) vs Test samples (non-members)
        y_prob_train = predict_proba(X_train, w_global)
        y_prob_test = predict_proba(X_test, w_global)
        
        # Attacker score = confidence in the correct class
        score_train = y_train * y_prob_train + (1 - y_train) * (1 - y_prob_train)
        score_test = y_test * y_prob_test + (1 - y_test) * (1 - y_prob_test)
        
        # Combine labels and scores
        mia_labels = np.hstack([np.ones_like(score_train), np.zeros_like(score_test)])
        mia_scores = np.hstack([score_train, score_test])
        
        # Compute MIA Attacker AUC
        mia_auc = roc_auc_score(mia_labels, mia_scores)
        
        # Compute attacker advantage
        # Threshold at median of combined scores
        median_score = np.median(mia_scores)
        pred_train = (score_train >= median_score).astype(int)
        pred_test = (score_test >= median_score).astype(int)
        
        tpr = np.mean(pred_train)
        fpr = np.mean(pred_test)
        advantage = tpr - fpr
        
        results.append({
            'Privacy': eps_label,
            'MIA_Attacker_AUC': mia_auc,
            'MIA_Advantage': advantage
        })
        
        print(f"  Privacy: {eps_label:6s} | MIA Attacker AUC: {mia_auc:.4f} | Attacker Advantage: {advantage:.4f}")
        
    return results

def main():
    print("==================================================")
    print("STARTING COMPREHENSIVE 9.5/10 EXPERIMENT SUITE")
    print("==================================================")
    
    suite_results = {}
    
    # 1. 2D Privacy x Heterogeneity Sweep
    suite_results['privacy_heterogeneity_2d'] = run_experiment_c_2d()
    
    # 2. Personalization Baselines (Local vs FedAvg vs FedProx vs PFL)
    suite_results['personalization_baselines'] = run_experiment_d_personalization()
    
    # 3. Shapley under Privacy Noise
    suite_results['shapley_privacy_noise'] = run_experiment_f_shapley_privacy()
    
    # 4. Shapley-Degradation Consistency
    suite_results['contribution_consistency'] = run_experiment_g_consistency()
    
    # 5. Membership Inference Attack (MIA)
    suite_results['mia_attack_results'] = run_experiment_i_privacy_attack()
    
    # Save all results
    os.makedirs("reports", exist_ok=True)
    with open("reports/comprehensive_9_5_results.json", "w") as f:
        json.dump(suite_results, f, indent=4)
        
    print("\n[SUCCESS] ALL COMPREHENSIVE 9.5 EXPERIMENTS LOGGED TO reports/comprehensive_9_5_results.json")
    print("==================================================")

if __name__ == "__main__":
    main()
