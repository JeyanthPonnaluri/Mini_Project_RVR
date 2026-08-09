import os
import sys
import numpy as np
import pandas as pd

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from federated import compute_rdp_step, compose_rdp_to_dp, calibrate_noise_multiplier

def validate_rdp_mathematics():
    print("--- RDP Accountant Mathematical Validation ---")
    
    # 1. Base Gaussian mechanism validation (q = 1.0)
    # Under q=1.0, RDP at order alpha should be exactly: alpha / (2 * sigma^2)
    alpha = 4.0
    sigma = 2.0
    expected_rdp_base = alpha / (2.0 * sigma**2) # 4 / 8 = 0.5
    actual_rdp_base = compute_rdp_step(q=1.0, sigma=sigma, alpha=alpha)
    print(f"Base Gaussian (q=1.0, sigma={sigma}, alpha={alpha}):")
    print(f"  Expected RDP step: {expected_rdp_base:.6f}")
    print(f"  Actual RDP step:   {actual_rdp_base:.6f}")
    assert np.isclose(actual_rdp_base, expected_rdp_base), "Base Gaussian RDP step calculation is incorrect!"
    
    # 2. Subsampled Gaussian mechanism validation (q < 1.0)
    # Let's check q=0.01, sigma=4.0, alpha=8.0
    q = 0.01
    sigma_sub = 4.0
    alpha_sub = 8.0
    # Analytical bound for small q: RDP <= (q^2 * alpha) / (2 * sigma^2 * (1 - q))
    analytical_upper_bound = (q**2 * alpha_sub) / (2.0 * sigma_sub**2 * (1.0 - q))
    actual_rdp_sub = compute_rdp_step(q=q, sigma=sigma_sub, alpha=alpha_sub)
    print(f"Subsampled Gaussian (q={q}, sigma={sigma_sub}, alpha={alpha_sub}):")
    print(f"  Analytical Upper Bound: {analytical_upper_bound:.10f}")
    relative_diff = abs(actual_rdp_sub - analytical_upper_bound) / analytical_upper_bound
    print(f"  Relative difference: {relative_diff:.6f}")
    assert relative_diff < 0.05, f"Subsampled Gaussian RDP step is too far from analytical approximation! Rel diff: {relative_diff:.6f}"
    
    # 3. Composition validation
    # If we compose 50 rounds with local_epochs = 3, total_steps = 150
    # For q=1.0, sigma=2.5, alpha=16.0, delta=1e-5
    steps = 150
    sigma_comp = 2.5
    alpha_comp = 16.0
    delta = 1e-5
    
    rdp_step = compute_rdp_step(q=1.0, sigma=sigma_comp, alpha=alpha_comp)
    expected_rdp_total = steps * rdp_step
    expected_epsilon = expected_rdp_total + np.log(1.0 / delta) / (alpha_comp - 1.0)
    
    # Run composer
    actual_epsilon, opt_alpha = compose_rdp_to_dp(q=1.0, sigma=sigma_comp, steps=steps, delta=delta, orders=[alpha_comp])
    print(f"Composition (steps={steps}, sigma={sigma_comp}, alpha={alpha_comp}, delta={delta}):")
    print(f"  Expected composed Epsilon: {expected_epsilon:.6f}")
    print(f"  Actual composed Epsilon:   {actual_epsilon:.6f}")
    assert np.isclose(actual_epsilon, expected_epsilon), "Composed RDP conversion is incorrect!"
    
    # 4. Calibration verification
    # Calibrate noise multiplier for target_epsilon = 3.0, steps = 100, delta = 1e-5
    target_epsilon = 3.0
    cal_sigma = calibrate_noise_multiplier(q=1.0, steps=100, target_epsilon=target_epsilon, delta=1e-5)
    composed_eps, opt_alpha_cal = compose_rdp_to_dp(q=1.0, sigma=cal_sigma, steps=100, delta=1e-5)
    print(f"Calibration (target Epsilon={target_epsilon}, steps=100, delta=1e-5):")
    print(f"  Calibrated noise multiplier sigma: {cal_sigma:.6f}")
    print(f"  Re-composed Epsilon at cal_sigma:  {composed_eps:.6f} (Optimal alpha={opt_alpha_cal})")
    assert np.isclose(composed_eps, target_epsilon, rtol=1e-2), "Calibrated noise multiplier did not reproduce the target Epsilon!"
    
    print("\n[SUCCESS] Mathematical RDP accountant validation PASSED successfully.")

def verify_privacy_unit():
    print("\n--- Privacy Unit Dataset Verification ---")
    clinical_path = "datasets/TCGA-PRAD.clinical.tsv/TCGA-PRAD.clinical.tsv"
    
    if not os.path.exists(clinical_path):
        clinical_path = "data/temp_clinical.tsv"
        
    if not os.path.exists(clinical_path):
        print("[WARNING] Clinical dataset not found. Skipping privacy unit check.")
        return
        
    print(f"Loading clinical dataset from: {clinical_path}")
    df = pd.read_csv(clinical_path, sep='\t')
    print(f"Loaded clinical data with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Find patient ID columns
    id_cols = [c for c in df.columns if any(x in c.lower() for x in ['patient', 'case_id', 'submitter_id', 'barcode', 'sample'])]
    print(f"Identified potential ID columns: {id_cols}")
    
    # Check uniqueness in key columns
    for col in id_cols:
        n_unique = df[col].nunique()
        n_total = len(df)
        print(f"Column '{col}': {n_unique} unique values out of {n_total} rows.")
        if n_unique == n_total:
            print(f"  -> Column '{col}' is a unique primary key! (1-to-1 mapping with rows)")
        else:
            print(f"  -> Column '{col}' contains duplicates!")
            
    # Check specifically if BCR patient barcode or patient ID is unique
    barcode_cols = [c for c in id_cols if 'barcode' in c.lower() or 'patient' in c.lower() or 'case_id' in c.lower()]
    is_unique_patient = False
    for col in barcode_cols:
        if df[col].nunique() == len(df):
            is_unique_patient = True
            print(f"\n[VERIFIED] Patient ID column '{col}' is unique across all rows.")
            break
            
    if is_unique_patient:
        print("\n[SUCCESS] Verification complete: Each patient corresponds to exactly ONE row in the clinical dataset.")
        print("  Therefore, sample-level Differential Privacy is mathematically equivalent to patient-level Differential Privacy in this study.")
        print("  This establishes the strict adjacency definition for the methodology: D and D' are adjacent if they differ by exactly one patient record.")
    else:
        print("\n[WARNING] No unique patient ID columns found. The privacy unit remains strictly sample-level (row-level), and patient-level DP is not guaranteed if multiple samples map to the same patient.")

if __name__ == "__main__":
    print("========================================")
    print("RDP MATHEMATICAL & PRIVACY UNIT VALIDATION")
    print("========================================")
    try:
        validate_rdp_mathematics()
        verify_privacy_unit()
        print("\n[SUCCESS] ALL CHECKS COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n[ERROR] VALIDATION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
