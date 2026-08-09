import os
import sys
import numpy as np
import pandas as pd

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocessing import preprocess_features, generate_domain_shifted_cohort, apply_pca
from federated import compute_rdp_step, compose_rdp_to_dp, calibrate_noise_multiplier

def test_zero_data_leakage():
    print("--- Running Test: Zero Data Leakage ---")
    # Generate random raw dataset with some missing values and float features
    np.random.seed(42)
    data = {
        'age': np.random.choice([40, 50, 60, np.nan], size=100),
        'psa': np.random.uniform(2.0, 20.0, size=100),
        'gleason': np.random.choice([6, 7, 8, np.nan], size=100),
        'ajcc_pathologic_t': np.random.choice(['T1', 'T2', 'T3', 'T4'], size=100)
    }
    df = pd.DataFrame(data)
    
    # Split raw dataset
    train_df = df.iloc[:80].copy()
    test_df = df.iloc[80:].copy()
    
    # Preprocess train and get preprocessor object
    X_train, feature_names, preprocessor = preprocess_features(train_df)
    
    # Preprocess test using train preprocessor
    X_test, _, _ = preprocess_features(test_df, preprocessor=preprocessor)
    
    # Verify shape compatibility
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions must match between train and test splits."
    
    # Ensure that test data scaling parameters do not leak back into the training preprocessor.
    # The preprocessor contains the scaling mean/std, which must be exactly equal to the train split statistics
    scaler = preprocessor['transformer'].named_transformers_['num']
    medians = preprocessor['medians']
    
    # Verify that the scaler is indeed fitted only on the training numeric features
    train_numeric = train_df[['age', 'psa', 'gleason']].copy()
    for col in train_numeric.columns:
        train_numeric[col] = train_numeric[col].fillna(medians[col])
        
    expected_means = np.mean(train_numeric.values, axis=0)
    
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5, err_msg="Scaler parameters leaked or incorrect!")
    print("  [PASS] Zero Data Leakage Verification.")

def test_rdp_accountant():
    print("--- Running Test: RDP Accountant Monotonicity & Validity ---")
    
    orders = [2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 32.0, 64.0, 100.0, 128.0, 200.0]
    delta = 1e-5
    
    # 1. Monotonicity with respect to noise multiplier (sigma)
    # Larger sigma (more noise) must result in a smaller epsilon (more privacy)
    eps_low, _ = compose_rdp_to_dp(q=1.0, sigma=0.5, steps=50, delta=delta, orders=orders)
    eps_high, _ = compose_rdp_to_dp(q=1.0, sigma=2.0, steps=50, delta=delta, orders=orders)
    assert eps_high < eps_low, f"Epsilon must decrease with larger noise. Got eps(0.5)={eps_low:.4f}, eps(2.0)={eps_high:.4f}"
    
    # 2. Monotonicity with respect to steps
    # More training steps (more composition) must result in a larger epsilon (less privacy)
    eps_few, _ = compose_rdp_to_dp(q=0.1, sigma=1.5, steps=10, delta=delta, orders=orders)
    eps_many, _ = compose_rdp_to_dp(q=0.1, sigma=1.5, steps=100, delta=delta, orders=orders)
    assert eps_many > eps_few, f"Epsilon must increase with steps. Got eps(10)={eps_few:.4f}, eps(100)={eps_many:.4f}"
    
    # 3. Monotonicity with respect to sampling rate (q)
    # Higher sampling rate must result in a larger epsilon
    eps_q_low, _ = compose_rdp_to_dp(q=0.05, sigma=2.0, steps=20, delta=delta, orders=orders)
    eps_q_high, _ = compose_rdp_to_dp(q=0.2, sigma=2.0, steps=20, delta=delta, orders=orders)
    assert eps_q_high > eps_q_low, f"Epsilon must increase with sampling rate. Got eps(0.05)={eps_q_low:.4f}, eps(0.2)={eps_q_high:.4f}"
    
    # 4. Monotonicity of target calibration
    # Smaller target epsilon (stricter privacy) must require a larger noise multiplier
    sig_loose = calibrate_noise_multiplier(q=1.0, steps=100, target_epsilon=5.0, delta=delta, orders=orders)
    sig_strict = calibrate_noise_multiplier(q=1.0, steps=100, target_epsilon=1.0, delta=delta, orders=orders)
    assert sig_strict > sig_loose, f"Stricter target epsilon must require larger noise multiplier. Got sig(5.0)={sig_loose:.4f}, sig(1.0)={sig_strict:.4f}"
    
    print("  [PASS] RDP Accountant Monotonicity Checks.")

def test_domain_shifted_cohort():
    print("--- Running Test: Domain-Shift Generation ---")
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = np.random.binomial(1, 0.5, size=50)
    times = np.random.uniform(10, 100, size=50)
    events = np.random.binomial(1, 0.4, size=50)
    
    # Covariate shift: P(X) changes, P(Y|X) preserved (labels, times, events must be unchanged)
    X_cov, y_cov, t_cov, e_cov = generate_domain_shifted_cohort(
        X, y, times, events, shift_type='covariate', severity=1.0, random_seed=42
    )
    
    # Verify shapes and dimensions
    assert X_cov.shape == X.shape
    # Labels and survival times must be identical
    np.testing.assert_allclose(y_cov, y)
    np.testing.assert_allclose(t_cov, times)
    np.testing.assert_allclose(e_cov, events)
    # Features must be modified
    assert not np.allclose(X_cov, X)
    
    # Concept shift: labels are modified
    X_con, y_con, t_con, e_con = generate_domain_shifted_cohort(
        X, y, times, events, shift_type='concept', severity=1.5, random_seed=42
    )
    
    # Features must be identical
    np.testing.assert_allclose(X_con, X)
    # A fraction of labels should be flipped, so they cannot be identical
    assert not np.allclose(y_con, y)
    
    print("  [PASS] Domain-Shift Generation Verification.")

if __name__ == "__main__":
    print("========================================")
    print("SCIENTIFIC FOUNDATION VERIFICATION SCRIPT")
    print("========================================")
    try:
        test_zero_data_leakage()
        test_rdp_accountant()
        test_domain_shifted_cohort()
        print("\n[SUCCESS] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[FAILURE] VERIFICATION FAILURE: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
