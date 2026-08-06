import os
import sys
import numpy as np

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

def run_integration_test():
    print("==================================================")
    print("RUNNING TCGA-PRAD DATASET INTEGRATION TEST")
    print("==================================================")
    
    clinical_path = "D:/Mini_project_JP/datasets/TCGA-PRAD.clinical.tsv/TCGA-PRAD.clinical.tsv"
    protein_path = "D:/Mini_project_JP/datasets/TCGA-PRAD.protein.tsv/TCGA-PRAD.protein.tsv"
    
    # 1. Load clinical
    print("\n1. Loading clinical data...")
    clinical_df = load_clinical(clinical_path)
    print(f"Clinical shape: {clinical_df.shape}")
    
    # 2. Load protein
    print("\n2. Loading protein data...")
    protein_df = load_protein(protein_path)
    print(f"Protein shape: {protein_df.shape}")
    
    # 3. Merge clinical and protein
    print("\n3. Merging clinical and protein datasets...")
    merged_df = merge_clinical_protein(clinical_df, protein_df)
    print(f"Merged shape: {merged_df.shape}")
    
    # 4. Create target
    print("\n4. Creating target variable...")
    df_filtered, target = create_target(merged_df)
    print(f"Filtered samples: {len(df_filtered)}")
    print(f"Target distribution: {dict(target.value_counts())}")
    
    # 5. Preprocess features
    print("\n5. Preprocessing clinical features...")
    protein_cols = [c for c in protein_df.columns if c not in ['sample', 'case_id', 'patient_id', 'submitter_id', 'bcr_patient_barcode']]
    clinical_cols = [c for c in df_filtered.columns if c not in protein_cols]
    
    X_clin, feature_names_clin, _ = preprocess_features(df_filtered[clinical_cols])
    print(f"Processed Clinical feature matrix shape: {X_clin.shape}")
    
    print("\n6. Preprocessing protein features...")
    protein_part = df_filtered[['sample'] + protein_cols]
    X_prot, feature_names_prot = preprocess_protein(protein_part)
    print(f"Processed Protein feature matrix shape: {X_prot.shape}")
    
    # 6. Apply PCA
    print("\n7. Applying PCA to protein features...")
    X_prot_pca, pca_model, n_components = apply_pca(X_prot, variance_threshold=0.95)
    print(f"Protein PCA shape: {X_prot_pca.shape} (reduced to {n_components} components)")
    
    # 7. Stack features
    X = np.hstack([X_clin, X_prot_pca])
    print(f"\nFinal feature matrix shape (stacked): {X.shape}")
    print("==================================================")
    print("ALL DATASET INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    try:
        run_integration_test()
    except Exception as e:
        print(f"\n[ERROR] INTEGRATION TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
