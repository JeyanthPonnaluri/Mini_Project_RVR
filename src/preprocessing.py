"""
Data preprocessing module for TCGA-PRAD clinical stage classification.
Handles loading, target creation, and feature preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_clinical(file_path):
    """
    Load clinical TSV file into a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the clinical TSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded clinical dataframe
        
    Raises:
    -------
    FileNotFoundError
        If the file does not exist
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Successfully loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {str(e)}")


def load_survival_data(file_path):
    """
    Load survival TSV file containing overall survival (OS) metrics.
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Successfully loaded survival data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise Exception(f"Error loading survival file: {str(e)}")


def merge_clinical_survival(df_clinical, df_survival):
    """
    Merge clinical features dataframe with survival target columns (OS.time, OS).
    """
    # Attempt to merge on 'sample' column
    if 'sample' in df_clinical.columns and 'sample' in df_survival.columns:
        # Avoid column conflicts but keep OS targets
        survival_cols = ['sample', 'OS.time', 'OS']
        merged = pd.merge(df_clinical, df_survival[survival_cols], on='sample', how='inner')
        print(f"Merged clinical & survival dataframe shape: {merged.shape}")
        return merged
    else:
        # Fallback to patient barcode matching
        df_clinical_tmp = df_clinical.copy()
        df_survival_tmp = df_survival.copy()
        
        # TCGA patient barcodes are 12 chars: e.g. TCGA-HC-8264
        if 'bcr_patient_barcode' in df_clinical_tmp.columns:
            df_clinical_tmp['patient_id_tmp'] = df_clinical_tmp['bcr_patient_barcode']
        else:
            df_clinical_tmp['patient_id_tmp'] = df_clinical_tmp['sample'].str[:12] if 'sample' in df_clinical_tmp.columns else df_clinical_tmp.index
            
        df_survival_tmp['patient_id_tmp'] = df_survival_tmp['_PATIENT'] if '_PATIENT' in df_survival_tmp.columns else df_survival_tmp['sample'].str[:12]
        
        survival_cols = ['patient_id_tmp', 'OS.time', 'OS']
        merged = pd.merge(df_clinical_tmp, df_survival_tmp[survival_cols], on='patient_id_tmp', how='inner')
        merged = merged.drop(columns=['patient_id_tmp'])
        print(f"Merged clinical & survival dataframe (barcode fallback): {merged.shape}")
        return merged


def create_target(df):
    """
    Create binary target variable from pathologic T stage.
    
    Target definition:
    - 1 if stage starts with "T3" or "T4" (advanced stage)
    - 0 otherwise (early stage: T1, T2, etc.)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clinical dataframe with 'ajcc_pathologic_t.diagnoses' column
        
    Returns:
    --------
    tuple
        (df_filtered, target_series)
        - df_filtered: DataFrame with non-null stage values
        - target_series: Binary target (0 or 1)
        
    Raises:
    -------
    ValueError
        If target column not found
    """
    target_col = 'ajcc_pathologic_t.diagnoses'
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Filter out rows with null stage
    df_filtered = df[df[target_col].notna()].copy()
    print(f"Rows with valid stage: {len(df_filtered)}")
    
    # Create binary target: 1 for T3/T4 (advanced), 0 for others (early)
    stage_values = df_filtered[target_col].astype(str)
    target = ((stage_values.str.startswith('T3')) | (stage_values.str.startswith('T4'))).astype(int)
    
    print(f"Target distribution: {dict(target.value_counts().sort_index())}")
    
    return df_filtered, target


def preprocess_features(df, target_col='ajcc_pathologic_t.diagnoses', preprocessor=None):
    """
    Preprocess features: remove identifiers, encode categoricals, scale numericals.
    If preprocessor is provided, it uses the fitted preprocessor state to transform df.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clinical dataframe
    target_col : str
        Name of target column to exclude from features
    preprocessor : dict, optional
        Fitted preprocessor state dictionary containing columns, medians, and transformers.
        
    Returns:
    --------
    tuple
        (X_processed, feature_names, preprocessor_state)
    """
    # Identify columns to remove (identifiers and target)
    id_columns = [
        'sample', 'id', 'case_id', 'submitter_id', 'patient_id',
        'bcr_patient_barcode', 'sample_id', 'entity_id',
        target_col
    ]
    
    # Remove identifier columns that exist
    cols_to_drop = [col for col in id_columns if col in df.columns]
    df_features = df.drop(columns=cols_to_drop, errors='ignore').copy()
    
    if preprocessor is None:
        # Fit Mode
        print(f"[FIT] Features after removing identifiers: {df_features.shape[1]} columns")
        
        # Separate numerical and categorical columns
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        
        print(f"[FIT] Numerical features: {len(numerical_cols)}")
        print(f"[FIT] Categorical features: {len(categorical_cols)}")
        
        # Remove columns with too many missing values (>50%)
        threshold = len(df_features) * 0.5
        cols_to_keep = df_features.columns[df_features.notna().sum() > threshold].tolist()
        
        numerical_cols = [col for col in numerical_cols if col in cols_to_keep]
        categorical_cols = [col for col in categorical_cols if col in cols_to_keep]
        
        print(f"[FIT] After removing high-missing columns: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")
        
        # Fill missing values in numerical columns with median and compute medians map
        medians = {}
        for col in numerical_cols:
            val = df_features[col].median()
            medians[col] = 0.0 if pd.isna(val) else val
            df_features[col] = df_features[col].fillna(medians[col])
        
        # Fill missing values in categorical columns with 'Unknown' and convert all to string
        for col in categorical_cols:
            # Convert to string first to handle mixed types (bool, str, etc.)
            df_features[col] = df_features[col].astype(str)
            # Replace 'nan' string with 'Unknown'
            df_features[col] = df_features[col].replace('nan', 'Unknown')
            df_features[col] = df_features[col].fillna('Unknown')
        
        print(f"[FIT] Missing values handled - Numerical: median imputation, Categorical: 'Unknown'")
        
        # Create preprocessing pipeline
        preprocessor_obj = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='drop'
        )
        
        # Fit and transform
        X_processed = preprocessor_obj.fit_transform(df_features)
        
        # Get feature names
        feature_names = []
        feature_names.extend(numerical_cols)
        if len(categorical_cols) > 0:
            cat_encoder = preprocessor_obj.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        
        print(f"[FIT] Final feature matrix shape: {X_processed.shape}")
        
        preprocessor_state = {
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'medians': medians,
            'transformer': preprocessor_obj,
            'feature_names': feature_names
        }
        
        return X_processed, feature_names, preprocessor_state
        
    else:
        # Transform Mode
        numerical_cols = preprocessor['numerical_cols']
        categorical_cols = preprocessor['categorical_cols']
        medians = preprocessor['medians']
        transformer = preprocessor['transformer']
        feature_names = preprocessor['feature_names']
        
        # Align columns to fit configuration
        df_aligned = pd.DataFrame(index=df_features.index)
        for col in numerical_cols:
            if col in df_features.columns:
                df_aligned[col] = df_features[col].fillna(medians[col])
            else:
                df_aligned[col] = medians[col]
                
        for col in categorical_cols:
            if col in df_features.columns:
                df_aligned[col] = df_features[col].astype(str).replace('nan', 'Unknown').fillna('Unknown')
            else:
                df_aligned[col] = 'Unknown'
                
        # Transform using pre-fitted ColumnTransformer
        X_processed = transformer.transform(df_aligned)
        
        return X_processed, feature_names, preprocessor



def load_protein(file_path):
    """
    Load protein expression TSV file.
    Transposes it so that samples are rows and proteins are columns.
    
    Parameters:
    -----------
    file_path : str
        Path to protein TSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded and transposed protein dataframe
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Successfully loaded raw protein data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Transpose so that samples are rows and peptide targets are columns
        if 'peptide_target' in df.columns:
            df = df.set_index('peptide_target').T
            df.index.name = 'sample'
            df = df.reset_index()
            df.columns.name = None
            print(f"Transposed protein data: {df.shape[0]} samples, {df.shape[1]} columns")
            
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Protein file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading protein file: {str(e)}")


def merge_clinical_protein(clinical_df, protein_df, on_column='sample'):
    """
    Merge clinical and protein dataframes.
    
    Parameters:
    -----------
    clinical_df : pd.DataFrame
        Clinical dataframe
    protein_df : pd.DataFrame
        Protein expression dataframe
    on_column : str
        Column name to merge on
        
    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    # Find common merge column
    if on_column not in clinical_df.columns:
        # Try alternative column names
        for alt_col in ['case_id', 'patient_id', 'submitter_id', 'bcr_patient_barcode']:
            if alt_col in clinical_df.columns and alt_col in protein_df.columns:
                on_column = alt_col
                break
    
    print(f"Merging on column: {on_column}")
    merged_df = clinical_df.merge(protein_df, on=on_column, how='inner', suffixes=('_clinical', '_protein'))
    
    print(f"Merged dataframe: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    print(f"Samples retained: {len(merged_df)} / {len(clinical_df)} clinical, {len(merged_df)} / {len(protein_df)} protein")
    
    return merged_df


def preprocess_protein(protein_df, missing_threshold=0.3, preprocessor=None):
    """
    Preprocess protein expression data.
    If preprocessor is provided, it uses the fitted preprocessor state to transform protein_df.
    
    Parameters:
    -----------
    protein_df : pd.DataFrame
        Protein expression dataframe
    missing_threshold : float
        Maximum fraction of missing values allowed per protein
    preprocessor : dict, optional
        Fitted preprocessor state dict containing proteins_to_keep, medians, and scaler.
        
    Returns:
    --------
    tuple
        (X_protein, protein_names, preprocessor_state)
    """
    # Remove identifier columns
    id_columns = ['sample', 'case_id', 'patient_id', 'submitter_id', 'bcr_patient_barcode']
    protein_features = protein_df.drop(columns=[col for col in id_columns if col in protein_df.columns], errors='ignore')
    
    # Select only numerical columns
    protein_features = protein_features.select_dtypes(include=[np.number])
    
    if preprocessor is None:
        # Fit Mode
        print(f"[FIT] Protein features before filtering: {protein_features.shape[1]}")
        
        # Remove proteins with too many missing values
        missing_fraction = protein_features.isna().sum() / len(protein_features)
        proteins_to_keep = missing_fraction[missing_fraction <= missing_threshold].index.tolist()
        protein_features = protein_features[proteins_to_keep]
        
        print(f"[FIT] Proteins after removing high-missing (>{missing_threshold*100}%): {protein_features.shape[1]}")
        
        # Impute remaining missing with median
        medians = {}
        for col in protein_features.columns:
            val = protein_features[col].median()
            medians[col] = 0.0 if pd.isna(val) else val
            protein_features[col] = protein_features[col].fillna(medians[col])
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_protein = scaler.fit_transform(protein_features)
        
        preprocessor_state = {
            'proteins_to_keep': proteins_to_keep,
            'medians': medians,
            'scaler': scaler
        }
        
        return X_protein, proteins_to_keep, preprocessor_state
    else:
        # Transform Mode
        proteins_to_keep = preprocessor['proteins_to_keep']
        medians = preprocessor['medians']
        scaler = preprocessor['scaler']
        
        # Align columns
        df_aligned = pd.DataFrame(index=protein_features.index)
        for col in proteins_to_keep:
            if col in protein_features.columns:
                df_aligned[col] = protein_features[col].fillna(medians[col])
            else:
                df_aligned[col] = medians[col]
                
        # Transform using pre-fitted scaler
        X_protein = scaler.transform(df_aligned)
        
        return X_protein, proteins_to_keep, preprocessor


def apply_pca(X, n_components=None, variance_threshold=0.95, pca_model=None):
    """
    Apply PCA for dimensionality reduction.
    If pca_model is provided, use it to transform X.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_components : int, optional
        Number of components (if None, use variance_threshold)
    variance_threshold : float
        Cumulative variance to retain (if n_components is None)
    pca_model : PCA, optional
        Pre-fitted PCA model
        
    Returns:
    --------
    tuple
        (X_pca, pca_model, n_components_used)
    """
    from sklearn.decomposition import PCA
    
    if pca_model is None:
        # Fit Mode
        if n_components is None:
            # Determine n_components from variance threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        total_variance = np.sum(pca.explained_variance_ratio_)
        
        print(f"[FIT] PCA: {X.shape[1]} features -> {n_components} components")
        print(f"[FIT] Explained variance: {total_variance:.4f}")
        
        return X_pca, pca, n_components
    else:
        # Transform Mode
        X_pca = pca_model.transform(X)
        n_components = pca_model.n_components_
        
        return X_pca, pca_model, n_components


def apply_feature_selection(X, y, method='variance', threshold=0.01):
    """
    Apply feature selection.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    method : str
        'variance' or 'l1'
    threshold : float
        Threshold for variance or L1 regularization
        
    Returns:
    --------
    tuple
        (X_selected, selected_indices)
    """
    if method == 'variance':
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        selected_indices = selector.get_support(indices=True)
        
        print(f"Variance threshold: {X.shape[1]} -> {X_selected.shape[1]} features")
        
    elif method == 'l1':
        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import LogisticRegression
        
        selector = SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42),
            threshold=threshold
        )
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        print(f"L1 selection: {X.shape[1]} -> {X_selected.shape[1]} features")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X_selected, selected_indices


def generate_domain_shifted_cohort(X, y, times=None, events=None, shift_type='covariate', severity=0.5, random_seed=42):
    """
    Generate a domain-shifted cohort by perturbing features (covariate shift)
    or labels (concept/distribution shift) based on the test set.
    
    Parameters:
    -----------
    X : np.ndarray
        Base features to perturb
    y : np.ndarray
        Base labels
    times : np.ndarray, optional
        Base survival times
    events : np.ndarray, optional
        Base survival event indicators
    shift_type : str
        'covariate' (perturb P(X), preserve P(Y|X)) or 'concept' (perturb P(Y|X))
    severity : float
        Shift severity factor (0.0 means no shift)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_shifted, y_shifted, times_shifted, events_shifted)
    """
    np.random.seed(random_seed)
    X_shifted = X.copy()
    
    if shift_type == 'covariate':
        # Apply scaling and additive noise shift (P(X) changes, P(Y|X) preserved)
        # 1. Scale shift: multiply by a random scale vector near 1.0
        scale = np.random.uniform(1.0 - 0.25 * severity, 1.0 + 0.25 * severity, size=X.shape[1])
        X_shifted = X_shifted * scale
        # 2. Additive Gaussian noise
        noise = np.random.normal(0, 0.15 * severity, size=X.shape)
        X_shifted += noise
        
        y_shifted = y.copy()
        times_shifted = times.copy() if times is not None else None
        events_shifted = events.copy() if events is not None else None
        
    elif shift_type == 'concept':
        # Perturb label mapping P(Y|X) directly by flipping a fraction of targets
        y_shifted = y.copy()
        flip_fraction = np.clip(0.25 * severity, 0.0, 0.5)
        n_flips = int(len(y) * flip_fraction)
        if n_flips > 0:
            flip_idx = np.random.choice(len(y), size=n_flips, replace=False)
            y_shifted[flip_idx] = 1 - y_shifted[flip_idx]
        
        # Keep same features and survival
        times_shifted = times.copy() if times is not None else None
        events_shifted = events.copy() if events is not None else None
        
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")
        
    print(f"Generated synthetic domain-shifted cohort: {len(X_shifted)} samples, shift_type={shift_type}, severity={severity}")
    return X_shifted, y_shifted, times_shifted, events_shifted
