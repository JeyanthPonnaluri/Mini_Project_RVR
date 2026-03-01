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


def preprocess_features(df, target_col='ajcc_pathologic_t.diagnoses'):
    """
    Preprocess features: remove identifiers, encode categoricals, scale numericals.
    
    Steps:
    1. Remove identifier and target columns
    2. Separate numerical and categorical features
    3. Handle missing values (impute or drop)
    4. OneHotEncode categorical features
    5. StandardScale numerical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clinical dataframe
    target_col : str
        Name of target column to exclude from features
        
    Returns:
    --------
    tuple
        (X_processed, feature_names, preprocessor)
        - X_processed: Processed feature matrix (numpy array)
        - feature_names: List of feature names after preprocessing
        - preprocessor: Fitted ColumnTransformer for future use
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
    
    print(f"Features after removing identifiers: {df_features.shape[1]} columns")
    
    # Separate numerical and categorical columns
    numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    # Remove columns with too many missing values (>50%)
    threshold = len(df_features) * 0.5
    cols_to_keep = df_features.columns[df_features.notna().sum() > threshold].tolist()
    
    numerical_cols = [col for col in numerical_cols if col in cols_to_keep]
    categorical_cols = [col for col in categorical_cols if col in cols_to_keep]
    
    print(f"After removing high-missing columns: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")
    
    # Fill missing values in numerical columns with median
    for col in numerical_cols:
        if df_features[col].isna().any():
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)
    
    # Fill missing values in categorical columns with 'Unknown' and convert all to string
    for col in categorical_cols:
        # Convert to string first to handle mixed types (bool, str, etc.)
        df_features[col] = df_features[col].astype(str)
        # Replace 'nan' string with 'Unknown'
        df_features[col] = df_features[col].replace('nan', 'Unknown')
        if df_features[col].isna().any():
            df_features[col] = df_features[col].fillna('Unknown')
    
    print(f"Missing values handled - Numerical: median imputation, Categorical: 'Unknown'")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(df_features)
    
    # Get feature names
    feature_names = []
    
    # Numerical feature names
    feature_names.extend(numerical_cols)
    
    # Categorical feature names (after one-hot encoding)
    if len(categorical_cols) > 0:
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names)
    
    print(f"Final feature matrix shape: {X_processed.shape}")
    print(f"Contains NaN: {np.isnan(X_processed).any()}")
    
    if np.isnan(X_processed).any():
        print(f"WARNING: NaN values detected! Count: {np.isnan(X_processed).sum()}")
    else:
        print(f"✓ No NaN values in final feature matrix")
    
    return X_processed, feature_names, preprocessor



def load_protein(file_path):
    """
    Load protein expression TSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to protein TSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded protein dataframe
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Successfully loaded protein data: {df.shape[0]} rows, {df.shape[1]} columns")
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


def preprocess_protein(protein_df, missing_threshold=0.3):
    """
    Preprocess protein expression data.
    
    Steps:
    1. Remove identifier columns
    2. Remove proteins with >30% missing values
    3. Impute remaining missing with median
    4. Standardize
    
    Parameters:
    -----------
    protein_df : pd.DataFrame
        Protein expression dataframe
    missing_threshold : float
        Maximum fraction of missing values allowed per protein
        
    Returns:
    --------
    tuple
        (X_protein, protein_names)
    """
    # Remove identifier columns
    id_columns = ['sample', 'case_id', 'patient_id', 'submitter_id', 'bcr_patient_barcode']
    protein_features = protein_df.drop(columns=[col for col in id_columns if col in protein_df.columns], errors='ignore')
    
    # Select only numerical columns
    protein_features = protein_features.select_dtypes(include=[np.number])
    
    print(f"Protein features before filtering: {protein_features.shape[1]}")
    
    # Remove proteins with too many missing values
    missing_fraction = protein_features.isna().sum() / len(protein_features)
    proteins_to_keep = missing_fraction[missing_fraction <= missing_threshold].index.tolist()
    protein_features = protein_features[proteins_to_keep]
    
    print(f"Proteins after removing high-missing (>{missing_threshold*100}%): {protein_features.shape[1]}")
    
    # Impute remaining missing with median
    for col in protein_features.columns:
        if protein_features[col].isna().any():
            median_val = protein_features[col].median()
            protein_features[col] = protein_features[col].fillna(median_val)
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_protein = scaler.fit_transform(protein_features)
    
    print(f"Final protein matrix: {X_protein.shape}")
    print(f"Contains NaN: {np.isnan(X_protein).any()}")
    
    return X_protein, protein_features.columns.tolist()


def apply_pca(X, n_components=None, variance_threshold=0.95):
    """
    Apply PCA for dimensionality reduction.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_components : int, optional
        Number of components (if None, use variance_threshold)
    variance_threshold : float
        Cumulative variance to retain (if n_components is None)
        
    Returns:
    --------
    tuple
        (X_pca, pca_model, n_components_used)
    """
    from sklearn.decomposition import PCA
    
    if n_components is None:
        # Determine n_components from variance threshold
        pca_temp = PCA()
        pca_temp.fit(X)
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    total_variance = np.sum(pca.explained_variance_ratio_)
    
    print(f"PCA: {X.shape[1]} features → {n_components} components")
    print(f"Explained variance: {total_variance:.4f}")
    
    return X_pca, pca, n_components


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
        
        print(f"Variance threshold: {X.shape[1]} → {X_selected.shape[1]} features")
        
    elif method == 'l1':
        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import LogisticRegression
        
        selector = SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42),
            threshold=threshold
        )
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        print(f"L1 selection: {X.shape[1]} → {X_selected.shape[1]} features")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X_selected, selected_indices
