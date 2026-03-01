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
