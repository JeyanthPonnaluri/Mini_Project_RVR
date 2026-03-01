"""
Streamlit application for TCGA-PRAD clinical stage classification.
Predicts advanced stage (T3/T4) vs early stage (T1/T2) from clinical features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Import custom modules
from preprocessing import load_clinical, create_target, preprocess_features
from model import train_model, predict_model
from evaluation import evaluate_model, plot_roc_curve, plot_confusion_matrix


# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create necessary directories
os.makedirs('reports', exist_ok=True)
os.makedirs('data', exist_ok=True)


def main():
    """Main Streamlit application."""
    
    st.title("🏥 TCGA-PRAD Clinical Stage Classification")
    st.markdown("### VERSION-1: Pathologic T Stage Prediction (T3/T4 vs T1/T2)")
    st.markdown("---")
    
    st.sidebar.header("📁 Upload Clinical Data")
    st.sidebar.markdown("Upload TCGA-PRAD clinical TSV file:")
    
    # File uploader
    clinical_file = st.sidebar.file_uploader("Clinical Data (TSV)", type=['tsv', 'csv'])
    
    if clinical_file:
        
        st.success("✅ Clinical dataset uploaded successfully!")
        
        # Load dataset
        with st.spinner("Loading clinical data..."):
            try:
                # Save uploaded file temporarily
                clinical_path = f"data/temp_clinical.tsv"
                
                with open(clinical_path, 'wb') as f:
                    f.write(clinical_file.getbuffer())
                
                clinical_df = load_clinical(clinical_path)
                
                st.info(f"📊 Loaded: {clinical_df.shape[0]} patients, {clinical_df.shape[1]} features")
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                return
        
        # Create target variable
        with st.spinner("Creating target variable..."):
            try:
                df_filtered, target = create_target(clinical_df)
                
                st.success(f"✅ Target created successfully! {len(df_filtered)} patients with valid stage")
                
                # Display class distribution
                st.markdown("### 🎯 Target Variable: Pathologic T Stage")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Early Stage (T1/T2)", f"{(target == 0).sum()} patients")
                
                with col2:
                    st.metric("Advanced Stage (T3/T4)", f"{(target == 1).sum()} patients")
                
                # Calculate class balance
                class_ratio = (target == 1).sum() / len(target) * 100
                st.info(f"📈 Class balance: {class_ratio:.1f}% advanced stage")
                
                # Display sample
                with st.expander("👀 View Sample Data"):
                    st.dataframe(df_filtered.head())
                
            except ValueError as e:
                st.error(f"Error creating target: {str(e)}")
                st.info("Please ensure your clinical file contains 'ajcc_pathologic_t.diagnoses' column")
                return
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return
        
        # Preprocess features
        st.markdown("---")
        st.subheader("🔧 Feature Preprocessing")
        
        with st.spinner("Preprocessing features..."):
            try:
                X, feature_names, preprocessor = preprocess_features(df_filtered)
                
                st.success(f"✅ Preprocessing complete!")
                st.info(f"📊 Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
                
            except Exception as e:
                st.error(f"Error preprocessing features: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
        
        # Train model button
        st.markdown("---")
        st.subheader("🚀 Model Training")
        
        if st.button("Train Model", type="primary"):
            
            with st.spinner("Training logistic regression model..."):
                try:
                    # Train-test split (stratified 80/20)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, target, test_size=0.2, random_state=RANDOM_SEED, stratify=target
                    )
                    
                    st.info(f"📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
                    
                    # Train model
                    model = train_model(X_train, y_train, RANDOM_SEED)
                    
                    # Evaluate model
                    results = evaluate_model(model, X_test, y_test)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Model Performance")
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("🎯 AUC-ROC Score", f"{results['auc']:.4f}")
                    
                    with col2:
                        st.metric("✅ Accuracy", f"{results['accuracy']:.4f}")
                    
                    # Confusion Matrix
                    st.markdown("### 📈 Confusion Matrix")
                    fig_cm = plot_confusion_matrix(results['confusion_matrix'], save_path='reports/confusion_matrix.png')
                    st.pyplot(fig_cm)
                    
                    # ROC Curve
                    st.markdown("### 📈 ROC Curve")
                    fig_roc = plot_roc_curve(y_test, results['y_pred_proba'], save_path='reports/roc_curve.png')
                    st.pyplot(fig_roc)
                    
                    st.success("✅ Model training complete! Results saved to reports/")
                    
                    # Additional insights
                    st.markdown("---")
                    st.markdown("### 💡 Model Insights")
                    
                    tn, fp, fn, tp = results['confusion_matrix'].ravel()
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Sensitivity (Recall)", f"{sensitivity:.4f}")
                        st.caption("Ability to detect advanced stage (T3/T4)")
                    
                    with col2:
                        st.metric("Specificity", f"{specificity:.4f}")
                        st.caption("Ability to identify early stage (T1/T2)")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        st.info("👈 Please upload clinical dataset to begin")
        
        st.markdown("---")
        st.markdown("### 📖 About This Application")
        st.markdown("""
        This application predicts **pathologic T stage** for prostate cancer patients using clinical features.
        
        **Target Variable:**
        - **Advanced Stage (1)**: T3 or T4 tumors (locally advanced)
        - **Early Stage (0)**: T1 or T2 tumors (organ-confined)
        
        **Why This Task?**
        - Clinically meaningful: T stage is a key prognostic factor
        - Better class balance than survival prediction
        - Useful for treatment planning and risk stratification
        
        **Model:**
        - Logistic Regression with balanced class weights
        - Stratified 80/20 train-test split
        - Features: Clinical variables (encoded and scaled)
        
        **Evaluation Metrics:**
        - AUC-ROC: Overall discrimination ability
        - Accuracy: Correct classification rate
        - Confusion Matrix: Detailed performance breakdown
        - Sensitivity: Detection of advanced stage
        - Specificity: Identification of early stage
        """)


if __name__ == "__main__":
    main()
