"""
Streamlit application for TCGA-PRAD clinical stage classification.
VERSION-1: Centralized sklearn model
VERSION-2: Federated Learning with FedAvg
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import VERSION-1 modules
from preprocessing import load_clinical, create_target, preprocess_features
from model import train_model, predict_model
from evaluation import evaluate_model, plot_roc_curve, plot_confusion_matrix

# Import VERSION-2 modules
from logistic_numpy import predict_proba as numpy_predict_proba
from federated import partition_equal, fedavg_train, train_local_models
from experiments import centralized_train_numpy, save_fedavg_metrics, save_comparison_summary


# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create necessary directories
os.makedirs('reports', exist_ok=True)
os.makedirs('reports/version2', exist_ok=True)
os.makedirs('data', exist_ok=True)


def plot_fedavg_convergence(round_aucs, round_losses):
    """Plot FedAvg convergence curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # AUC curve
    rounds = list(range(1, len(round_aucs) + 1))
    ax1.plot(rounds, round_aucs, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Test AUC', fontsize=12)
    ax1.set_title('FedAvg: AUC vs Communication Rounds', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_ylim([min(round_aucs) - 0.02, max(round_aucs) + 0.02])
    
    # Loss curve
    ax2.plot(rounds, round_losses, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('FedAvg: Loss vs Communication Rounds', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application."""
    
    st.title("🏥 TCGA-PRAD Clinical Stage Classification")
    
    # Version selector
    version = st.sidebar.radio(
        "Select Version:",
        ["VERSION-1: Centralized (sklearn)", "VERSION-2: Federated Learning (FedAvg)"],
        index=0
    )
    
    st.markdown(f"### {version}")
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
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.2, random_state=RANDOM_SEED, stratify=target
        )
        
        # Convert to numpy arrays for federated learning
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        st.info(f"📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
        
        # VERSION-1: Centralized sklearn
        if "VERSION-1" in version:
            st.markdown("---")
            st.subheader("🚀 Model Training (sklearn)")
            
            if st.button("Train Centralized Model (sklearn)", type="primary"):
                
                with st.spinner("Training logistic regression model..."):
                    try:
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
        
        # VERSION-2: Federated Learning
        else:
            st.markdown("---")
            st.subheader("🌐 Federated Learning Configuration")
            
            # Federated learning parameters
            col1, col2 = st.columns(2)
            
            with col1:
                num_hospitals = st.slider("Number of Hospitals", min_value=2, max_value=10, value=5, step=1)
                rounds = st.slider("Communication Rounds", min_value=10, max_value=100, value=50, step=10)
            
            with col2:
                local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=10, value=5, step=1)
                lr = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
            
            st.markdown("---")
            
            # Buttons for different experiments
            col1, col2, col3 = st.columns(3)
            
            with col1:
                run_centralized = st.button("🖥️ Run Centralized (NumPy)", type="secondary")
            
            with col2:
                run_fedavg = st.button("🌐 Run FedAvg", type="primary")
            
            with col3:
                run_local = st.button("🏥 Run Local Models", type="secondary")
            
            # Run Centralized NumPy
            if run_centralized:
                with st.spinner("Training centralized model with NumPy..."):
                    try:
                        cent_results = centralized_train_numpy(
                            X_train, y_train, X_test, y_test,
                            epochs=local_epochs * rounds,  # Total epochs
                            lr=lr,
                            random_seed=RANDOM_SEED
                        )
                        
                        st.session_state['cent_results'] = cent_results
                        
                        st.success(f"✅ Centralized training complete!")
                        st.metric("Centralized AUC (NumPy)", f"{cent_results['test_auc']:.4f}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Run FedAvg
            if run_fedavg:
                with st.spinner(f"Running FedAvg with {num_hospitals} hospitals..."):
                    try:
                        # Partition data
                        hospitals = partition_equal(X_train, y_train, num_hospitals, RANDOM_SEED)
                        
                        # Train FedAvg
                        fedavg_results = fedavg_train(
                            hospitals, X_test, y_test,
                            rounds=rounds,
                            epochs=local_epochs,
                            lr=lr,
                            random_seed=RANDOM_SEED
                        )
                        
                        st.session_state['fedavg_results'] = fedavg_results
                        
                        # Save metrics
                        save_fedavg_metrics(fedavg_results['round_metrics'])
                        
                        st.success(f"✅ FedAvg training complete!")
                        
                        # Display results
                        st.markdown("### 📊 FedAvg Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Final FedAvg AUC", f"{fedavg_results['round_aucs'][-1]:.4f}")
                        
                        with col2:
                            improvement = fedavg_results['round_aucs'][-1] - fedavg_results['round_aucs'][0]
                            st.metric("AUC Improvement", f"{improvement:.4f}")
                        
                        # Plot convergence
                        st.markdown("### 📈 Convergence Curves")
                        fig = plot_fedavg_convergence(fedavg_results['round_aucs'], fedavg_results['round_losses'])
                        st.pyplot(fig)
                        plt.savefig('reports/version2/fedavg_convergence.png', dpi=300, bbox_inches='tight')
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Run Local Models
            if run_local:
                with st.spinner(f"Training {num_hospitals} local models..."):
                    try:
                        # Partition data
                        hospitals = partition_equal(X_train, y_train, num_hospitals, RANDOM_SEED)
                        
                        # Train local models
                        local_aucs = train_local_models(
                            hospitals, X_test, y_test,
                            epochs=local_epochs * rounds,  # Total epochs
                            lr=lr,
                            random_seed=RANDOM_SEED
                        )
                        
                        st.session_state['local_aucs'] = local_aucs
                        
                        st.success(f"✅ Local models training complete!")
                        
                        # Display results
                        st.markdown("### 📊 Local Model Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Average Local AUC", f"{np.mean(local_aucs):.4f}")
                        
                        with col2:
                            st.metric("Std Dev", f"{np.std(local_aucs):.4f}")
                        
                        # Show individual hospital AUCs
                        st.markdown("#### Hospital-wise AUCs")
                        for i, auc in enumerate(local_aucs):
                            st.write(f"Hospital {i+1}: {auc:.4f}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Comparison summary
            if 'cent_results' in st.session_state and 'fedavg_results' in st.session_state and 'local_aucs' in st.session_state:
                st.markdown("---")
                st.markdown("### 📊 Comparison Summary")
                
                cent_auc = st.session_state['cent_results']['test_auc']
                fedavg_auc = st.session_state['fedavg_results']['round_aucs'][-1]
                local_auc = np.mean(st.session_state['local_aucs'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Centralized AUC", f"{cent_auc:.4f}")
                
                with col2:
                    delta = fedavg_auc - cent_auc
                    st.metric("FedAvg AUC", f"{fedavg_auc:.4f}", delta=f"{delta:.4f}")
                
                with col3:
                    delta = local_auc - cent_auc
                    st.metric("Avg Local AUC", f"{local_auc:.4f}", delta=f"{delta:.4f}")
                
                # Save comparison
                save_comparison_summary(
                    st.session_state['cent_results'],
                    st.session_state['fedavg_results'],
                    st.session_state['local_aucs'],
                    num_hospitals, rounds, local_epochs, lr
                )
                
                st.success("✅ Comparison summary saved to reports/version2/")
    
    else:
        st.info("👈 Please upload clinical dataset to begin")
        
        st.markdown("---")
        st.markdown("### 📖 About This Application")
        
        if "VERSION-1" in version:
            st.markdown("""
            **VERSION-1: Centralized Learning**
            
            This version uses traditional centralized machine learning with sklearn's LogisticRegression.
            
            - **Task**: Predict pathologic T stage (T3/T4 vs T1/T2)
            - **Model**: Logistic Regression with balanced class weights
            - **Evaluation**: AUC-ROC, Accuracy, Confusion Matrix
            """)
        else:
            st.markdown("""
            **VERSION-2: Federated Learning (FedAvg)**
            
            This version implements Federated Averaging (FedAvg) algorithm using manual NumPy-based logistic regression.
            
            **FedAvg Algorithm:**
            1. Initialize global weights w_global
            2. For each communication round:
               - Send w_global to all hospitals
               - Each hospital trains locally
               - Aggregate weights: w_global = Σ(n_k/n_total × w_k)
            3. Return final w_global
            
            **Why Manual Implementation?**
            - Federated learning requires weight aggregation
            - sklearn's .fit() doesn't expose weights directly
            - NumPy implementation gives full control over training
            
            **Experiments:**
            - **Centralized (NumPy)**: Train on all data (baseline)
            - **FedAvg**: Federated training across hospitals
            - **Local Models**: Each hospital trains independently
            
            **Expected Results:**
            - FedAvg AUC ≈ Centralized AUC (with enough rounds)
            - Local AUC < FedAvg AUC (benefits of collaboration)
            """)


if __name__ == "__main__":
    main()
