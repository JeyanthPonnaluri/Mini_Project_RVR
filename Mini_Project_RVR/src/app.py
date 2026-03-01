"""
Streamlit application for TCGA-PRAD clinical stage classification.
VERSION-1: Centralized sklearn model
VERSION-2: Federated Learning with FedAvg
VERSION-3: Sustainability & Free-Rider Analysis
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

# Import VERSION-3 modules
from sustainability import (
    run_learning_curve,
    run_free_rider_experiment,
    compare_partitions,
    plot_learning_curve,
    plot_free_rider_curve,
    plot_partition_comparison,
    save_sustainability_results,
    save_partition_comparison_results
)


# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create necessary directories
os.makedirs('reports', exist_ok=True)
os.makedirs('reports/version2', exist_ok=True)
os.makedirs('reports/version3', exist_ok=True)
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
        [
            "VERSION-1: Centralized (sklearn)",
            "VERSION-2: Federated Learning (FedAvg)",
            "VERSION-3: Sustainability Analysis"
        ],
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
        elif "VERSION-2" in version:
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
        
        # VERSION-3: Sustainability Analysis
        elif "VERSION-3" in version:
            st.markdown("---")
            st.subheader("🔬 Sustainability & Free-Rider Analysis")
            
            st.markdown("""
            Study how federated learning performance changes with:
            - **Number of hospitals** (scalability)
            - **Free-rider scenarios** (non-participating hospitals)
            """)
            
            # Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                max_hospitals = st.slider("Max Hospitals", min_value=2, max_value=15, value=10, step=1)
                trials = st.slider("Monte Carlo Trials", min_value=5, max_value=20, value=10, step=5)
                partition_type = st.selectbox("Partition Type", ["equal", "imbalanced"])
            
            with col2:
                rounds = st.slider("Communication Rounds", min_value=10, max_value=50, value=30, step=10)
                local_epochs = st.slider("Local Epochs", min_value=1, max_value=10, value=3, step=1)
                lr = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
            
            # Generate hospital counts
            hospital_counts = list(range(2, max_hospitals + 1, 2))  # [2, 4, 6, 8, ...]
            if max_hospitals not in hospital_counts:
                hospital_counts.append(max_hospitals)
            
            st.info(f"📊 Will test: {hospital_counts} hospitals")
            
            st.markdown("---")
            
            # Buttons
            col1, col2 = st.columns(2)
            
            with col1:
                run_learning = st.button("📈 Run Learning Curve", type="primary")
            
            with col2:
                run_freerider = st.button("🎭 Run Free-Rider Experiment", type="primary")
            
            # Run Learning Curve
            if run_learning:
                with st.spinner(f"Running learning curve experiment ({trials} trials per configuration)..."):
                    try:
                        lc_df = run_learning_curve(
                            X_train, y_train, X_test, y_test,
                            hospital_counts=hospital_counts,
                            rounds=rounds,
                            epochs=local_epochs,
                            lr=lr,
                            trials=trials,
                            partition_type=partition_type,
                            random_seed=RANDOM_SEED
                        )
                        
                        st.session_state['lc_df'] = lc_df
                        
                        st.success("✅ Learning curve experiment complete!")
                        
                        # Plot
                        st.markdown("### 📈 Learning Curve")
                        fig = plot_learning_curve(lc_df, save_path='reports/version3/learning_curve_plot.png')
                        st.pyplot(fig)
                        
                        # Summary table
                        st.markdown("### 📊 Summary Statistics")
                        summary = lc_df.groupby('K').agg({
                            'global_auc': ['mean', 'std', 'min', 'max'],
                            'avg_local_auc': ['mean', 'std', 'min', 'max']
                        }).round(4)
                        st.dataframe(summary)
                        
                        # Save results
                        lc_df.to_csv('reports/version3/learning_curve_results.csv', index=False)
                        st.success("Results saved to reports/version3/")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Run Free-Rider Experiment
            if run_freerider:
                with st.spinner(f"Running free-rider experiment ({trials} trials per configuration)..."):
                    try:
                        fr_df = run_free_rider_experiment(
                            X_train, y_train, X_test, y_test,
                            hospital_counts=hospital_counts,
                            rounds=rounds,
                            epochs=local_epochs,
                            lr=lr,
                            trials=trials,
                            random_seed=RANDOM_SEED
                        )
                        
                        st.session_state['fr_df'] = fr_df
                        
                        st.success("✅ Free-rider experiment complete!")
                        
                        # Plot
                        st.markdown("### 🎭 Free-Rider Analysis")
                        fig = plot_free_rider_curve(fr_df, save_path='reports/version3/free_rider_plot.png')
                        st.pyplot(fig)
                        
                        # Summary table
                        st.markdown("### 📊 Summary Statistics")
                        summary = fr_df.groupby('K').agg({
                            'free_rider_auc': ['mean', 'std', 'min', 'max'],
                            'global_auc': ['mean', 'std', 'min', 'max']
                        }).round(4)
                        st.dataframe(summary)
                        
                        # Insights
                        st.markdown("### 💡 Key Insights")
                        
                        avg_fr_auc = fr_df.groupby('K')['free_rider_auc'].mean()
                        avg_global_auc = fr_df.groupby('K')['global_auc'].mean()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Avg Free-Rider AUC", f"{avg_fr_auc.mean():.4f}")
                            st.caption("Average across all K values")
                        
                        with col2:
                            gap = avg_global_auc.mean() - avg_fr_auc.mean()
                            st.metric("Performance Gap", f"{gap:.4f}")
                            st.caption("Global AUC - Free-Rider AUC")
                        
                        # Save results
                        fr_df.to_csv('reports/version3/free_rider_results.csv', index=False)
                        st.success("Results saved to reports/version3/")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Partition Comparison Study
            st.markdown("---")
            st.markdown("### ⚖️ Partition Comparison Study")
            st.markdown("Compare Equal vs Imbalanced data distribution across hospitals")
            
            col1, col2 = st.columns(2)
            
            with col1:
                comp_max_hospitals = st.slider("Max Hospitals (Comparison)", min_value=2, max_value=10, value=6, step=2, key="comp_max_hosp")
                comp_trials = st.slider("Trials (Comparison)", min_value=5, max_value=15, value=10, step=5, key="comp_trials")
            
            with col2:
                comp_rounds = st.slider("Rounds (Comparison)", min_value=10, max_value=40, value=20, step=10, key="comp_rounds")
                comp_epochs = st.slider("Epochs (Comparison)", min_value=1, max_value=5, value=3, step=1, key="comp_epochs")
                comp_lr = st.number_input("LR (Comparison)", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f", key="comp_lr")
            
            # Generate hospital counts for comparison
            comp_hospital_counts = list(range(2, comp_max_hospitals + 1, 2))
            if comp_max_hospitals not in comp_hospital_counts:
                comp_hospital_counts.append(comp_max_hospitals)
            
            st.info(f"📊 Will compare: {comp_hospital_counts} hospitals")
            
            if st.button("⚖️ Run Partition Comparison", type="primary", key="run_comparison"):
                with st.spinner(f"Running partition comparison ({comp_trials} trials per configuration)..."):
                    try:
                        comp_df = compare_partitions(
                            X_train, y_train, X_test, y_test,
                            hospital_counts=comp_hospital_counts,
                            rounds=comp_rounds,
                            epochs=comp_epochs,
                            lr=comp_lr,
                            trials=comp_trials,
                            random_seed=RANDOM_SEED
                        )
                        
                        st.session_state['comp_df'] = comp_df
                        
                        st.success("✅ Partition comparison complete!")
                        
                        # Plot comparison
                        st.markdown("### 📊 Partition Comparison Results")
                        fig = plot_partition_comparison(comp_df, save_path='reports/version3_partition_comparison/comparison_plot.png')
                        st.pyplot(fig)
                        
                        # Summary table
                        st.markdown("### 📋 Statistical Summary")
                        
                        # Format display dataframe
                        display_df = comp_df.copy()
                        display_df['Equal Global AUC'] = display_df.apply(
                            lambda row: f"{row['equal_global_auc_mean']:.4f} ± {row['equal_global_auc_std']:.4f}", axis=1
                        )
                        display_df['Imbalanced Global AUC'] = display_df.apply(
                            lambda row: f"{row['imbalanced_global_auc_mean']:.4f} ± {row['imbalanced_global_auc_std']:.4f}", axis=1
                        )
                        display_df['Global p-value'] = display_df['global_auc_pvalue'].apply(lambda x: f"{x:.4f}")
                        display_df['Significant?'] = display_df['global_auc_pvalue'].apply(lambda x: "✓" if x < 0.05 else "✗")
                        
                        st.dataframe(display_df[['K', 'Equal Global AUC', 'Imbalanced Global AUC', 'Global p-value', 'Significant?']])
                        
                        # Interpretation
                        st.markdown("### 💡 Research Interpretation")
                        
                        avg_diff = (comp_df['equal_global_auc_mean'] - comp_df['imbalanced_global_auc_mean']).mean()
                        significant_count = (comp_df['global_auc_pvalue'] < 0.05).sum()
                        
                        st.markdown(f"""
                        **Key Findings:**
                        
                        1. **Performance Gap**: Average difference = {avg_diff:.4f}
                           - {'Equal partition performs BETTER' if avg_diff > 0 else 'Imbalanced partition performs BETTER'}
                           - {significant_count}/{len(comp_df)} configurations show statistically significant differences (p < 0.05)
                        
                        2. **Data Heterogeneity Impact**:
                           - Imbalanced data distribution affects federated convergence
                           - Larger hospitals dominate the global model in imbalanced scenarios
                           - Smaller hospitals may underfit due to limited local data
                        
                        3. **Implications for Real-World Deployment**:
                           - Real hospitals have naturally imbalanced data sizes
                           - FedAvg may not be optimal for heterogeneous settings
                           - Consider: FedProx, FedNova, or personalized federated learning
                        
                        4. **Free-Rider Behavior**:
                           - Free-riders benefit differently under equal vs imbalanced partitions
                           - Data heterogeneity affects incentive structures
                        
                        **Next Steps:**
                        - Implement FedProx to handle data heterogeneity
                        - Study personalized federated learning approaches
                        - Analyze convergence rates under different distributions
                        """)
                        
                        # Save results
                        save_partition_comparison_results(comp_df)
                        st.success("Results saved to reports/version3_partition_comparison/")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Combined analysis
            if 'lc_df' in st.session_state and 'fr_df' in st.session_state:
                st.markdown("---")
                st.markdown("### 🔬 Combined Analysis")
                
                st.markdown("""
                **Key Findings:**
                - **Scalability**: How does performance scale with more hospitals?
                - **Free-Riding**: Can non-participating hospitals benefit from the global model?
                - **Sustainability**: Is federated learning sustainable at scale?
                """)
                
                # Save combined results
                save_sustainability_results(
                    st.session_state['lc_df'],
                    st.session_state['fr_df']
                )
    
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
        elif "VERSION-2" in version:
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
        else:  # VERSION-3
            st.markdown("""
            **VERSION-3: Sustainability & Free-Rider Analysis**
            
            This version studies the sustainability and scalability of federated learning.
            
            **Research Questions:**
            1. **Scalability**: How does performance change as we add more hospitals?
            2. **Free-Riding**: Can non-participating hospitals benefit from the global model?
            3. **Sustainability**: Is federated learning sustainable at scale?
            
            **Learning Curve Experiment:**
            - Test different numbers of hospitals (K = 2, 4, 6, ..., max)
            - Run multiple Monte Carlo trials for statistical significance
            - Compare FedAvg vs Local models
            - Analyze: Does more data (more hospitals) always help?
            
            **Free-Rider Experiment:**
            - Simulate hospitals that don't participate in training
            - Train FedAvg on K-1 hospitals
            - Evaluate global model on excluded hospital's data
            - Question: Do free-riders benefit from collaboration without contributing?
            
            **Expected Insights:**
            - **Learning Curve**: Performance improves with more hospitals, but with diminishing returns
            - **Free-Rider**: Non-participants still benefit, but less than active participants
            - **Sustainability**: Federated learning remains effective even at scale
            
            **Why This Matters:**
            - **Real-world**: Not all hospitals may participate equally
            - **Incentives**: Understanding free-rider benefits helps design participation incentives
            - **Scalability**: Knowing performance limits helps plan federated deployments
            """)


if __name__ == "__main__":
    main()
