"""
Streamlit application for TCGA-PRAD clinical stage classification.
VERSION-1: Centralized sklearn model
VERSION-2: Federated Learning with FedAvg
VERSION-3: Sustainability & Free-Rider Analysis
VERSION-4: FedProx & Non-IID Study
VERSION-5: Research Lab - Advanced Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to Python path for module imports
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Fix matplotlib backend for cloud deployment (headless environment)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Import VERSION-1 modules
from preprocessing import load_clinical, create_target, preprocess_features
from model import train_model, predict_model
from evaluation import evaluate_model, plot_roc_curve, plot_confusion_matrix

# Import VERSION-2 modules
from logistic_numpy import predict_proba as numpy_predict_proba
from federated import (
    partition_equal, 
    fedavg_train, 
    train_local_models,
    partition_dirichlet,
    partition_imbalanced,
    generate_imbalanced_distribution
)
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

# Import VERSION-4 modules
from fedprox_experiments import (
    run_fedavg_vs_fedprox_experiment,
    plot_convergence_curves,
    plot_stability_comparison,
    save_fedprox_results
)

# Import VERSION-5 modules
from contribution import measure_hospital_contribution, plot_contribution_analysis
from experiment_manager import ExperimentManager, set_global_seed

# Import UI components
from ui_components import (
    render_header,
    render_card,
    render_metrics_row,
    render_section_header,
    render_divider,
    render_info_box,
    render_experiment_status,
    render_comparison_table,
    render_footer,
    render_sidebar_section,
    render_key_findings,
    render_version_selector,
    apply_custom_css
)


# Set page configuration
st.set_page_config(
    page_title="Federated Learning Research Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render professional header
    render_header()
    
    # Version selector in main area
    version = render_version_selector()
    
    render_divider()
    
    # Sidebar configuration
    render_sidebar_section("📁 Data Upload", "")
    st.sidebar.markdown("Upload TCGA-PRAD clinical TSV file:")
    
    # File uploader
    clinical_file = st.sidebar.file_uploader("Clinical Data (TSV)", type=['tsv', 'csv'])
    
    if clinical_file:
        
        # Current version badge
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1rem 1.5rem; border-radius: 8px; 
                        margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 1.4rem; font-weight: 600; margin-bottom: 0.25rem;">
                    {version}
                </div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    Active Experiment Configuration
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        render_experiment_status('running', 'Loading and preprocessing data...')
        
        # Load dataset
        with st.spinner("Loading clinical data..."):
            try:
                # Save uploaded file temporarily
                clinical_path = f"data/temp_clinical.tsv"
                
                with open(clinical_path, 'wb') as f:
                    f.write(clinical_file.getbuffer())
                
                clinical_df = load_clinical(clinical_path)
                
                render_info_box(f"📊 Loaded: {clinical_df.shape[0]} patients, {clinical_df.shape[1]} features", 'success')
                
            except Exception as e:
                render_info_box(f"Error loading dataset: {str(e)}", 'error')
                return
        
        # Create target variable
        with st.spinner("Creating target variable..."):
            try:
                df_filtered, target = create_target(clinical_df)
                
                render_divider()
                render_section_header("🎯 Target Variable: Pathologic T Stage", 
                                     "Binary classification: Early Stage (T1/T2) vs Advanced Stage (T3/T4)")
                
                # Display class distribution in metrics row
                metrics = [
                    {
                        'label': 'Early Stage (T1/T2)',
                        'value': f"{(target == 0).sum()}",
                        'help': 'Number of patients with early stage cancer'
                    },
                    {
                        'label': 'Advanced Stage (T3/T4)',
                        'value': f"{(target == 1).sum()}",
                        'help': 'Number of patients with advanced stage cancer'
                    },
                    {
                        'label': 'Class Balance',
                        'value': f"{(target == 1).sum() / len(target) * 100:.1f}%",
                        'help': 'Percentage of advanced stage patients'
                    }
                ]
                render_metrics_row(metrics, columns=3)
                
            except ValueError as e:
                render_info_box(f"Error creating target: {str(e)}\n\nPlease ensure your clinical file contains 'ajcc_pathologic_t.diagnoses' column", 'error')
                return
            except Exception as e:
                render_info_box(f"Error: {str(e)}", 'error')
                return
        
        # Preprocess features
        render_divider()
        render_section_header("🔧 Feature Preprocessing", "Handling missing values, encoding categorical variables, and scaling features")
        
        with st.spinner("Preprocessing features..."):
            try:
                X, feature_names, preprocessor = preprocess_features(df_filtered)
                
                render_info_box(f"✅ Preprocessing complete! Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features", 'success')
                
            except Exception as e:
                render_info_box(f"Error preprocessing features: {str(e)}", 'error')
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
        
        metrics = [
            {'label': 'Training Samples', 'value': f"{X_train.shape[0]}"},
            {'label': 'Test Samples', 'value': f"{X_test.shape[0]}"},
            {'label': 'Train/Test Split', 'value': "80/20"}
        ]
        render_metrics_row(metrics, columns=3)
        
        render_divider()
        
        # VERSION-1: Centralized sklearn
        if "VERSION-1" in version:
            render_section_header("🚀 Model Training", "Centralized logistic regression using scikit-learn")
            
            if st.button("Train Centralized Model (sklearn)", type="primary", use_container_width=True):
                
                render_experiment_status('running', 'Training logistic regression model...')
                
                with st.spinner("Training logistic regression model..."):
                    try:
                        # Train model
                        model = train_model(X_train, y_train, RANDOM_SEED)
                        
                        # Evaluate model
                        results = evaluate_model(model, X_test, y_test)
                        
                        render_experiment_status('complete', 'Model training completed successfully!')
                        
                        # Display results
                        render_divider()
                        render_section_header("📊 Model Performance", "Evaluation metrics on test set")
                        
                        # Metrics
                        metrics = [
                            {
                                'label': '🎯 AUC-ROC Score',
                                'value': f"{results['auc']:.4f}",
                                'help': 'Area Under the ROC Curve - measures discrimination ability'
                            },
                            {
                                'label': '✅ Accuracy',
                                'value': f"{results['accuracy']:.4f}",
                                'help': 'Overall classification accuracy'
                            }
                        ]
                        render_metrics_row(metrics, columns=2)
                        
                        # Confusion Matrix and ROC Curve
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Confusion Matrix")
                            fig_cm = plot_confusion_matrix(results['confusion_matrix'], save_path='reports/confusion_matrix.png')
                            st.pyplot(fig_cm)
                        
                        with col2:
                            st.markdown("#### 📈 ROC Curve")
                            fig_roc = plot_roc_curve(y_test, results['y_pred_proba'], save_path='reports/roc_curve.png')
                            st.pyplot(fig_roc)
                        
                        # Additional insights
                        render_divider()
                        render_section_header("💡 Model Insights", "Detailed performance metrics")
                        
                        tn, fp, fn, tp = results['confusion_matrix'].ravel()
                        
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        metrics = [
                            {
                                'label': 'Sensitivity (Recall)',
                                'value': f"{sensitivity:.4f}",
                                'help': 'Ability to detect advanced stage (T3/T4)'
                            },
                            {
                                'label': 'Specificity',
                                'value': f"{specificity:.4f}",
                                'help': 'Ability to identify early stage (T1/T2)'
                            }
                        ]
                        render_metrics_row(metrics, columns=2)
                        
                        render_info_box("✅ Model training complete! Results saved to reports/", 'success')
                        
                    except Exception as e:
                        render_experiment_status('error', f'Training failed: {str(e)}')
                        import traceback
                        st.code(traceback.format_exc())
        
        # VERSION-2: Federated Learning
        elif "VERSION-2" in version:
            render_section_header("🌐 Federated Learning Configuration", "Distributed training across multiple hospitals using FedAvg")
            
            # Federated learning parameters
            with st.expander("⚙️ Federated Learning Parameters", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    num_hospitals = st.slider("Number of Hospitals", min_value=2, max_value=10, value=5, step=1)
                    rounds = st.slider("Communication Rounds", min_value=10, max_value=100, value=50, step=10)
                
                with col2:
                    local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=10, value=5, step=1)
                    lr = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
            
            render_divider()
            
            # Buttons for different experiments
            col1, col2, col3 = st.columns(3)
            
            with col1:
                run_centralized = st.button("🖥️ Run Centralized (NumPy)", type="secondary", use_container_width=True)
            
            with col2:
                run_fedavg = st.button("🌐 Run FedAvg", type="primary", use_container_width=True)
            
            with col3:
                run_local = st.button("🏥 Run Local Models", type="secondary", use_container_width=True)
            
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
            render_section_header("🔬 Sustainability & Free-Rider Analysis", 
                                 "Study scalability and free-rider behavior in federated learning")
            
            st.markdown("""
            Study how federated learning performance changes with:
            - **Number of hospitals** (scalability)
            - **Free-rider scenarios** (non-participating hospitals)
            """)
            
            # Configuration
            with st.expander("⚙️ Experiment Configuration", expanded=True):
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
            
            render_info_box(f"📊 Will test: {hospital_counts} hospitals", 'info')
            
            render_divider()
            
            # Buttons
            col1, col2 = st.columns(2)
            
            with col1:
                run_learning = st.button("📈 Run Learning Curve", type="primary", use_container_width=True)
            
            with col2:
                run_freerider = st.button("🎭 Run Free-Rider Experiment", type="primary", use_container_width=True)
            
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
        
        # VERSION-4: FedProx & Non-IID Study
        elif "VERSION-4" in version:
            render_section_header("🔬 FedProx & Non-IID Heterogeneity Study", 
                                 "Compare FedAvg vs FedProx under data heterogeneity")
            
            st.markdown("""
            Study how **FedProx** handles data heterogeneity compared to FedAvg:
            - **Proximal regularization** prevents client drift
            - **Dirichlet non-IID** simulates realistic heterogeneity
            - **Convergence analysis** shows stability improvements
            """)
            
            # Configuration
            with st.expander("⚙️ Experiment Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    num_hospitals_v4 = st.slider("Number of Hospitals", min_value=3, max_value=10, value=5, step=1, key="v4_hospitals")
                    partition_type_v4 = st.selectbox("Partition Strategy", ["equal", "imbalanced", "dirichlet"], key="v4_partition")
                    
                    if partition_type_v4 == "dirichlet":
                        alpha_v4 = st.slider("Dirichlet Alpha (α)", min_value=0.1, max_value=10.0, value=0.5, step=0.1, key="v4_alpha")
                        st.caption(f"α={alpha_v4:.1f}: {'Strong non-IID' if alpha_v4 < 1 else 'Moderate' if alpha_v4 < 5 else 'Nearly IID'}")
                    else:
                        alpha_v4 = None
                
                with col2:
                    rounds_v4 = st.slider("Communication Rounds", min_value=20, max_value=100, value=50, step=10, key="v4_rounds")
                    epochs_v4 = st.slider("Local Epochs", min_value=1, max_value=10, value=5, step=1, key="v4_epochs")
                    lr_v4 = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f", key="v4_lr")
            
            # Mu values for FedProx
            with st.expander("🔧 FedProx Configuration", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mu1 = st.number_input("μ₁ (small)", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f", key="v4_mu1")
                with col2:
                    mu2 = st.number_input("μ₂ (medium)", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.2f", key="v4_mu2")
                with col3:
                    mu3 = st.number_input("μ₃ (large)", min_value=0.001, max_value=1.0, value=0.5, step=0.1, format="%.1f", key="v4_mu3")
                
                mu_values_v4 = [mu1, mu2, mu3]
            
            render_info_box(f"📊 Will compare: FedAvg vs FedProx with μ = {mu_values_v4}", 'info')
            
            render_divider()
            
            # Run comparison button
            if st.button("🚀 Run FedAvg vs FedProx Comparison", type="primary", key="run_v4_comparison", use_container_width=True):
                render_experiment_status('running', f'Running comparison with {num_hospitals_v4} hospitals...')
                
                with st.spinner(f"Running comparison experiment..."):
                    try:
                        # Run experiment
                        results_df = run_fedavg_vs_fedprox_experiment(
                            X_train, y_train, X_test, y_test,
                            num_hospitals=num_hospitals_v4,
                            partition_type=partition_type_v4,
                            alpha=alpha_v4,
                            mu_values=mu_values_v4,
                            rounds=rounds_v4,
                            epochs=epochs_v4,
                            lr=lr_v4,
                            random_seed=RANDOM_SEED
                        )
                        
                        st.session_state['v4_results'] = results_df
                        
                        render_experiment_status('complete', 'Comparison experiment completed successfully!')
                        
                        # Display results
                        render_divider()
                        render_section_header("📊 Performance Comparison", "FedAvg vs FedProx results")
                        
                        # Summary table
                        summary_df = results_df[['algorithm', 'mu', 'final_auc', 'convergence_std', 'avg_weight_drift']].copy()
                        summary_df['mu'] = summary_df['mu'].apply(lambda x: f"{x:.3f}")
                        summary_df['final_auc'] = summary_df['final_auc'].apply(lambda x: f"{x:.4f}")
                        summary_df['convergence_std'] = summary_df['convergence_std'].apply(lambda x: f"{x:.4f}")
                        summary_df['avg_weight_drift'] = summary_df['avg_weight_drift'].apply(lambda x: f"{x:.4f}")
                        
                        render_comparison_table(summary_df, highlight_best=True)
                        
                        # Convergence curves
                        st.markdown("### 📈 Convergence Analysis")
                        fig_conv = plot_convergence_curves(results_df, save_path='reports/version4_fedprox/convergence_plot.png')
                        st.pyplot(fig_conv)
                        
                        # Stability comparison
                        st.markdown("### 📊 Stability Comparison")
                        fig_stab = plot_stability_comparison(results_df, save_path='reports/version4_fedprox/stability_plot.png')
                        st.pyplot(fig_stab)
                        
                        # Key insights
                        st.markdown("### 💡 Key Insights")
                        
                        fedavg_row = results_df[results_df['algorithm'] == 'FedAvg'].iloc[0]
                        fedprox_rows = results_df[results_df['algorithm'] == 'FedProx']
                        best_fedprox = fedprox_rows.loc[fedprox_rows['final_auc'].idxmax()]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("FedAvg Final AUC", f"{fedavg_row['final_auc']:.4f}")
                        
                        with col2:
                            improvement = best_fedprox['final_auc'] - fedavg_row['final_auc']
                            st.metric("Best FedProx AUC", f"{best_fedprox['final_auc']:.4f}", 
                                     delta=f"{improvement:.4f}", delta_color="normal")
                            st.caption(f"μ = {best_fedprox['mu']:.3f}")
                        
                        with col3:
                            stability_improvement = fedavg_row['convergence_std'] - best_fedprox['convergence_std']
                            st.metric("Stability Improvement", f"{stability_improvement:.4f}")
                            st.caption("Lower std = more stable")
                        
                        # Interpretation
                        st.markdown("""
                        **Research Interpretation:**
                        
                        1. **Performance**: FedProx with optimal μ typically improves AUC under non-IID settings
                        2. **Stability**: Proximal term reduces oscillations in convergence
                        3. **Weight Drift**: FedProx controls how far local models deviate from global
                        4. **Optimal μ**: Balance between local adaptation and global consistency
                        
                        **When to use FedProx:**
                        - Strong data heterogeneity (Dirichlet α < 1)
                        - Unstable FedAvg convergence
                        - Need for convergence guarantees
                        """)
                        
                        # Save results
                        save_fedprox_results(results_df)
                        st.success("Results saved to reports/version4_fedprox/")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # VERSION-5: Research Lab
        elif "VERSION-5" in version:
            render_section_header("🔬 Research Lab - Advanced Analysis", 
                                 "Publication-quality research tools for federated learning")
            
            st.markdown("""
            **VERSION-5** provides publication-quality research tools:
            - 🏥 **Hospital Contribution Analysis**: Measure each hospital's impact
            - 📊 **Experiment Management**: Reproducible research with automatic logging
            - 🧬 **Multi-Modal Support**: Clinical + Protein data (backend ready)
            """)
            
            # Configuration
            render_divider()
            render_section_header("⚙️ Configuration", "Set up your federated learning experiment")
            
            with st.expander("🌐 Federated Learning Parameters", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    num_hospitals_v5 = st.slider("Number of Hospitals", min_value=3, max_value=8, value=5, step=1, key="v5_hospitals")
                    partition_type_v5 = st.selectbox("Partition Strategy", ["equal", "imbalanced", "dirichlet"], key="v5_partition")
                    
                    if partition_type_v5 == "dirichlet":
                        alpha_v5 = st.slider("Dirichlet Alpha (α)", min_value=0.1, max_value=10.0, value=0.5, step=0.1, key="v5_alpha")
                    else:
                        alpha_v5 = None
                
                with col2:
                    rounds_v5 = st.slider("Communication Rounds", min_value=20, max_value=50, value=30, step=10, key="v5_rounds")
                    epochs_v5 = st.slider("Local Epochs", min_value=3, max_value=10, value=5, step=1, key="v5_epochs")
                    lr_v5 = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f", key="v5_lr")
            
            # Algorithm selection
            with st.expander("🤖 Algorithm Selection", expanded=True):
                algorithm_v5 = st.selectbox("Algorithm", ["FedAvg", "FedProx"], key="v5_algorithm")
                if algorithm_v5 == "FedProx":
                    mu_v5 = st.slider("Proximal Coefficient (μ)", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="v5_mu")
                else:
                    mu_v5 = 0.0
            
            render_divider()
            
            # Hospital Contribution Analysis
            render_section_header("🏥 Hospital Contribution Analysis", 
                                 "Measure each hospital's impact using leave-one-out analysis")
            
            if st.button("🔍 Run Contribution Analysis", type="primary", key="run_contribution", use_container_width=True):
                render_experiment_status('running', f'Analyzing {num_hospitals_v5} hospitals...')
                
                with st.spinner(f"Analyzing {num_hospitals_v5} hospitals..."):
                    try:
                        # Create experiment
                        exp = ExperimentManager()
                        exp_id = exp.create_experiment({
                            'version': 'VERSION-5',
                            'analysis': 'contribution',
                            'num_hospitals': num_hospitals_v5,
                            'partition_type': partition_type_v5,
                            'alpha': alpha_v5,
                            'algorithm': algorithm_v5.lower(),
                            'mu': mu_v5,
                            'rounds': rounds_v5,
                            'epochs': epochs_v5,
                            'lr': lr_v5,
                            'random_seed': RANDOM_SEED
                        })
                        
                        render_info_box(f"📝 Experiment ID: {exp_id}", 'info')
                        
                        # Partition data
                        if partition_type_v5 == 'equal':
                            hospitals = partition_equal(X_train, y_train, num_hospitals_v5, RANDOM_SEED)
                        elif partition_type_v5 == 'imbalanced':
                            distribution = generate_imbalanced_distribution(num_hospitals_v5, RANDOM_SEED)
                            hospitals = partition_imbalanced(X_train, y_train, distribution, RANDOM_SEED)
                        else:  # dirichlet
                            hospitals = partition_dirichlet(X_train, y_train, num_hospitals_v5, alpha_v5, RANDOM_SEED)
                        
                        # Run contribution analysis
                        contribution_df = measure_hospital_contribution(
                            hospitals, X_test, y_test,
                            rounds=rounds_v5,
                            epochs=epochs_v5,
                            lr=lr_v5,
                            algorithm=algorithm_v5.lower(),
                            mu=mu_v5,
                            random_seed=RANDOM_SEED
                        )
                        
                        st.session_state['v5_contribution'] = contribution_df
                        
                        render_experiment_status('complete', 'Contribution analysis completed successfully!')
                        
                        # Display results
                        render_divider()
                        render_section_header("📊 Contribution Results", "Hospital-wise impact on federated learning performance")
                        
                        # Summary metrics
                        metrics = [
                            {
                                'label': 'Baseline AUC',
                                'value': f"{contribution_df['baseline_auc'].iloc[0]:.4f}",
                                'help': 'Performance with all hospitals'
                            },
                            {
                                'label': 'Max Contribution',
                                'value': f"{contribution_df['contribution'].max():.4f}",
                                'help': f"Hospital {int(contribution_df.loc[contribution_df['contribution'].idxmax(), 'hospital_id'])}"
                            },
                            {
                                'label': 'Mean Contribution',
                                'value': f"{contribution_df['contribution'].mean():.4f}",
                                'help': 'Average impact across all hospitals'
                            }
                        ]
                        render_metrics_row(metrics, columns=3)
                        
                        # Contribution table
                        st.markdown("#### Detailed Contributions")
                        display_df = contribution_df[['hospital_id', 'num_samples', 'contribution', 'contribution_pct']].copy()
                        display_df['hospital_id'] = display_df['hospital_id'].astype(int)
                        display_df['contribution'] = display_df['contribution'].apply(lambda x: f"{x:.4f}")
                        display_df['contribution_pct'] = display_df['contribution_pct'].apply(lambda x: f"{x:.2f}%")
                        display_df.columns = ['Hospital ID', 'Samples', 'Contribution (ΔAUC)', 'Contribution %']
                        
                        render_comparison_table(display_df, highlight_best=False)
                        
                        # Visualizations
                        render_divider()
                        render_section_header("📈 Contribution Visualizations", "Visual analysis of hospital contributions")
                        fig = plot_contribution_analysis(
                            contribution_df,
                            save_path=exp.get_plot_path('contribution_analysis.png')
                        )
                        st.pyplot(fig)
                        
                        # Insights
                        render_divider()
                        render_section_header("💡 Key Insights", "Research interpretation and implications")
                        
                        # Correlation analysis
                        correlation = contribution_df[['num_samples', 'contribution']].corr().iloc[0, 1]
                        max_contrib = contribution_df['contribution'].max()
                        max_hospital = contribution_df.loc[contribution_df['contribution'].idxmax(), 'hospital_id']
                        
                        findings = [
                            f"**Contribution Range**: {contribution_df['contribution'].min():.4f} to {contribution_df['contribution'].max():.4f}",
                            f"**Size-Contribution Correlation**: {correlation:.3f} - {'Strong positive' if correlation > 0.7 else 'Moderate' if correlation > 0.3 else 'Weak'} correlation. {'Larger hospitals contribute more' if correlation > 0.5 else 'Contribution not strongly tied to size'}",
                            f"**Critical Hospitals**: Hospital {int(max_hospital)} is most critical with contribution of {max_contrib:.4f}",
                            f"**Redundancy**: {'Low redundancy - all hospitals important' if contribution_df['contribution'].min() > 0.001 else 'Some hospitals may be redundant'}"
                        ]
                        render_key_findings(findings)
                        
                        st.markdown("""
                        **Implications:**
                        - Use this to prioritize hospital recruitment
                        - Identify critical vs redundant participants
                        - Optimize consortium composition
                        """)
                        
                        # Save results
                        exp.save_dataframe(contribution_df, 'hospital_contributions')
                        exp.log_results({
                            'baseline_auc': float(contribution_df['baseline_auc'].iloc[0]),
                            'max_contribution': float(max_contrib),
                            'mean_contribution': float(contribution_df['contribution'].mean()),
                            'size_contribution_correlation': float(correlation)
                        })
                        exp.generate_summary_report()
                        
                        render_info_box(f"✅ Results saved to {exp.experiment_dir}", 'success')
                        
                    except Exception as e:
                        render_experiment_status('error', f'Analysis failed: {str(e)}')
                        import traceback
                        st.code(traceback.format_exc())
            
            # Information about other VERSION-5 features
            render_divider()
            render_section_header("🚀 Additional VERSION-5 Features", "Backend modules ready for integration")
            
            st.markdown("""
            **Backend Ready (UI Integration Pending):**
            
            🧬 **Multi-Modal Learning**
            - Clinical + Protein expression data
            - PCA dimensionality reduction
            - Feature selection methods
            
            📊 **Experiment Management**
            - Automatic experiment logging
            - Reproducibility controls
            - Timestamped results
            
            📈 **Statistical Validation**
            - Bootstrap confidence intervals
            - Paired statistical tests
            - Effect size calculations
            
            ⚖️ **Fairness Analysis**
            - Subgroup performance evaluation
            - Disparity metrics
            - Bias detection
            
            *These features are implemented in the backend and can be accessed programmatically.*
            """)
    
    else:
        render_info_box("👈 Please upload clinical dataset to begin", 'info')
        
        render_divider()
        render_section_header("📖 About This Application", "Learn about each version and its capabilities")
        
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
            
            **Expected Results:**
            - FedAvg AUC ≈ Centralized AUC (with enough rounds)
            - Local AUC < FedAvg AUC (benefits of collaboration)
            """)
        elif "VERSION-3" in version:
            st.markdown("""
            **VERSION-3: Sustainability & Free-Rider Analysis**
            
            This version studies the sustainability and scalability of federated learning.
            
            **Research Questions:**
            1. **Scalability**: How does performance change as we add more hospitals?
            2. **Free-Riding**: Can non-participating hospitals benefit from the global model?
            3. **Sustainability**: Is federated learning sustainable at scale?
            
            **Why This Matters:**
            - Understanding free-rider benefits helps design participation incentives
            - Knowing performance limits helps plan federated deployments
            """)
        elif "VERSION-4" in version:
            st.markdown("""
            **VERSION-4: FedProx & Non-IID Study**
            
            Study how FedProx handles data heterogeneity compared to FedAvg.
            
            **Key Features:**
            - **Proximal regularization** prevents client drift
            - **Dirichlet non-IID** simulates realistic heterogeneity
            - **Convergence analysis** shows stability improvements
            
            **When to use FedProx:**
            - Strong data heterogeneity (Dirichlet α < 1)
            - Unstable FedAvg convergence
            - Need for convergence guarantees
            """)
        elif "VERSION-5" in version:
            st.markdown("""
            **VERSION-5: Research Lab - Advanced Analysis**
            
            Publication-quality research tools for federated learning.
            
            **Available Features:**
            - 🏥 **Hospital Contribution Analysis**: Measure each hospital's impact
            - 📊 **Experiment Management**: Reproducible research with automatic logging
            - 🧬 **Multi-Modal Support**: Clinical + Protein data (backend ready)
            - ⚖️ **Fairness Analysis**: Subgroup performance evaluation
            - 📈 **Statistical Validation**: Bootstrap confidence intervals
            """)
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
