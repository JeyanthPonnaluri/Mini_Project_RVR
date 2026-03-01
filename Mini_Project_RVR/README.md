# TCGA-PRAD Clinical Stage Classification - VERSION 1

## Project Overview

This project implements a machine learning pipeline to predict pathologic T stage (advanced vs early) for prostate adenocarcinoma (PRAD) patients using clinical features from TCGA data.

## Why Clinical Stage Classification?

### Problem with Survival-Based Targets

In our initial approach, we attempted to predict survival outcomes by merging clinical, survival, and protein expression data. This led to several critical issues:

1. **Extreme Class Imbalance**: After data cleaning and merging, the survival target had severe class imbalance, with some test sets containing only one class
2. **Meaningless AUC**: With only one class in the test set, AUC-ROC returned NaN values
3. **Limited Clinical Utility**: Survival prediction requires longitudinal follow-up data and is affected by many external factors

### Solution: Pathologic T Stage Classification

We redesigned VERSION-1 to predict **pathologic T stage**, which offers:

1. **Clinical Relevance**: T stage is a key prognostic factor in prostate cancer
2. **Better Class Balance**: More balanced distribution between early (T1/T2) and advanced (T3/T4) stages
3. **Actionable Insights**: Helps with treatment planning and risk stratification
4. **Simpler Data Requirements**: Uses only clinical data, no complex merging needed

## Target Variable Definition

**Binary Classification Task:**
- **Class 1 (Advanced Stage)**: Tumors staged as T3 or T4 (locally advanced, extending beyond prostate)
- **Class 0 (Early Stage)**: Tumors staged as T1 or T2 (organ-confined)

**Clinical Significance:**
- T1/T2: Tumor confined to prostate → Better prognosis, less aggressive treatment
- T3/T4: Tumor extends beyond prostate → Worse prognosis, more aggressive treatment needed

## Project Structure

```
Mini_Project_RVR/
│
├── data/                          # Place your datasets here
├── datasets/                      # Original TCGA datasets
│   └── TCGA-PRAD.clinical.tsv/
├── src/
│   ├── preprocessing.py           # Data loading and preprocessing
│   ├── model.py                   # Model training and prediction
│   ├── evaluation.py              # Model evaluation metrics
│   └── app.py                     # Streamlit application
├── notebooks/
│   └── version1_exploration.ipynb # Exploratory analysis
├── reports/                       # Generated plots and reports
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## Technical Details

### Data Preprocessing

1. **Load Clinical Data**: Read TCGA-PRAD clinical TSV file
2. **Create Target**: Extract pathologic T stage, create binary target (T3/T4 vs others)
3. **Remove Identifiers**: Drop patient IDs and other identifier columns
4. **Handle Missing Values**: Remove columns with >50% missing data
5. **Feature Engineering**:
   - Numerical features: StandardScaler normalization
   - Categorical features: OneHotEncoder (drop first to avoid multicollinearity)
6. **Quality Checks**: Ensure no NaN values in final feature matrix

### Model

- **Algorithm**: Logistic Regression
- **Solver**: liblinear (good for small-medium datasets)
- **Class Weights**: balanced (handles class imbalance)
- **Random Seed**: 42 (reproducibility)

### Train-Test Split

- **Split Ratio**: 80% train, 20% test
- **Strategy**: Stratified (maintains class distribution)
- **Random Seed**: 42

### Evaluation Metrics

1. **AUC-ROC**: Area under ROC curve (overall discrimination)
2. **Accuracy**: Percentage of correct predictions
3. **Confusion Matrix**: True/False Positives/Negatives
4. **Sensitivity (Recall)**: Ability to detect advanced stage
5. **Specificity**: Ability to identify early stage

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this project

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure your clinical dataset is available:
   - Place `TCGA-PRAD.clinical.tsv` in the `datasets/` folder
   - Or upload via the Streamlit interface

## How to Run

### Run the Streamlit Application

```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload Data**: Click "Browse files" and upload `TCGA-PRAD.clinical.tsv`
2. **Review Target**: Check class distribution (Early vs Advanced stage)
3. **Train Model**: Click "Train Model" button
4. **View Results**:
   - AUC-ROC score and Accuracy
   - Confusion Matrix visualization
   - ROC Curve plot
   - Sensitivity and Specificity metrics
5. **Results**: Automatically saved to `reports/` folder

## Features

- **Clinically Meaningful Task**: Predicts actionable pathologic T stage
- **Balanced Classes**: Better class distribution than survival prediction
- **Robust Preprocessing**: Handles missing data, encodes categoricals, scales numericals
- **Class Imbalance Handling**: Uses balanced class weights
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Interactive UI**: Streamlit-based interface for easy interaction
- **Modular Code**: Clean separation of concerns across modules
- **Error Handling**: Graceful handling of missing columns and data issues

## Model Performance Expectations

Based on the TCGA-PRAD dataset characteristics:

- **Expected AUC**: 0.65 - 0.80 (clinical features alone)
- **Expected Accuracy**: 60% - 75%
- **Class Distribution**: Typically 70-80% early stage, 20-30% advanced stage

Note: Performance depends on data quality and feature availability after preprocessing.

## Version History

**VERSION-2** (Current):
- Task: Federated Learning with FedAvg algorithm
- Implementation: Manual NumPy-based logistic regression
- Features: Hospital data partitioning, weighted aggregation, convergence tracking
- Comparison: Centralized vs FedAvg vs Local models

**VERSION-1**:
- Task: Clinical stage classification (T3/T4 vs T1/T2)
- Data: TCGA-PRAD clinical features only
- Model: Logistic Regression with balanced class weights (sklearn)
- Status: Baseline implementation

---

# VERSION-2: Federated Learning (FedAvg)

## Overview

VERSION-2 implements Federated Averaging (FedAvg), a foundational federated learning algorithm that enables collaborative model training across multiple hospitals without sharing raw patient data.

## Why Federated Learning?

**Privacy-Preserving**: Hospitals keep patient data locally - only model weights are shared  
**Collaborative**: Multiple institutions benefit from combined knowledge  
**Regulatory Compliance**: Meets data privacy regulations (HIPAA, GDPR)  
**Realistic**: Simulates real-world multi-institutional medical research

## FedAvg Algorithm

### Mathematical Formulation

**Initialization:**
```
w_global ← initialize_weights()
```

**For each communication round t = 1, 2, ..., T:**

1. **Broadcast**: Send w_global to all K hospitals

2. **Local Training**: Each hospital k trains locally
   ```
   w_k^(t) ← LocalTrain(X_k, y_k, w_global^(t-1), E, η)
   ```
   where:
   - E = number of local epochs
   - η = learning rate

3. **Weighted Aggregation**:
   ```
   w_global^(t) ← Σ(n_k / n_total) × w_k^(t)
   ```
   where:
   - n_k = number of samples at hospital k
   - n_total = total samples across all hospitals

**Return**: w_global^(T)

### Local Training (Gradient Descent)

For each local epoch:
```
1. Compute predictions: ŷ = sigmoid(X @ w)
2. Compute gradient: ∇L = (1/n) × X^T @ (ŷ - y)
3. Update weights: w ← w - η × ∇L
```

**Loss Function**: Binary Cross-Entropy
```
L(w) = -(1/n) × Σ[y log(ŷ) + (1-y) log(1-ŷ)]
```

## Why Manual NumPy Implementation?

**sklearn Limitation**: sklearn's `.fit()` method doesn't expose intermediate weights needed for federated aggregation

**Full Control**: Manual implementation provides:
- Direct access to weights for aggregation
- Custom training loops for local epochs
- Gradient computation for debugging
- Deterministic behavior with fixed seeds

**Educational Value**: Understanding the mathematics behind logistic regression and federated learning

## Implementation Details

### File Structure

```
src/
├── logistic_numpy.py      # Manual logistic regression
│   ├── sigmoid()
│   ├── initialize_weights()
│   ├── compute_loss()
│   ├── compute_gradient()
│   ├── local_train()
│   └── predict_proba()
│
├── federated.py           # FedAvg implementation
│   ├── partition_equal()
│   ├── fedavg_train()
│   └── train_local_models()
│
└── experiments.py         # Experiment utilities
    ├── centralized_train_numpy()
    ├── save_fedavg_metrics()
    └── save_comparison_summary()
```

### Hospital Data Partitioning

**Strategy**: Stratified equal partitioning
- Each hospital receives equal number of samples
- Class distribution maintained (stratification)
- Random shuffling with fixed seed

**Example**: 450 samples, 5 hospitals
- Hospital 1: 90 samples (stratified)
- Hospital 2: 90 samples (stratified)
- ...
- Hospital 5: 90 samples (stratified)

### Weighted Aggregation

Hospitals with more data have proportionally more influence:

```python
w_global = Σ(weight_k × w_k)
where weight_k = n_k / n_total
```

**Example**: 3 hospitals with 100, 200, 100 samples
- Hospital 1 weight: 100/400 = 0.25
- Hospital 2 weight: 200/400 = 0.50
- Hospital 3 weight: 100/400 = 0.25

## Experiments

### 1. Centralized Training (NumPy)

Train on all data using manual logistic regression.

**Purpose**: Baseline for comparison  
**Expected**: Best possible AUC (no data distribution)

### 2. FedAvg Training

Federated training across K hospitals.

**Purpose**: Evaluate federated learning performance  
**Expected**: AUC close to centralized (with sufficient rounds)

### 3. Local Models

Each hospital trains independently (no collaboration).

**Purpose**: Show benefit of federation  
**Expected**: Lower AUC than FedAvg (limited data per hospital)

## Hyperparameters

**Number of Hospitals (K)**: 2-10
- More hospitals → more realistic but slower convergence
- Fewer hospitals → faster but less federated

**Communication Rounds (T)**: 10-100
- More rounds → better convergence
- Diminishing returns after convergence

**Local Epochs (E)**: 1-10
- More epochs → better local training but risk of overfitting
- Fewer epochs → more communication needed

**Learning Rate (η)**: 0.001-1.0
- Higher → faster convergence but risk of instability
- Lower → stable but slower convergence

## Expected Results

**Typical Performance** (5 hospitals, 50 rounds, 5 local epochs):
- Centralized AUC: 0.85
- FedAvg AUC: 0.83-0.84 (2-3% gap)
- Average Local AUC: 0.75-0.78

**Convergence**: FedAvg typically converges within 30-50 rounds

**Gap Analysis**:
- FedAvg vs Centralized: Small gap due to data heterogeneity
- FedAvg vs Local: Significant improvement from collaboration

## Running VERSION-2

```bash
streamlit run src/app.py
```

1. Select "VERSION-2: Federated Learning (FedAvg)"
2. Upload clinical data
3. Configure parameters:
   - Number of hospitals
   - Communication rounds
   - Local epochs
   - Learning rate
4. Run experiments:
   - Centralized (NumPy)
   - FedAvg
   - Local Models
5. View comparison summary

## Output Files

```
reports/version2/
├── fedavg_round_metrics.csv      # Round-wise metrics
├── comparison_summary.txt         # Performance comparison
└── fedavg_convergence.png         # Convergence plots
```

## Key Insights

**Privacy**: Raw data never leaves hospitals  
**Collaboration**: Hospitals benefit from shared learning  
**Performance**: FedAvg achieves near-centralized performance  
**Convergence**: Stable learning with proper hyperparameters  
**Scalability**: Handles 2-10 hospitals efficiently

## Future Improvements

Potential enhancements for future versions:

1. **Feature Engineering**: Add domain-specific features (PSA ratios, Gleason score combinations)
2. **Advanced Models**: Try Random Forest, XGBoost, or Neural Networks
3. **Feature Selection**: Identify most important clinical predictors
4. **Cross-Validation**: Implement k-fold CV for more robust evaluation
5. **Multi-Class**: Predict all T stages (T1, T2a, T2b, T2c, T3a, T3b, T4)
6. **Integration**: Combine with molecular data (protein, RNA) in future versions

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
streamlit
```

## License

This project uses publicly available TCGA data. Please cite TCGA appropriately if using this code for research.

## Contact

For questions or issues, please open an issue in the repository.

---

**Git Commit Message:**
```
feat: version-1 clinical stage classification baseline

- Replace survival prediction with pathologic T stage classification
- Use only clinical data (no merging required)
- Implement binary classification: T3/T4 vs T1/T2
- Add balanced class weights to handle imbalance
- Comprehensive preprocessing with OneHotEncoder and StandardScaler
- Multiple evaluation metrics: AUC, accuracy, confusion matrix
- Updated Streamlit UI with detailed insights
- Resolve NaN AUC issue from previous survival-based approach
```
