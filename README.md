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

**VERSION-5** (Current - Research Edition):
- Task: Ultimate Federated Learning Research Framework
- Features: Multi-modal learning (clinical + protein), hospital contribution analysis, experiment management
- Advanced: Feature selection, PCA, reproducibility controls, automated logging
- Research-grade: Publication-ready visualizations, statistical validation, experiment tracking

**VERSION-4**:
- Task: FedProx & Non-IID Heterogeneity Study
- Features: FedProx algorithm, Dirichlet non-IID partitioning, convergence analysis
- Comparison: FedAvg vs FedProx under equal, imbalanced, and Dirichlet partitions
- Insights: Proximal regularization benefits, stability improvements, optimal μ selection

**VERSION-3**:
- Task: Sustainability & Free-Rider Analysis
- Features: Learning curve experiments, free-rider detection, scalability analysis, partition comparison
- Insights: Performance vs number of hospitals, non-participant benefits, data heterogeneity impact
- Tools: Monte Carlo trials, statistical testing, visualization

**VERSION-2**:
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

# VERSION-3: Sustainability & Free-Rider Analysis

## Overview

VERSION-3 extends federated learning research to study **sustainability** and **scalability** through systematic experiments analyzing how performance changes with the number of participating hospitals and the impact of non-participating (free-rider) institutions.

## Research Questions

### 1. Scalability
**Question**: How does federated learning performance scale with the number of hospitals?

**Hypothesis**: More hospitals → more data → better performance, but with diminishing returns

**Method**: Learning curve experiment

### 2. Free-Riding
**Question**: Can non-participating hospitals benefit from the global model without contributing their data?

**Hypothesis**: Free-riders benefit from collaboration but achieve lower performance than active participants

**Method**: Free-rider experiment

### 3. Sustainability
**Question**: Is federated learning sustainable and effective at scale?

**Analysis**: Combined insights from scalability and free-rider experiments

## Experiments

### Learning Curve Experiment

**Purpose**: Study performance vs number of hospitals

**Algorithm**:
```
For each K in [2, 4, 6, 8, 10, ...]:
    For trial in 1..N:
        1. Partition data into K hospitals (stratified)
        2. Train FedAvg on all K hospitals
        3. Train K local models independently
        4. Evaluate:
           - global_auc (FedAvg on test set)
           - avg_local_auc (average of local models)
        5. Store results
    
    Compute statistics: mean, std, min, max
```

**Output**:
- Learning curve plot: K vs AUC (with confidence intervals)
- Summary table: statistics per K value
- CSV: detailed results for all trials

**Expected Pattern**:
```
AUC
 │
 │     ┌─────────  (plateau)
 │    ╱
 │   ╱
 │  ╱
 │ ╱
 └─────────────────── K (hospitals)
  2  4  6  8  10  12
```

**Interpretation**:
- **Initial growth**: Performance improves rapidly with first few hospitals
- **Diminishing returns**: Improvement slows as K increases
- **Plateau**: Performance stabilizes after certain K
- **Gap**: FedAvg consistently outperforms local models

### Free-Rider Experiment

**Purpose**: Analyze non-participant benefits

**Scenario**: Hospital that doesn't contribute data but uses global model

**Algorithm**:
```
For each K in [2, 4, 6, 8, 10, ...]:
    For trial in 1..N:
        1. Partition data into K hospitals
        2. Randomly select 1 hospital as "free-rider"
        3. Train FedAvg on remaining K-1 hospitals
        4. Evaluate global model on:
           - Test set (global_auc)
           - Free-rider's data (free_rider_auc)
        5. Store results
    
    Compute statistics: mean, std, min, max
```

**Output**:
- Free-rider curve plot: K vs Free-Rider AUC
- Comparison: Free-rider vs Participants
- CSV: detailed results

**Expected Pattern**:
```
AUC
 │
 │  Participants ──────────
 │                    ╱
 │  Free-Rider  ────╱
 │                 ╱
 │               ╱
 │             ╱
 └─────────────────── K (hospitals)
  2  4  6  8  10  12
```

**Interpretation**:
- **Free-rider benefits**: Non-participants still achieve reasonable performance
- **Performance gap**: Free-riders perform worse than active participants
- **Scaling effect**: Gap may narrow as K increases (more diverse global model)

## Implementation Details

### Monte Carlo Trials

**Why**: Account for randomness in data partitioning and training

**Method**:
- Run each configuration N times (default: 10 trials)
- Use deterministic seeds: `seed = base_seed + trial * 100 + K`
- Compute statistics: mean, std, min, max

**Benefits**:
- Statistical significance
- Confidence intervals
- Robustness to random variations

### Partition Types

**Equal Partitioning**:
- Each hospital receives equal number of samples
- Maintains stratification (class balance)
- Ideal scenario

**Imbalanced Partitioning**:
- Hospitals receive different amounts of data
- Distribution: randomly generated, sums to 1.0
- Realistic scenario (hospitals vary in size)

### Visualization

**Learning Curve Plot**:
- X-axis: Number of hospitals (K)
- Y-axis: AUC score
- Lines: FedAvg (global) vs Local (average)
- Shading: ±1 standard deviation (confidence interval)

**Free-Rider Plot**:
- X-axis: Number of hospitals (K)
- Y-axis: AUC score
- Lines: Free-rider vs Participants (global)
- Shading: ±1 standard deviation

## Key Findings (Expected)

### Scalability Insights

1. **Optimal K**: Performance plateaus around K=8-10 hospitals
2. **Diminishing Returns**: Adding hospitals beyond plateau provides minimal benefit
3. **Collaboration Benefit**: FedAvg consistently outperforms local models by 5-10%

### Free-Rider Insights

1. **Benefit Without Contribution**: Free-riders achieve 80-90% of participant performance
2. **Incentive Problem**: Low cost of free-riding may discourage participation
3. **Scaling Effect**: Free-rider performance improves with more participants

### Sustainability Insights

1. **Scalable**: Federated learning remains effective with 10+ hospitals
2. **Robust**: Performance stable across different partitioning strategies
3. **Practical**: Real-world deployment feasible despite free-riders

## Running VERSION-3

```bash
streamlit run src/app.py
```

**Steps**:
1. Select "VERSION-3: Sustainability Analysis"
2. Upload clinical data
3. Configure parameters:
   - Max hospitals (2-15)
   - Monte Carlo trials (5-20)
   - Partition type (equal/imbalanced)
   - Rounds, epochs, learning rate
4. Run experiments:
   - Learning Curve
   - Free-Rider Analysis
   - **Partition Comparison** (NEW)
5. View plots and statistics

## Equal vs Imbalanced Federated Sustainability

### Why Data Heterogeneity Matters

In real-world federated learning deployments, hospitals have **naturally imbalanced data sizes**:
- Large academic medical centers: 1000s of patients
- Community hospitals: 100s of patients
- Rural clinics: 10s of patients

This **data heterogeneity** significantly impacts:
1. **Convergence**: Larger hospitals dominate the global model
2. **Fairness**: Smaller hospitals may underfit
3. **Free-Riding**: Incentive structures change with imbalance

### Partition Comparison Experiment

**Objective**: Quantify the impact of data heterogeneity on federated learning performance

**Methodology**:
- **Equal Partition**: All hospitals have equal data (baseline)
- **Imbalanced Partition**: Long-tail distribution (realistic)
  - Default: [35%, 25%, 15%, 10%, 8%, 5%, 2%]
  - Automatically generated for K > 7 hospitals
- **Statistical Testing**: Paired t-test for significance (α = 0.05)

**Metrics Compared**:
1. Global AUC (FedAvg performance)
2. Free-Rider AUC (non-participant benefit)
3. Variance across trials

### Expected Research Findings

**Under Imbalanced Partition**:
1. **Global AUC**: Slightly decreased (2-5% typical)
   - Larger hospitals dominate aggregation
   - Smaller hospitals contribute less effectively
   
2. **Free-Rider AUC**: May decrease more significantly
   - Free-riders with small data benefit less
   - Global model biased toward large hospitals
   
3. **Variance**: Increased instability
   - More sensitive to random initialization
   - Convergence less predictable

### Why FedProx is Needed Next

**FedAvg Limitations with Heterogeneity**:
- Assumes all hospitals converge to same optimum
- No mechanism to handle statistical heterogeneity
- Larger hospitals can cause divergence

**FedProx Solution**:
- Adds proximal term: `μ/2 ||w - w_global||²`
- Limits local drift from global model
- Better handles non-IID data
- Improves convergence under heterogeneity

**Other Approaches**:
- **FedNova**: Normalized averaging to handle different local steps
- **Personalized FL**: Allow hospital-specific models
- **Clustered FL**: Group similar hospitals

## Output Files

```
reports/version3/
├── learning_curve_results.csv     # Detailed results
├── learning_curve_plot.png        # Visualization
├── free_rider_results.csv         # Detailed results
└── free_rider_plot.png            # Visualization

reports/version3_partition_comparison/
├── comparison_results.csv         # Equal vs Imbalanced comparison
├── comparison_plot.png            # Side-by-side visualization
└── statistical_test_results.txt   # Paired t-test results & interpretation
```

## Research Implications

### For Healthcare

**Collaboration Incentives**:
- Design mechanisms to encourage participation
- Quantify benefits of contribution vs free-riding
- Fair resource allocation
- **NEW**: Account for data size heterogeneity in incentive design

**Deployment Planning**:
- Determine optimal consortium size
- **NEW**: Assess impact of hospital size imbalance
- **NEW**: Choose appropriate FL algorithm (FedAvg vs FedProx)
- Predict performance with expected participants
- Plan for non-participating institutions

### For Federated Learning

**Scalability Limits**:
- Understand performance saturation
- Optimize communication costs
- Balance K vs performance

**Robustness**:
- Handle heterogeneous participation
- Maintain performance with dropouts
- Design fair aggregation schemes

## Theoretical Background

### Learning Curve Theory

**Statistical Learning**: More data → better generalization (up to a point)

**Federated Context**: More hospitals → more diverse data → better global model

**Limitations**:
- Data heterogeneity across hospitals
- Communication overhead
- Diminishing marginal utility

### Free-Rider Problem

**Game Theory**: Rational agents may choose not to contribute if they can benefit without cost

**Federated Context**: Hospitals may use global model without sharing data

**Solutions**:
- Contribution-based access control
- Performance-based incentives
- Reputation systems

## Future Extensions

1. **Dynamic Participation**: Hospitals join/leave over time
2. **Contribution Metrics**: Quantify each hospital's contribution
3. **Incentive Mechanisms**: Reward active participants
4. **Heterogeneity Analysis**: Impact of data quality differences
5. **Communication Costs**: Trade-off between K and communication overhead

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


---

# VERSION-4: Handling Heterogeneity with FedProx

## Overview

VERSION-4 addresses the **client drift problem** in federated learning by implementing **FedProx** (Federated Proximal), an algorithm designed to handle data heterogeneity more effectively than FedAvg.

## The Client Drift Problem

### What is Client Drift?

In FedAvg, each hospital trains independently on its local data. When data is **non-IID** (non-identically and independently distributed), local models can drift significantly from the global model, causing:

1. **Convergence instability**: Oscillating loss/AUC curves
2. **Performance degradation**: Lower final accuracy
3. **Fairness issues**: Some hospitals benefit more than others

### Why Does It Happen?

- **Data heterogeneity**: Different hospitals have different class distributions
- **Local optimization**: Each hospital optimizes for its own data
- **Conflicting gradients**: Updates from different hospitals may contradict each other

## FedProx Solution

### Algorithm

FedProx adds a **proximal term** to the local objective function:

```
L_k(w) = cross_entropy(w) + (μ/2) * ||w - w_global||²
```

Where:
- `cross_entropy(w)`: Standard loss on local data
- `μ`: Proximal coefficient (regularization strength)
- `||w - w_global||²`: L2 distance from global model

### How It Works

1. **Prevents drift**: Proximal term penalizes deviation from global model
2. **Balances adaptation**: Allows local learning while maintaining global consistency
3. **Stabilizes convergence**: Reduces oscillations in training

### Gradient Update

```python
gradient = ∇cross_entropy + μ * (w_local - w_global)
```

The proximal term pulls local weights back toward the global model.

## Dirichlet Non-IID Simulation

### What is Dirichlet Distribution?

The Dirichlet distribution is used to simulate realistic data heterogeneity by controlling class proportions across hospitals.

### Alpha Parameter (α)

- **α < 1** (e.g., 0.1): **Strong non-IID**
  - Each hospital has very different class distributions
  - Some hospitals may have mostly one class
  - Realistic for specialized medical centers

- **α ≈ 1** (e.g., 0.5-2): **Moderate non-IID**
  - Noticeable heterogeneity but not extreme
  - Typical real-world scenario

- **α > 10**: **Nearly IID**
  - All hospitals have similar class distributions
  - Approaches centralized learning

### Example

With α=0.1 and 5 hospitals:
- Hospital 1: 90% class 0, 10% class 1
- Hospital 2: 20% class 0, 80% class 1
- Hospital 3: 70% class 0, 30% class 1
- Hospital 4: 40% class 0, 60% class 1
- Hospital 5: 85% class 0, 15% class 1

## Convergence Analysis

### Metrics Tracked

1. **Round-wise AUC**: Test performance per communication round
2. **Round-wise Loss**: Training loss per round
3. **Weight Drift**: L2 norm of weight changes between rounds
4. **Convergence Stability**: Standard deviation of AUC in last 10 rounds

### Expected Behavior

**Under Strong Non-IID (α=0.1)**:
- FedAvg: Oscillating convergence, lower final AUC
- FedProx: Smoother convergence, higher final AUC
- Optimal μ: Typically 0.01-0.1

**Under Nearly IID (α=10)**:
- FedAvg and FedProx: Similar performance
- Proximal term has minimal effect

## Running VERSION-4

```bash
streamlit run src/app.py
```

**Steps**:
1. Select "VERSION-4: FedProx & Non-IID Study"
2. Upload clinical data
3. Configure:
   - Number of hospitals (3-10)
   - Partition strategy: equal, imbalanced, or **dirichlet**
   - If dirichlet: Set α (0.1 for strong non-IID, 10 for nearly IID)
   - Proximal coefficients: μ₁, μ₂, μ₃ (e.g., 0.01, 0.1, 0.5)
   - Rounds, epochs, learning rate
4. Click "Run FedAvg vs FedProx Comparison"
5. View:
   - Performance comparison table
   - Convergence curves (AUC and Loss vs rounds)
   - Stability comparison (convergence std, weight drift)
   - Key insights and interpretation

## Output Files

```
reports/version4_fedprox/
├── comparison_summary.csv         # Summary metrics
├── detailed_results.txt           # Full analysis with interpretation
├── convergence_plot.png           # AUC and Loss curves
└── stability_plot.png             # Stability comparison
```

## Research Insights

### When to Use FedProx

✅ **Use FedProx when:**
- Data is highly heterogeneous (Dirichlet α < 1)
- FedAvg shows unstable convergence
- Need convergence guarantees
- Fairness across hospitals is important

❌ **FedAvg is sufficient when:**
- Data is nearly IID (α > 10)
- Equal or balanced partitions
- Convergence is already stable

### Optimal μ Selection

- **Too small (μ < 0.001)**: Minimal effect, similar to FedAvg
- **Optimal (μ ≈ 0.01-0.1)**: Best balance between local adaptation and global consistency
- **Too large (μ > 1)**: Over-regularization, limits local learning

### Performance Expectations

Under Dirichlet α=0.1:
- FedAvg AUC: ~0.75-0.78
- FedProx AUC: ~0.78-0.82 (3-5% improvement)
- Convergence std reduction: 30-50%

## Theoretical Background

### FedProx Paper

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
**Federated Optimization in Heterogeneous Networks**
*MLSys 2020*

Key contributions:
1. Proximal term for handling heterogeneity
2. Convergence guarantees under non-IID data
3. Partial work allowance (variable local epochs)

### Why It Works

1. **Regularization**: Proximal term acts as regularizer
2. **Variance reduction**: Limits gradient variance across clients
3. **Convergence theory**: Provable convergence under heterogeneity

## Next Steps

After VERSION-4, consider:

1. **FedNova**: Normalized averaging for variable local steps
2. **Personalized FL**: Client-specific models
3. **Clustered FL**: Group similar hospitals
4. **Adaptive μ**: Dynamic proximal coefficient
5. **Privacy**: Differential privacy with FedProx

## Code Structure

```python
# FedProx local training
def local_train_fedprox(X, y, w_global, epochs, lr, mu):
    for epoch in range(epochs):
        grad_ce = compute_gradient(X, y, w)
        grad_prox = mu * (w - w_global)  # Proximal term
        grad = grad_ce + grad_prox
        w = w - lr * grad
    return w

# Dirichlet partition
def partition_dirichlet(X, y, num_hospitals, alpha):
    for each class:
        proportions = Dirichlet(alpha * ones(K))
        assign samples based on proportions
    return hospitals
```

## Research Questions Answered

1. **Does FedProx improve performance under non-IID?** → Yes, especially for α < 1
2. **How does μ affect convergence?** → Optimal μ balances local/global
3. **Is Dirichlet realistic?** → Yes, models real-world heterogeneity
4. **When is FedAvg sufficient?** → When data is nearly IID (α > 10)

## Practical Implications

### For Healthcare Deployment

1. **Hospital diversity**: Real hospitals have heterogeneous data
2. **Algorithm selection**: Use FedProx for diverse consortiums
3. **Hyperparameter tuning**: Cross-validate μ on validation set
4. **Monitoring**: Track convergence stability in production

### For Research

1. **Baseline comparison**: Always compare against FedProx
2. **Heterogeneity simulation**: Use Dirichlet with multiple α values
3. **Ablation studies**: Test different μ values
4. **Convergence analysis**: Report stability metrics

---


---

# VERSION-5: Ultimate Federated Learning Research Framework

## Overview

VERSION-5 represents the **ultimate research edition** of the federated learning framework, designed for publication-quality experiments with comprehensive analysis capabilities, multi-modal data support, and rigorous statistical validation.

## Key Features

### 1. Multi-Modal Federated Learning

**Clinical + Protein Expression Integration**
- Merge clinical and protein expression data
- Handle missing protein values (>30% threshold)
- Standardization and normalization
- Optional PCA for dimensionality reduction

**Data Modes:**
- Clinical only (baseline)
- Protein only (molecular features)
- Combined (multi-modal fusion)

### 2. Hospital Contribution Analysis

**Leave-One-Out Impact Measurement**
- Train baseline model with all hospitals
- For each hospital: retrain without it
- Measure AUC drop = contribution score
- Visualize contribution vs hospital size

**Insights:**
- Which hospitals contribute most?
- Is contribution proportional to data size?
- Identify critical vs redundant participants

### 3. Advanced Feature Engineering

**Feature Selection Methods:**
- Variance Threshold: Remove low-variance features
- L1 Regularization: Sparse feature selection
- PCA: Dimensionality reduction (retain 95% variance)
- No reduction: Use all features

**Benefits:**
- Reduced communication cost
- Faster convergence
- Better generalization

### 4. Experiment Management System

**Reproducibility Controls:**
- Timestamped experiment directories
- Configuration JSON with hash ID
- Automatic result logging
- Global seed management

**Directory Structure:**
```
reports/version5/
└── exp_YYYYMMDD_HHMMSS_<hash>/
    ├── config.json
    ├── results.json
    ├── summary_report.txt
    ├── plots/
    │   ├── convergence.png
    │   ├── contribution.png
    │   └── ...
    └── *.csv
```

### 5. Statistical Validation (Planned)

- Bootstrap confidence intervals (95%)
- Paired t-tests for algorithm comparison
- Wilcoxon signed-rank tests
- Effect size calculations

### 6. Fairness Analysis (Planned)

**Subgroup Evaluation:**
- Age groups (<65 vs ≥65)
- Race/ethnicity (if available)
- Tumor stage distribution

**Metrics:**
- AUC per subgroup
- TPR per subgroup
- Disparity scores

## New Modules

### `contribution.py`

Hospital contribution analysis module.

**Functions:**
- `measure_hospital_contribution()`: Leave-one-out analysis
- `plot_contribution_analysis()`: Visualization

**Usage:**
```python
from contribution import measure_hospital_contribution, plot_contribution_analysis

contribution_df = measure_hospital_contribution(
    hospitals, X_test, y_test,
    rounds=30, epochs=5, lr=0.1,
    algorithm='fedavg'
)

fig = plot_contribution_analysis(contribution_df, save_path='contribution.png')
```

### `experiment_manager.py`

Experiment tracking and reproducibility.

**Class: ExperimentManager**

**Methods:**
- `create_experiment(config)`: Initialize experiment with config
- `log_results(results)`: Log experiment results
- `save_dataframe(df, name)`: Save DataFrame to experiment dir
- `get_plot_path(plot_name)`: Get path for plots
- `generate_summary_report()`: Create text summary
- `load_experiment(dir)`: Load previous experiment

**Usage:**
```python
from experiment_manager import ExperimentManager, set_global_seed

# Set global seed
set_global_seed(42)

# Create experiment
exp = ExperimentManager()
exp_id = exp.create_experiment({
    'algorithm': 'fedprox',
    'mu': 0.1,
    'rounds': 50,
    'hospitals': 5
})

# Log results
exp.log_results({'final_auc': 0.85, 'convergence_std': 0.02})

# Save dataframes
exp.save_dataframe(results_df, 'comparison_results')

# Generate report
exp.generate_summary_report()
```

### Extended `preprocessing.py`

Multi-modal data preprocessing.

**New Functions:**
- `load_protein(file_path)`: Load protein expression data
- `merge_clinical_protein(clinical_df, protein_df)`: Merge modalities
- `preprocess_protein(protein_df)`: Protein-specific preprocessing
- `apply_pca(X, variance_threshold)`: PCA dimensionality reduction
- `apply_feature_selection(X, y, method)`: Feature selection

**Usage:**
```python
from preprocessing import (
    load_clinical, load_protein, merge_clinical_protein,
    preprocess_protein, apply_pca
)

# Load data
clinical_df = load_clinical('clinical.tsv')
protein_df = load_protein('protein.tsv')

# Merge
merged_df = merge_clinical_protein(clinical_df, protein_df)

# Preprocess protein
X_protein, protein_names = preprocess_protein(protein_df)

# Apply PCA
X_pca, pca_model, n_components = apply_pca(X_protein, variance_threshold=0.95)
```

## Research Capabilities

### Contribution Analysis Workflow

1. **Setup**: Partition data across hospitals
2. **Baseline**: Train with all hospitals
3. **Leave-One-Out**: For each hospital:
   - Remove hospital k
   - Retrain model
   - Measure AUC drop
4. **Analysis**: 
   - Rank hospitals by contribution
   - Correlate with sample size
   - Identify critical participants

### Multi-Modal Comparison

Compare three data modes:
1. Clinical only (baseline)
2. Protein only (molecular)
3. Combined (multi-modal)

**Expected Findings:**
- Combined > Clinical only
- Protein captures complementary information
- Federated learning preserves multi-modal benefits

### Experiment Reproducibility

Every experiment generates:
- Unique experiment ID with timestamp
- Configuration hash for verification
- Complete parameter logging
- Automatic result archiving

**Benefits:**
- Reproduce any experiment exactly
- Compare across experiments
- Track parameter sensitivity
- Publication-ready documentation

## Running VERSION-5

```bash
streamlit run src/app.py
```

**Steps:**
1. Select "VERSION-5: Research Lab" (when implemented)
2. Upload clinical data (required)
3. Upload protein data (optional)
4. Select data mode: clinical / protein / combined
5. Configure feature reduction: none / variance / PCA / L1
6. Set up federated learning parameters
7. Run experiments:
   - Standard training
   - Contribution analysis
   - Fairness evaluation (planned)
8. View results and download experiment package

## Output Structure

```
reports/version5/
├── exp_20260301_125900_a1b2c3d4/
│   ├── config.json                 # Experiment configuration
│   ├── results.json                # Numerical results
│   ├── summary_report.txt          # Human-readable summary
│   ├── contribution_analysis.csv   # Hospital contributions
│   ├── plots/
│   │   ├── convergence.png
│   │   ├── contribution_bar.png
│   │   ├── contribution_scatter.png
│   │   └── ...
│   └── ...
└── ...
```

## Research Questions Addressed

1. **Multi-Modal Impact**: Does combining clinical + protein improve federated learning?
2. **Hospital Value**: Which hospitals contribute most to global model?
3. **Data Efficiency**: Can we achieve similar performance with fewer hospitals?
4. **Feature Importance**: Which features drive federated performance?
5. **Reproducibility**: Can experiments be exactly reproduced?

## Code Quality Standards

✅ **Modular Design**: Clear separation of concerns
✅ **Type Hints**: All functions annotated
✅ **Docstrings**: Comprehensive documentation
✅ **Error Handling**: Robust exception management
✅ **Logging**: Detailed progress tracking
✅ **Testing**: Diagnostic checks throughout
✅ **Reproducibility**: Deterministic seeding

## Future Enhancements (VERSION-6 Ideas)

1. **Privacy**: Differential privacy mechanisms
2. **Communication**: Compression and quantization
3. **Personalization**: Client-specific model adaptation
4. **Robustness**: Byzantine-resilient aggregation
5. **Efficiency**: Asynchronous federated learning
6. **Deployment**: Production-ready API

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{mini_project_rvr_2026,
  title={Mini Project RVR: Federated Learning Research Framework},
  author={[Your Name]},
  year={2026},
  url={https://github.com/JeyanthPonnaluri/Mini_Project_RVR}
}
```

## Acknowledgments

This framework implements algorithms from:
- FedAvg: McMahan et al., 2017
- FedProx: Li et al., 2020
- Dirichlet Non-IID: Hsu et al., 2019

---

**VERSION-5 Status**: Core modules implemented (contribution analysis, experiment management, multi-modal preprocessing). Full Streamlit integration and statistical validation planned for next iteration.



## Deployment (Streamlit Cloud)

This project is configured for seamless deployment on Streamlit Cloud.

### Deployment Configuration

- **Main file path**: `src/app.py`
- **Branch**: `main`
- **Python version**: 3.10 (enforced via `runtime.txt`)

### Requirements

- **Python Version**: 3.10 (enforced via `runtime.txt`)
- **Dependencies**: Simplified for cloud compatibility (see `requirements.txt`)
- **Backend**: Matplotlib configured for headless environments (Agg backend)

### Configuration Files

- `runtime.txt`: Specifies Python 3.10 for consistent environment
- `requirements.txt`: Simplified dependencies without version pinning
- `.streamlit/config.toml`: Streamlit-specific configuration

### Why Python 3.10?

- Stable compatibility with all scientific libraries
- Proven track record with Streamlit Cloud
- Avoids breaking changes in Python 3.11+

### Cloud-Safe Features

- ✅ Matplotlib uses Agg backend (no GUI required)
- ✅ No strict version pinning (allows pip to resolve dependencies)
- ✅ Minimal dependencies for faster deployment
- ✅ Flattened repository structure for proper dependency installation

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).
