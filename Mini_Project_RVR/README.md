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

**VERSION-4** (Current):
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
