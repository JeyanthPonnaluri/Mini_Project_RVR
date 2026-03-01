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

**VERSION-1** (Current):
- Task: Clinical stage classification (T3/T4 vs T1/T2)
- Data: TCGA-PRAD clinical features only
- Model: Logistic Regression with balanced class weights
- Status: Baseline implementation

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
