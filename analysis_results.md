# Project Analysis: Federated Learning for Prostate Cancer Risk Prediction

This document provides a comprehensive summary of:
1. The research paper: **"Hospital Participation in Federated Learning: Evaluating Sustainability and Clinical Utility"**
2. The project report: **"A Federated Learning–Inspired Web Application for Privacy-Preserving Prostate Cancer Risk Prediction"**
3. The codebase structure and mappings to their respective functional components.
4. The core mathematical algorithms implemented in the project.

---

## 1. Overview of the Research Paper

The research paper investigates the practical feasibility and sustainability of establishing a **Federated Learning (FL)** consortium in cross-hospital networks, specifically applied to **Prostate Cancer (PCa)** biopsy outcome prediction.

### Key Aspects
* **Clinical Task**: Predicting clinically significant prostate cancer (csPCa, defined as ISUP Grade Group $\ge$ 2) from pre-biopsy variables. This helps avoid unnecessary, risky biopsies.
* **Participating Sites**: Uses 19 real-world heterogeneous datasets across Europe, America, and Asia (a total of 5,610 patients) instead of artificial data partitioning, providing a realistic simulation of hospital variability.
* **Evaluated Strategies**:
  * **Local Training (LOC)**: Training models exclusively on each hospital's own local dataset.
  * **Federated Learning (FL)**: Collaboratively training a global model using the Federated Averaging (FedAvg) algorithm.
  * **Free-Riding (FR)**: Using a pre-trained federated model without contributing any local data.
  * **Baseline (BL)**: Utilizing pre-existing, independently developed prostate cancer risk calculators (Ettala & Noh models).
  * **Centralized Learning (CEN)**: Pooling all hospital datasets together (serves as the theoretical upper bound).

### Key Insights & Findings
1. **Diminishing Returns for Large Sites**: Large hospitals with abundant patient data gain very little from participating in FL. Their local models (LOC) perform comparably to or occasionally better than federated models.
2. **Impracticality for Small Sites**: Very small hospitals contribute negligible information to the global model, making their overhead of active participation hard to justify.
3. **The Power of Free-Riding (FR)**: As the federation size grows, the performance of the model on excluded sites (free-riders) improves steadily. Once $\approx 10$ hospitals join the consortium, the global model performs just as well on external/non-contributing datasets as it does on participating datasets.
4. **Proposed Sustainable Business Model**: Instead of complex active training loops where every hospital participates, a stable consortium of a few large, high-quality, high-data-volume hospitals should train and maintain a global model. Other smaller hospitals can access it via a "Model-as-a-Service" approach (reputable free-riding) in exchange for financial compensation, aligning with frameworks like the European Health Data Space (EHDS).

---

## 2. Overview of the Project Documentation

The project document describes the development and evaluation of a B. Tech academic project implementing **"A Federated Learning–Inspired Web Application for Privacy-Preserving Prostate Cancer Risk Prediction"**.

### Project Characteristics
* **Clinical Task**: Classifying prostate cancer staging into **Early Stage (T1/T2)** or **Advanced Stage (T3/T4)**.
* **Dataset**: Tabular clinical features obtained from **The Cancer Genome Atlas Prostate Adenocarcinoma (TCGA-PRAD)** database.
* **Target Feature**: `ajcc_pathologic_t.diagnoses` mapped to binary target (T1/T2 $\to$ 0, T3/T4 $\to$ 1).
* **Architecture**: A modular 5-version system implemented as an interactive Streamlit web dashboard.

### The Five Experimental Versions
* **Version 1: Centralized Baseline**: Baseline model trained using standard scikit-learn Logistic Regression on the pooled dataset. Establishes the performance upper bound (AUC $\approx 0.913$, Accuracy $\approx 86.5\%$).
* **Version 2: Federated Averaging (FedAvg)**: Simulates a multi-hospital consortium (default: 5) using a custom NumPy-based Logistic Regression model. Parameters (weights and bias) are collected and aggregated iteratively. Achieves performance near the centralized model (AUC $\approx 0.897$).
* **Version 3: Scalability and Free-Rider Analysis**: Evaluates performance as the number of hospitals scales ($2$ to $20$) and tests the impact of "malicious/unhelpful" free-riders (simulated via hospitals contributing randomly shuffled labels).
* **Version 4: FedProx & Non-IID Study**: Analyzes model performance and convergence stability under heterogeneous data conditions. Employs Dirichlet partitioning to create custom label skews (non-IID) and contrasts FedAvg with **FedProx** (which adds a proximal regularization term to stabilize client updates).
* **Version 5: Hospital Contribution Analysis**: Employs a **leave-one-out (LOO)** methodology to measure each hospital's marginal contribution to the global AUC-ROC, allowing for the quantification of data utility and fair incentive schemes.

---

## 3. Codebase Architecture & File Mapping

The codebase is written in Python, using Streamlit for the front-end dashboard and custom NumPy/scikit-learn for model and FL logic. Below is a mapping of the files located under the project folder [D:/Mini_project_JP](file:///D:/Mini_project_JP).

```
D:\Mini_project_JP\
├── app.py                     # Main Streamlit web application entrypoint
├── requirements.txt           # Project library dependencies
├── src\
│   ├── ui_components.py       # Custom styled Streamlit UI cards, badges, and components
│   ├── preprocessing.py       # Data loading, target mapping, scaling, and categorical encoding
│   ├── model.py               # Wrapper for standard Scikit-learn Logistic Regression (V1)
│   ├── evaluation.py          # AUC-ROC, confusion matrix computing, and plotting utilities
│   ├── logistic_numpy.py      # Manual NumPy implementation of Logistic Regression (Sigmoid, CE Loss, GD)
│   ├── federated.py           # Data partitioners (Equal, Imbalanced, Dirichlet) and FL training engines
│   ├── sustainability.py      # Version 3 logic: learning curves, free-rider tests, and partition comparisons
│   ├── fedprox_experiments.py # Version 4 logic: FedAvg vs FedProx convergence comparisons
│   └── contribution.py        # Version 5 logic: Leave-one-out contribution analysis
```

### Detailed Component Mapping

| File | Purpose | Key Functions/Classes |
| :--- | :--- | :--- |
| **[app.py](file:///D:/Mini_project_JP/app.py)** | Streamlit main entry point. Coordinates dataset loading, page selection, parameters, visual updates, and invokes the experiment scripts. | `main()`, `plot_fedavg_convergence()` |
| **[src/ui_components.py](file:///D:/Mini_project_JP/src/ui_components.py)** | Provides custom Tailwind/CSS styling, modern cards, and interactive version selectors to replace default Streamlit styling. | `render_header()`, `render_card()`, `render_version_selector()`, `apply_custom_css()` |
| **[src/preprocessing.py](file:///D:/Mini_project_JP/src/preprocessing.py)** | Preprocesses the TCGA-PRAD clinical files. Deals with missing values (median imputation) and maps pathologic stage diagnoses into classes. Merges protein data and applies PCA/selection. | `load_clinical()`, `create_target()`, `preprocess_features()`, `apply_pca()` |
| **[src/model.py](file:///D:/Mini_project_JP/src/model.py)** | Wrapper for training standard centralized models via scikit-learn. | `train_model()`, `predict_model()` |
| **[src/evaluation.py](file:///D:/Mini_project_JP/src/evaluation.py)** | Measures model accuracy, AUC, sensitivity, specificity, and draws ROC/Confusion Matrix charts. | `evaluate_model()`, `plot_roc_curve()`, `plot_confusion_matrix()` |
| **[src/logistic_numpy.py](file:///D:/Mini_project_JP/src/logistic_numpy.py)** | Custom NumPy implementation of Logistic Regression. Since we cannot modify/save weights mid-training in scikit-learn `.fit()`, this manual script implements Sigmoid and raw Gradient updates. | `sigmoid()`, `initialize_weights()`, `compute_loss()`, `compute_gradient()`, `local_train()` |
| **[src/federated.py](file:///D:/Mini_project_JP/src/federated.py)** | Houses data partitioning logic (equally, imbalanced long-tails, or Dirichlet non-IID skews) and the main FedAvg/FedProx training loops. | `partition_equal()`, `partition_dirichlet()`, `fedavg_train()`, `local_train_fedprox()`, `fedprox_train()` |
| **[src/sustainability.py](file:///D:/Mini_project_JP/src/sustainability.py)** | Handles Version 3 sustainability runs, Monte Carlo iterations for scalability/free-riders, and runs paired t-tests comparing equal vs. imbalanced partitioning. | `run_learning_curve()`, `run_free_rider_experiment()`, `compare_partitions()`, `plot_partition_comparison()` |
| **[src/fedprox_experiments.py](file:///D:/Mini_project_JP/src/fedprox_experiments.py)** | Coordinates Version 4 comparisons between FedAvg and FedProx under non-IID conditions. Evaluates weight drift and convergence. | `run_fedavg_vs_fedprox_experiment()`, `plot_convergence_curves()`, `plot_stability_comparison()` |
| **[src/contribution.py](file:///D:/Mini_project_JP/src/contribution.py)** | Handles Version 5 leave-one-out analysis. Sequentially trains the global model with hospital $k$ excluded and computes its contribution score. | `measure_hospital_contribution()`, `plot_contribution_analysis()` |
| **[src/experiment_manager.py](file:///D:/Mini_project_JP/src/experiment_manager.py)** | Standardizes reproducibility by tracking random seeds, hashing configs, and saving outputs in JSON format. | `ExperimentManager`, `set_global_seed()` |

---

## 4. Mathematical Details of the Algorithms

The custom ML models in this project use manual gradient descent for Logistic Regression.

### Logistic Regression
For sample $i$ with features $x_i \in \mathbb{R}^d$ and weight vector $w \in \mathbb{R}^d$, the predicted probability is:
$$\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-x_i^T w}}$$
The Binary Cross-Entropy (BCE) Loss is:
$$\mathcal{L}_{BCE}(w) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1 - \hat{y}_i) \right]$$
The gradient update is computed as:
$$\nabla_w \mathcal{L}_{BCE}(w) = \frac{1}{N} X^T (\hat{y} - y)$$

### Federated Averaging (FedAvg)
At each round $t$, a central server distributes global weights $w_t^{global}$ to all participating clients $k \in \{1,\dots,K\}$.
Each client trains locally for $E$ epochs using its private data $D_k$ (size $n_k$), obtaining updated local weights $w_{t+1}^k$.
The central server aggregates these parameters as a weighted average:
$$w_{t+1}^{global} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$
where $N = \sum_{k=1}^K n_k$.

### FedProx (Proximal Regularization)
To handle non-IID data where clients have divergent distributions, FedProx adds a proximal regularization term to the local objective function:
$$\min_{w} \mathcal{L}_{k}(w) = \mathcal{L}_{BCE}(w) + \frac{\mu}{2} \| w - w_t^{global} \|_2^2$$
This restricts the local update from drifting too far from the global parameters. The corresponding gradient used in local updates is:
$$\nabla_w \mathcal{L}_k(w) = \nabla_w \mathcal{L}_{BCE}(w) + \mu(w - w_t^{global})$$

### Leave-One-Out (LOO) Contribution
To calculate the marginal utility of hospital $k$ in the federation:
$$Cont(k) = AUC(D) - AUC(D \setminus D_k)$$
Where $AUC(D)$ is the global model test AUC when trained on all clients, and $AUC(D \setminus D_k)$ is the global model test AUC when trained without client $k$.

---

## 5. Synthesis: Connection Between the Research Paper and the Codebase

The project structure directly maps the concepts explored in the research paper into a simulated research dashboard:
* **The Core Thesis**: The paper's question—*is FL sustainable and what is the incentive for hospital participation?*—is translated in the app through **Version 3 (Scalability and Free-Rider Analysis)** and **Version 5 (Hospital Contribution Analysis)**.
* **Learning Curves**: The paper's methodology of plotting AUC curves as hospital counts increase (analyzing both participant FL curves and holdout/free-rider FR curves) is implemented in `sustainability.py` using Monte Carlo trials.
* **Baseline Comparison**: The paper uses risk calculators (Ettala & Noh) as baseline comparisons. In the project code, this is mimicked by comparing the federated global model against Version 1's centralized model and individual local models (LOC).
* **Data Heterogeneity**: The paper highlights radiologist scoring (PI-RADS) inconsistencies as a major hurdle. In the project, this challenge is tackled in **Version 4** using Dirichlet partitioning to simulate class imbalance and verifying if **FedProx** can mitigate client drift.
