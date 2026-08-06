# FederatedHPLearning 📦 (Federated Hyper-parameterized Learning)

`FederatedHPLearning` is a secure, lightweight, and mathematically rigorous Python library for **collaborative machine learning under privacy, data heterogeneity, and participation incentive constraints**.

It provides vectorized implementation of **Federated Cox Proportional Hazards**, **FedProx**, **Personalized Federated Learning (PFL)**, **Rényi Differential Privacy (RDP)** accounting, and **Game-Theoretic Shapley Value** contribution evaluations.

---

## 🚀 Installation

```bash
pip install FederatedHPLearning
```

---

## 🎯 Quick Start Example

Here is how you can train a private federated Cox model and evaluate patient contribution credits in under 15 lines of code:

```python
import numpy as np
import federated_hplearning as hpf

# 1. Load clinical & survival tables
df_clin = hpf.load_clinical("clinical.tsv")
df_surv = hpf.load_survival_data("survival.tsv")
merged = hpf.merge_clinical_survival(df_clin, df_surv)

X = merged.drop(columns=['OS.time', 'OS', 'sample']).values
times = merged['OS.time'].values
events = merged['OS'].values

# 2. Simulate hospital partitions using Dirichlet distribution skew
hospitals = hpf.partition_dirichlet(X, y=events, num_hospitals=3, alpha=0.5)

# 3. Train private federated Cox model
results = hpf.fedavg_cox_train(
    hospitals_surv=hospitals,
    X_test=X[:50], times_test=times[:50], events_test=events[:50],
    rounds=20, epochs=3, lr=0.01,
    dp_enabled=True, epsilon=1.5, delta=1e-5
)

# 4. Compute Shapley values to allocate incentives/credits
shapley_scores = hpf.compute_federated_shapley_values(
    hospitals, X_test=X[:50], y_test=events[:50]
)

print(f"Aggregated weights: {results['w_global']}")
print(f"Hospital Shapley values: {shapley_scores}")
```

---

## 🔒 License
MIT License. Feel free to use and publish!
