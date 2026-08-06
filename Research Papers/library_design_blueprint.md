# 📦 DP-FPS Framework: Python Library Packaging Blueprint (`fed-fps`)

This document outlines the real-world applications of the **DP-FedProx-Shapley (DP-FPS)** framework and provides a concrete technical blueprint to package, build, and publish the codebase as a reusable Python library on PyPI.

---

## 1. Where This Framework Can Be Used (Real-World Applications)

The DP-FPS framework is designed for **collaborative machine learning under privacy, data heterogeneity, and participation incentive constraints**. 

### 🏥 Clinical Healthcare Consortia (Primary Domain)
*   **Oncology & Genomics**: Multiple hospitals (e.g., TCGA, MSKCC, Mayo Clinic) want to train cancer survival models (Cox) or staging classifiers on patient genomics and clinical tables without transferring raw patient data due to HIPAA and GDPR.
*   **Rare Disease Diagnosis**: Because individual clinics have very few rare disease samples, collaboration is required. DP-FPS lets clinics combine features while protecting patient identities.
*   **EHDS Clinical Data Marketplaces**: Under the *European Health Data Space (EHDS)* framework, hospital networks can use Shapley Values to legally buy, lease, or get compensated for clinical model participation.

### 🏦 Financial Institutions & Anti-Money Laundering (AML)
*   **Fraud Detection networks**: Banks want to collaborate to detect credit card fraud or transaction-level money laundering. Since sharing customer bank statements is strictly regulated, they train federated classifiers.
*   **Credit Risk Assessment**: Fintechs and community banks collaborate to model credit scoring without leaking proprietary client portfolios.

### 🚗 Intelligent Transport Systems & IoT
*   **Smart City Sensors**: Autonomous vehicles or smart grid substations collaborate to model traffic congestion or energy surges. FedProx mitigates local sensor drift (device heterogeneity), and virtual latency bounds help adapt to bandwidth delays.

---

## 2. Standalone Python Library Architecture (`fed-fps`)

To package this framework into a library that developers can install with `pip install fed-fps`, organize the repository as follows:

```text
fed-fps/
├── README.md              # Installation and usage instructions
├── LICENSE                # MIT or Apache 2.0 license
├── pyproject.toml         # Build system configuration (PEP 517)
├── requirements.txt       # Dependency versions
├── fed_fps/               # Package source root
│   ├── __init__.py        # Library entry point (exposes APIs)
│   ├── preprocessing.py   # Multi-modal data fusion, imputation, and PCA
│   ├── optimizers.py      # Vectorized custom NumPy loops (Logistic, Cox)
│   ├── federated.py       # FedAvg, FedProx, PFL training loops
│   ├── privacy.py         # DP gradient clipping, noise injection, RDP accountant
│   └── valuation.py       # shapley Value & Leave-One-Out contribution analysis
└── tests/                 # Unit test folder
    ├── __init__.py
    ├── test_optimizers.py
    └── test_federated.py
```

---

## 3. Package Configuration Files

### `pyproject.toml`
This file configures the build backend (e.g., `setuptools`) and defines package metadata.

```toml
[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fed-fps"
version = "1.0.0"
authors = [
    { name="Your Name", email="your.email@domain.com" }
]
description = "A privacy-preserving, heterogeneity-aware federated learning library with game-theoretic client contribution valuation."
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]
dependencies = [
    "numpy>=1.22.0",
    "pandas>=1.4.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.8.0",
    "matplotlib>=3.5.0",
    "websockets>=10.0"
]

[tool.setuptools.packages.find]
where = ["."]
include = ["fed_fps*"]
```

### `fed_fps/__init__.py`
Exposes the core functions to the user:

```python
from .preprocessing import load_clinical, load_survival_data, merge_clinical_survival
from .optimizers import compute_cox_likelihood_and_grad, local_cox_train
from .federated import partition_dirichlet, fedavg_train, fedprox_train, fedavg_cox_train, evaluate_personalized_fl, compute_rdp_privacy_budget
from .valuation import compute_federated_shapley_values

__version__ = "1.0.0"
__all__ = [
    "load_clinical",
    "load_survival_data",
    "merge_clinical_survival",
    "compute_cox_likelihood_and_grad",
    "local_cox_train",
    "partition_dirichlet",
    "fedavg_train",
    "fedprox_train",
    "fedavg_cox_train",
    "evaluate_personalized_fl",
    "compute_rdp_privacy_budget",
    "compute_federated_shapley_values"
]
```

---

## 4. Usage Example for End-Users

Once published, a developer can run a collaborative private survival analysis in under 15 lines of code:

```python
import numpy as np
import fed_fps as fps

# 1. Load clinical & survival records
df_clin = fps.load_clinical("hospitals_clinical.tsv")
df_surv = fps.load_survival_data("hospitals_survival.tsv")
merged = fps.merge_clinical_survival(df_clin, df_surv)

# 2. Extract features and target arrays
X = merged.drop(columns=['OS.time', 'OS', 'sample']).values
times = merged['OS.time'].values
events = merged['OS'].values

# 3. Simulate statistical heterogeneity (skew) across 3 hospitals
hospitals = fps.partition_dirichlet(X, y=events, num_hospitals=3, alpha=0.5)

# 4. Train private Federated Cox model under bandwidth constraints
results = fps.fedavg_cox_train(
    hospitals_surv=hospitals,  # (X, times, events) partitions
    X_test=X[:50], times_test=times[:50], events_test=events[:50],
    rounds=20, epochs=3, lr=0.01,
    dp_enabled=True, epsilon=1.5, delta=1e-5,
    dropout_rate=0.1, bandwidth_mbps=10.0, latency_ms=50.0
)

# 5. compute game-theoretic rewards for participation
shapley_scores = fps.compute_federated_shapley_values(
    hospitals, X_test=X[:50], y_test=events[:50]
)

print(f"Aggregated consensus parameters: {results['w_global']}")
print(f"Fair hospital payout distribution shares: {shapley_scores}")
```

---

## 5. How to Publish the Library to PyPI

Run these commands in your package root directory:

```bash
# 1. Install packaging tools
python -m pip install --upgrade build twine

# 2. Build distribution archives (source tarball and wheel file)
python -m build

# 3. Upload build files to TestPyPI (to verify uploading works)
python -m twine upload --repository testpypi dist/*

# 4. Upload to PyPI (Production Release)
python -m twine upload dist/*
```
Once uploaded, anyone in the world can run `pip install fed-fps` to load your framework!
