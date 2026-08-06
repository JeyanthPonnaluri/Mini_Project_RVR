# 📦 FederatedHPL: Federated Hyper-parameterized Learning Library Blueprint

This document outlines the package layout and PyPI publishing workflow for **FederatedHPL** (Federated Hyper-parameterized Learning), a library implementing privacy-preserving, heterogeneity-aware federated optimization (FedProx, PFL) with game-theoretic client contribution valuation (Shapley Value).

---

## 1. Directory Layout of the Library (`FederatedHPL`)

To package this project as a reusable Python library, establish a separate repository containing the following folder structure:

```text
FederatedHPL/
├── pyproject.toml         # Modern build configuration (PEP 517)
├── README.md              # Installation and quick-start instructions
├── LICENSE                # Open source license (e.g. MIT, Apache 2.0)
├── requirements.txt       # Dependency versions
├── hpf_learning/          # Root package directory
│   ├── __init__.py        # Exposes main API modules & functions
│   ├── preprocessing.py   # Multi-modal fusion, imputation, cohort generators
│   ├── optimizers.py      # Custom vectorized NumPy optimizers (Logistic, Cox)
│   ├── federated.py       # FedAvg, FedProx, PFL local fine-tuning
│   ├── privacy.py         # DP noise, clipping, Rényi DP budget accountants
│   └── valuation.py       # Shapley Value & Leave-One-Out contributors
└── tests/                 # Unit tests
    ├── __init__.py
    ├── test_optimizers.py
    └── test_federated.py
```

---

## 2. Package Configuration Files

### `pyproject.toml`
This file declares packaging metadata and specifies build requirements.

```toml
[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "FederatedHPL"
version = "1.0.0"
authors = [
    { name="Your Name", email="your.email@domain.com" }
]
description = "FederatedHPL: Federated Hyper-parameterized Learning with differential privacy and game-theoretic Shapley valuations."
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
include = ["hpf_learning*"]
```

### `hpf_learning/__init__.py`
Exposes package functions for direct imports:

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

## 3. End-User Usage Example

A developer can import and execute the code as follows:

```python
import numpy as np
import hpf_learning as hpf

# 1. Prepare target structures
df_clin = hpf.load_clinical("clinical.tsv")
df_surv = hpf.load_survival_data("survival.tsv")
merged = hpf.merge_clinical_survival(df_clin, df_surv)

X = merged.drop(columns=['OS.time', 'OS', 'sample']).values
times = merged['OS.time'].values
events = merged['OS'].values

# 2. Partition clinical records
hospitals = hpf.partition_dirichlet(X, y=events, num_hospitals=3, alpha=0.5)

# 3. Train private Cox Proportional Hazards model
results = hpf.fedavg_cox_train(
    hospitals_surv=hospitals,
    X_test=X[:50], times_test=times[:50], events_test=events[:50],
    rounds=20, epochs=3, lr=0.01,
    dp_enabled=True, epsilon=1.5, delta=1e-5
)

print(f"Global weights: {results['w_global']}")
```

---

## 4. Step-by-Step Publishing Guide for PyPI

Follow this protocol to host your package on PyPI so others can run `pip install HPFLearning`:

### Step A: Create Accounts
1. **PyPI Account**: Register at [pypi.org/register](https://pypi.org/register/).
2. **TestPyPI Account** (Highly Recommended for testing): Register at [test.pypi.org/register](https://test.pypi.org/register/).

### Step B: Generate an API Token
PyPI requires authentication via API tokens for security (passwords are not accepted during upload):
1. Log in to [pypi.org](https://pypi.org/).
2. Go to **Account Settings** -> **API Tokens** -> **Add API Token**.
3. Set the scope to "Entire Account" (or a specific project once uploaded).
4. Copy the token. It will look like `pypi-AgEIcHlwaS5vcmc...`.

### Step C: Build the Distribution Package
In the project root folder (containing `pyproject.toml`), run:
```bash
# Upgrade build package
python -m pip install --upgrade build

# Compile source distribution and wheel files
python -m build
```
This generates a `dist/` directory containing two archives:
* `HPFLearning-1.0.0.tar.gz` (Source Tarball)
* `HPFLearning-1.0.0-py3-none-any.whl` (Built Wheel)

### Step D: Upload to PyPI
1. **Install Twine**:
   ```bash
   python -m pip install --upgrade twine
   ```
2. **Upload to TestPyPI** first to make sure there are no errors:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```
   * *Username*: Enter `__token__` (literally).
   * *Password*: Paste your TestPyPI API token (including the `pypi-` prefix).
3. **Verify Test Installation**:
   Verify you can download it onto a clean virtual environment:
   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ FederatedHPL
   ```
4. **Upload to Live PyPI (Production Release)**:
   ```bash
   python -m twine upload dist/*
   ```
   * *Username*: Enter `__token__`
   * *Password*: Paste your production PyPI API token.

Once completed, anyone can install your framework globally via:
```bash
pip install FederatedHPL
```
