"""
Cohort construction and leakage-free preprocessing for TCGA-PRAD.

Fixes relative to the original src/preprocessing.py
-------------------------------------------------
1. Features are chosen from an explicit WHITELIST. The original pipeline used
   "every column except a few exact ID names", which let ~1,650 one-hot columns
   built from per-patient identifiers (submitter_id.samples, sample_id.samples,
   pathology_report_uuid.samples, treatment ids, created/updated datetimes)
   into the model, plus follow-up/outcome columns (vital_status,
   days_to_last_follow_up).
2. One primary-tumour sample (-01) per patient, so the privacy unit
   (sample) coincides with the patient.
3. Two clinically distinct feature sets:
     PREOP - information available before surgery
     FULL  - PREOP + post-prostatectomy pathology (Gleason pattern, pathologic N).
            These come from the same pathology report as the pathologic T label,
            so they are strong but "label-adjacent"; this is disclosed in the app.
4. All imputation / scaling / encoding / PCA is fitted on training rows only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets")
CLINICAL_TSV = os.path.join(DATA_DIR, "TCGA-PRAD.clinical.tsv", "TCGA-PRAD.clinical.tsv")
PROTEIN_TSV = os.path.join(DATA_DIR, "TCGA-PRAD.protein.tsv", "TCGA-PRAD.protein.tsv")
SURVIVAL_TSV = os.path.join(DATA_DIR, "TCGA-PRAD.survival.tsv", "TCGA-PRAD.survival.tsv")

TARGET_COL = "ajcc_pathologic_t.diagnoses"
SITE_COL = "name.tissue_source_site"

# ----------------------------------------------------------------------------
# Feature whitelist
# ----------------------------------------------------------------------------
PREOP_NUMERIC = ["age"]
PREOP_CATEGORICAL = ["race", "ethnicity", "clinical_t", "clinical_m", "prior_malignancy"]
PATHOLOGY_NUMERIC = ["gleason_primary", "gleason_secondary", "gleason_sum"]
PATHOLOGY_CATEGORICAL = ["pathologic_n"]

FEATURE_SETS = {
    "full": (PREOP_NUMERIC + PATHOLOGY_NUMERIC, PREOP_CATEGORICAL + PATHOLOGY_CATEGORICAL),
    "preop": (PREOP_NUMERIC, PREOP_CATEGORICAL),
}

# Columns that must never be features (documented, and asserted in tests)
EXCLUDED_REASONS = {
    "identifiers": ["sample", "id", "case_id", "submitter_id", "sample_id.samples",
                    "pathology_report_uuid.samples", "treatment_id.treatments.diagnoses",
                    "submitter_id.treatments.diagnoses", "tissue_source_site_id.tissue_source_site"],
    "record timestamps": ["created_datetime.treatments.diagnoses", "updated_datetime.treatments.diagnoses"],
    "outcome / follow-up (not known at staging time)": ["vital_status.demographic",
                                                        "days_to_last_follow_up.diagnoses",
                                                        "treatment_type.treatments.diagnoses",
                                                        "treatment_or_therapy.treatments.diagnoses"],
    "target": [TARGET_COL],
    "hospital identity (used to define clients, not as a feature)": [SITE_COL, "code.tissue_source_site"],
}


def _gleason(v):
    if isinstance(v, str) and v.startswith("Pattern"):
        try:
            return float(v.split()[-1])
        except ValueError:
            return np.nan
    return np.nan


def _clin_t(v):
    if not isinstance(v, str):
        return "Unknown"
    for g in ("T1", "T2", "T3", "T4"):
        if v.startswith(g):
            return g
    return "Unknown"


def _race(v):
    if v in ("white", "black or african american", "asian"):
        return v
    if v == "not reported" or not isinstance(v, str):
        return "not reported"
    return "other"


def load_clinical_cohort() -> pd.DataFrame:
    """Return one row per patient (primary tumour sample, valid pathologic T stage)
    with tidy engineered columns, the binary target `y` and the hospital `site`."""
    raw = pd.read_csv(CLINICAL_TSV, sep="\t")
    df = raw[raw["sample"].astype(str).str[13:15] == "01"].copy()       # primary tumour only
    df = df[df[TARGET_COL].notna()]
    df = df.sort_values("sample").drop_duplicates("submitter_id", keep="first")  # 1 sample / patient

    out = pd.DataFrame({
        "sample": df["sample"].values,
        "patient": df["submitter_id"].values,
        "site": df[SITE_COL].fillna("Unknown").values,
        "t_stage": df[TARGET_COL].values,
        "age": pd.to_numeric(df["age_at_index.demographic"], errors="coerce").values,
        "race": [ _race(v) for v in df["race.demographic"] ],
        "ethnicity": df["ethnicity.demographic"].fillna("not reported").astype(str).values,
        "clinical_t": [_clin_t(v) for v in df["ajcc_clinical_t.diagnoses"]],
        "clinical_m": [("M1" if str(v).startswith("M1") else "M0" if str(v) == "M0" else "Unknown")
                       for v in df["ajcc_clinical_m.diagnoses"]],
        "prior_malignancy": df["prior_malignancy.diagnoses"].fillna("unknown").astype(str).values,
        "gleason_primary": [_gleason(v) for v in df["primary_gleason_grade.diagnoses"]],
        "gleason_secondary": [_gleason(v) for v in df["secondary_gleason_grade.diagnoses"]],
        "pathologic_n": [v if v in ("N0", "N1") else "Unknown" for v in df["ajcc_pathologic_n.diagnoses"]],
    })
    out["gleason_sum"] = out["gleason_primary"] + out["gleason_secondary"]
    out["y"] = out["t_stage"].astype(str).str.match(r"T[34]").astype(int)
    return out.reset_index(drop=True)


def load_protein_matrix() -> pd.DataFrame:
    p = pd.read_csv(PROTEIN_TSV, sep="\t").set_index("peptide_target").T
    p.index.name = "sample"
    return p.apply(pd.to_numeric, errors="coerce")


def load_matched_cohort():
    """Clinical cohort restricted to patients with RPPA protein data (the paper's cohort)."""
    clin = load_clinical_cohort()
    prot = load_protein_matrix()
    clin = clin[clin["sample"].isin(prot.index)].reset_index(drop=True)
    return clin, prot.loc[clin["sample"]].reset_index(drop=True)


def survival_events(samples) -> pd.DataFrame:
    s = pd.read_csv(SURVIVAL_TSV, sep="\t")
    return s[s["sample"].isin(set(samples))]


# ----------------------------------------------------------------------------
# Train-only fitted preprocessing
# ----------------------------------------------------------------------------
@dataclass
class TabularPreprocessor:
    feature_set: str = "full"
    numeric: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    medians_: dict = field(default_factory=dict)
    categories_: dict = field(default_factory=dict)
    scaler_: StandardScaler | None = None
    feature_names_: list = field(default_factory=list)

    def __post_init__(self):
        self.numeric, self.categorical = FEATURE_SETS[self.feature_set]

    def fit(self, df: pd.DataFrame):
        self.medians_ = {c: float(df[c].median()) if df[c].notna().any() else 0.0 for c in self.numeric}
        # categories seen in TRAIN only; first category dropped (reference level)
        self.categories_ = {c: sorted(df[c].astype(str).unique().tolist()) for c in self.categorical}
        num = self._num(df)
        self.scaler_ = StandardScaler().fit(num)
        self.feature_names_ = list(self.numeric) + [f"{c}={v}" for c in self.categorical
                                                     for v in self.categories_[c][1:]]
        return self

    def _num(self, df):
        return np.column_stack([df[c].fillna(self.medians_[c]).astype(float).values for c in self.numeric])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = [self.scaler_.transform(self._num(df))]
        for c in self.categorical:
            vals = df[c].astype(str).values
            cats = self.categories_[c][1:]
            parts.append(np.column_stack([(vals == v).astype(float) for v in cats]) if cats
                         else np.zeros((len(df), 0)))
        return np.hstack(parts)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


@dataclass
class ProteinPreprocessor:
    missing_threshold: float = 0.30
    variance: float = 0.95
    keep_: list = field(default_factory=list)
    medians_: pd.Series | None = None
    scaler_: StandardScaler | None = None
    pca_: PCA | None = None

    def fit(self, P: pd.DataFrame):
        miss = P.isna().mean()
        self.keep_ = miss[miss <= self.missing_threshold].index.tolist()
        X = P[self.keep_]
        self.medians_ = X.median()
        X = X.fillna(self.medians_)
        self.scaler_ = StandardScaler().fit(X.values)
        self.pca_ = PCA(n_components=self.variance, svd_solver="full").fit(self.scaler_.transform(X.values))
        return self

    def transform(self, P: pd.DataFrame) -> np.ndarray:
        X = P[self.keep_].fillna(self.medians_)
        return self.pca_.transform(self.scaler_.transform(X.values))

    @property
    def n_proteins(self):
        return len(self.keep_)

    @property
    def n_components(self):
        return int(self.pca_.n_components_)


def build_features(train_df, test_df, feature_set="full", P_train=None, P_test=None):
    """Fit on train, transform both. If protein frames are given, append protein PCs
    (feature_set may be 'protein' to use proteins only)."""
    blocks_tr, blocks_te, names = [], [], []
    if feature_set in ("full", "preop"):
        tp = TabularPreprocessor(feature_set).fit(train_df)
        blocks_tr.append(tp.transform(train_df)); blocks_te.append(tp.transform(test_df))
        names += tp.feature_names_
    if P_train is not None:
        pp = ProteinPreprocessor().fit(P_train)
        blocks_tr.append(pp.transform(P_train)); blocks_te.append(pp.transform(P_test))
        names += [f"PC{i+1}" for i in range(pp.n_components)]
    return np.hstack(blocks_tr), np.hstack(blocks_te), names
