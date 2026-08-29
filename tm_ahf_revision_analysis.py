"""
TM-AHF revised analysis pipeline
================================
Purpose
-------
Leakage-safe reproduction and strengthening of the manuscript:
"TM-AHF: A Text Mining and Vital Signs Ensemble Model for BNP/NT-proBNP-Validated
Acute Heart Failure Risk Stratification"

Main additions:
1) strict train-only preprocessing and feature selection
2) 10-model benchmark on combined/text/structured inputs
3) 95% bootstrap CIs
4) paired DeLong tests for AUROC
5) AUPRC and precision-recall curves
6) calibration curve, Brier score, calibration intercept/slope
7) decision-curve analysis (DCA)
8) two-centre external validation in both directions
9) SHAP global/local interpretation
10) reproducible baseline table and cohort checks

IMPORTANT
---------
Edit CONFIG below to match the actual CSV column names before running.
Do not use the generated example names without checking the source dataset.
"""

from __future__ import annotations

import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit
from scipy.stats import norm

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Optional libraries
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise ImportError("Install lightgbm: pip install lightgbm") from e

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("Install xgboost: pip install xgboost") from e

try:
    import shap
except Exception as e:
    raise ImportError("Install shap: pip install shap") from e

try:
    import statsmodels.api as sm
except Exception as e:
    raise ImportError("Install statsmodels: pip install statsmodels") from e


# ============================================================
# CONFIGURATION: EDIT THESE NAMES TO MATCH THE REAL DATA
# ============================================================

@dataclass
class Config:
    data_path: str = "data/tm_ahf.csv"
    output_dir: str = "outputs"

    # identifiers
    patient_id: str = "patient_id"
    centre_col: str = "hospital_id"

    # target: if it already exists, use target_col.
    # If derive_target_from_biomarkers=True, the target will be regenerated.
    target_col: str = "ahf_label"
    derive_target_from_biomarkers: bool = False
    bnp_col: str = "BNP"
    ntprobnp_col: str = "NT_proBNP"
    age_col: str = "age"

    # raw structured variables
    pulse_col: str = "pulse"
    sbp_col: str = "systolic_pressure"
    dbp_col: str = "diastolic_pressure"
    weight_col: str = "weight"
    height_col: str = "height"

    # optional baseline categorical variables
    sex_col: Optional[str] = "male"
    smoking_col: Optional[str] = "smoking"
    alcohol_col: Optional[str] = "alcohol"

    # free-text fields. They are concatenated row-wise.
    text_cols: Tuple[str, ...] = (
        "chief_complaint",
        "present_medical_history",
        "past_medical_history",
    )

    # modelling
    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 5
    max_tfidf_features: int = 5000
    selected_text_features: int = 38
    min_df: int = 2
    n_jobs: int = -1
    scoring: str = "roc_auc"

    # bootstrap
    bootstrap_iterations: int = 2000

    # threshold used for ordinary binary metrics
    classification_threshold: float = 0.50

    # DCA
    dca_min_threshold: float = 0.01
    dca_max_threshold: float = 0.80
    dca_points: int = 80


CFG = Config()


# ============================================================
# REPRODUCIBILITY / I/O
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    def converter(x):
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=converter)


# ============================================================
# TARGET DEFINITION
# ============================================================

def ntprobnp_threshold(age: float) -> float:
    """Age-stratified NT-proBNP threshold used in the manuscript."""
    if pd.isna(age):
        return np.nan
    if age < 50:
        return 450.0
    if age <= 75:
        return 900.0
    return 1800.0


def derive_ahf_label(
    df: pd.DataFrame,
    age_col: str,
    bnp_col: str,
    ntprobnp_col: str,
) -> pd.Series:
    """
    Positive if BNP >=300 ng/L OR NT-proBNP exceeds the age-specific threshold.
    Negative if at least one biomarker is available and none of the available
    measurements crosses its threshold.
    Missing if neither biomarker is available.

    Review this rule against the original clinical protocol before use.
    """
    bnp = pd.to_numeric(df[bnp_col], errors="coerce")
    ntp = pd.to_numeric(df[ntprobnp_col], errors="coerce")
    age = pd.to_numeric(df[age_col], errors="coerce")

    nt_thr = age.map(ntprobnp_threshold)

    positive = (bnp >= 300) | (ntp >= nt_thr)
    has_any_test = bnp.notna() | ntp.notna()

    y = pd.Series(np.nan, index=df.index, dtype="float")
    y.loc[has_any_test] = 0
    y.loc[positive & has_any_test] = 1
    return y


# ============================================================
# DATA PREPARATION
# ============================================================

def clean_binary(series: pd.Series) -> pd.Series:
    """Convert common binary encodings to 0/1 where possible."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, "male": 1, "m": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "female": 0, "f": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def prepare_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    required = [
        cfg.patient_id, cfg.centre_col, cfg.age_col, cfg.pulse_col,
        cfg.sbp_col, cfg.dbp_col, cfg.weight_col, cfg.height_col,
        *cfg.text_cols
    ]
    if not cfg.derive_target_from_biomarkers:
        required.append(cfg.target_col)
    else:
        required.extend([cfg.bnp_col, cfg.ntprobnp_col])

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns. Edit Config to match your dataset:\n"
            + "\n".join(missing)
        )

    out = df.copy()

    # One row per patient is assumed. Flag duplicates explicitly.
    dup = out[cfg.patient_id].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"{dup.sum()} rows have duplicated patient IDs. "
            "Resolve multiple admissions according to the study protocol before modelling."
        )

    if cfg.derive_target_from_biomarkers:
        out[cfg.target_col] = derive_ahf_label(
            out, cfg.age_col, cfg.bnp_col, cfg.ntprobnp_col
        )

    out[cfg.target_col] = pd.to_numeric(out[cfg.target_col], errors="coerce")
    out = out[out[cfg.target_col].isin([0, 1])].copy()
    out[cfg.target_col] = out[cfg.target_col].astype(int)

    # numeric structured variables
    for c in [cfg.age_col, cfg.pulse_col, cfg.sbp_col, cfg.dbp_col, cfg.weight_col, cfg.height_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Explicit missing-height indicator, retained as a predictor.
    out["missing_height_record"] = out[cfg.height_col].isna().astype(int)

    # Combine the free-text fields. Missing text becomes empty string.
    out["combined_text"] = (
        out[list(cfg.text_cols)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Optional baseline categorical fields
    for c in [cfg.sex_col, cfg.smoking_col, cfg.alcohol_col]:
        if c and c in out.columns:
            out[c] = clean_binary(out[c])

    return out


def cohort_checks(df: pd.DataFrame, cfg: Config) -> Dict:
    y = df[cfg.target_col]
    counts = y.value_counts().sort_index().to_dict()
    centres = df[cfg.centre_col].value_counts(dropna=False).to_dict()

    report = {
        "n_total": int(len(df)),
        "n_negative": int(counts.get(0, 0)),
        "n_positive": int(counts.get(1, 0)),
        "positive_prevalence": float(y.mean()),
        "centres": {str(k): int(v) for k, v in centres.items()},
        "duplicate_patient_ids": int(df[cfg.patient_id].duplicated().sum()),
        "empty_text_rows": int((df["combined_text"].str.len() == 0).sum()),
    }

    print("\nCOHORT CHECK")
    print(json.dumps(report, indent=2))
    return report


# ============================================================
# BASELINE TABLE
# ============================================================

def median_iqr(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def n_percent(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    n = int((x == 1).sum())
    return f"{n} ({100*n/len(x):.1f}%)"


def mann_whitney_p(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue


def categorical_p(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    table = np.array([
        [(a == 1).sum(), (a == 0).sum()],
        [(b == 1).sum(), (b == 0).sum()],
    ])
    if table.min() < 5:
        return stats.fisher_exact(table)[1]
    return stats.chi2_contingency(table, correction=False)[1]


def make_baseline_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    pos = df[df[cfg.target_col] == 1]
    neg = df[df[cfg.target_col] == 0]

    rows = []

    continuous = [
        ("Age (years)", cfg.age_col),
        ("Pulse rate (bpm)", cfg.pulse_col),
        ("Weight (kg)", cfg.weight_col),
        ("Systolic BP (mmHg)", cfg.sbp_col),
        ("Diastolic BP (mmHg)", cfg.dbp_col),
    ]
    for label, col in continuous:
        rows.append({
            "Characteristic": label,
            "AHF": median_iqr(pos[col]),
            "Others": median_iqr(neg[col]),
            "P_value": mann_whitney_p(pos[col], neg[col]),
            "Test": "Mann-Whitney U",
        })

    rows.append({
        "Characteristic": "Missing height record",
        "AHF": n_percent(pos["missing_height_record"]),
        "Others": n_percent(neg["missing_height_record"]),
        "P_value": categorical_p(pos["missing_height_record"], neg["missing_height_record"]),
        "Test": "Chi-square/Fisher",
    })

    for label, col in [
        ("Male", cfg.sex_col),
        ("Smoking", cfg.smoking_col),
        ("Alcohol consumption", cfg.alcohol_col),
    ]:
        if col and col in df.columns:
            rows.append({
                "Characteristic": label,
                "AHF": n_percent(pos[col]),
                "Others": n_percent(neg[col]),
                "P_value": categorical_p(pos[col], neg[col]),
                "Test": "Chi-square/Fisher",
            })

    return pd.DataFrame(rows)


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor(cfg: Config, modality: str) -> ColumnTransformer:
    """
    Text branch:
        TF-IDF(max 5000) -> chi-square SelectKBest(k=38)
    Structured branch:
        KNN imputation -> StandardScaler

    Selection is fit inside each training fold, preventing leakage.
    """
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=cfg.max_tfidf_features,
            min_df=cfg.min_df,
            ngram_range=(1, 1),
            lowercase=False,  # Chinese is unaffected; preserves English tokens as recorded
            sublinear_tf=True,
        )),
        ("select", SelectKBest(score_func=chi2, k=cfg.selected_text_features)),
    ])

    structured_cols = [
        cfg.age_col,
        "missing_height_record",
        cfg.pulse_col,
        cfg.sbp_col,
        cfg.dbp_col,
        cfg.weight_col,
    ]

    structured_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler()),
    ])

    transformers = []
    if modality in ("text", "combined"):
        transformers.append(("text", text_pipe, "combined_text"))
    if modality in ("structured", "combined"):
        transformers.append(("structured", structured_pipe, structured_cols))

    if not transformers:
        raise ValueError("modality must be 'text', 'structured', or 'combined'")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrix to dense only for estimators that require it."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# ============================================================
# MODELS AND SEARCH GRIDS
# ============================================================

def get_models_and_grids(cfg: Config):
    rs = cfg.random_state

    lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=rs)
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=rs)
    rf = RandomForestClassifier(
        class_weight="balanced", random_state=rs, n_jobs=cfg.n_jobs
    )
    ada = AdaBoostClassifier(random_state=rs)
    gbt = GradientBoostingClassifier(random_state=rs)
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=rs,
        n_jobs=cfg.n_jobs,
        tree_method="hist",
    )
    lgbm = LGBMClassifier(
        objective="binary",
        random_state=rs,
        n_jobs=cfg.n_jobs,
        verbosity=-1,
    )

    # Base model for bagging
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=rs
        ),
        random_state=rs,
        n_jobs=cfg.n_jobs,
    )

    voting = VotingClassifier(
        estimators=[
            ("lr", clone(lr)),
            ("rf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=rs, n_jobs=cfg.n_jobs
            )),
            ("lgbm", LGBMClassifier(
                n_estimators=300, random_state=rs,
                n_jobs=cfg.n_jobs, verbosity=-1
            )),
        ],
        voting="soft",
        n_jobs=cfg.n_jobs,
    )

    # GradientBoosting requires dense input. Other pipelines remain sparse-friendly.
    models = {
        "Logistic Regression": (lr, {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear"],
        }),
        "KNN": (knn, {
            "model__n_neighbors": [5, 11, 21],
            "model__weights": ["uniform", "distance"],
        }),
        "Decision Tree": (dt, {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_leaf": [1, 5, 10],
        }),
        "Random Forest": (rf, {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.5],
        }),
        "AdaBoost": (ada, {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.03, 0.1, 0.5],
        }),
        "Gradient Boosting": (gbt, {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
        }),
        "XGBoost": (xgb, {
            "model__n_estimators": [300, 600],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }),
        "LightGBM": (lgbm, {
            "model__n_estimators": [300, 600],
            "model__learning_rate": [0.03, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 8, 12],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }),
        "Bagging": (bag, {
            "model__n_estimators": [100, 300],
            "model__max_samples": [0.7, 1.0],
            "model__max_features": [0.7, 1.0],
        }),
        "Voting": (voting, {
            "model__weights": [(1, 1, 1), (1, 1, 2), (1, 2, 2)],
        }),
    }
    return models


def build_pipeline(cfg: Config, modality: str, model_name: str, estimator) -> Pipeline:
    steps = [("preprocess", build_preprocessor(cfg, modality))]
    if model_name == "Gradient Boosting":
        steps.append(("to_dense", DenseTransformer()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ============================================================
# METRICS
# ============================================================

def specificity_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def metric_dict(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Recall_Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_true, y_prob),
    }


def bootstrap_metric_cis(
    y_true,
    y_prob,
    threshold=0.5,
    n_boot=2000,
    seed=42,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    store = {k: [] for k in metric_dict(y_true, y_prob, threshold)}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals = metric_dict(yt, yp, threshold)
        for k, v in vals.items():
            if np.isfinite(v):
                store[k].append(v)

    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in store.items()
        if len(v)
    }


def evaluate_predictions(
    y_true,
    y_prob,
    name: str,
    cfg: Config,
) -> Dict:
    point = metric_dict(y_true, y_prob, cfg.classification_threshold)
    ci = bootstrap_metric_cis(
        y_true,
        y_prob,
        threshold=cfg.classification_threshold,
        n_boot=cfg.bootstrap_iterations,
        seed=cfg.random_state,
    )
    return {"name": name, "point": point, "ci95": ci}


# ============================================================
# FAST DELONG TEST FOR CORRELATED ROC AUCs
# Based on the standard Sun & Xu fast DeLong formulation.
# ============================================================

def compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true, pred_one, pred_two) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred_one = np.asarray(pred_one, dtype=float)
    pred_two = np.asarray(pred_two, dtype=float)

    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("DeLong requires binary labels containing both 0 and 1.")

    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())

    preds = np.vstack((pred_one, pred_two))[:, order]
    aucs, cov = fast_delong(preds, label_1_count)

    l = np.array([[1, -1]])
    var = float(l @ cov @ l.T)
    if var <= 0:
        z = np.nan
        p = np.nan
    else:
        z = float(abs(aucs[0] - aucs[1]) / math.sqrt(var))
        p = float(2 * norm.sf(z))

    return {
        "auc_model_1": float(aucs[0]),
        "auc_model_2": float(aucs[1]),
        "auc_difference": float(aucs[0] - aucs[1]),
        "z": z,
        "p_value": p,
    }


# ============================================================
# CALIBRATION
# ============================================================

def calibration_intercept_slope(y_true, y_prob) -> Dict[str, float]:
    """
    Logistic recalibration:
        logit(P(Y=1)) = intercept + slope * logit(predicted_probability)

    Ideal: intercept=0, slope=1.
    """
    eps = 1e-6
    p = np.clip(np.asarray(y_prob), eps, 1 - eps)
    lp = np.log(p / (1 - p))
    X = sm.add_constant(lp)
    model = sm.GLM(np.asarray(y_true), X, family=sm.families.Binomial()).fit()

    return {
        "calibration_intercept": float(model.params[0]),
        "calibration_slope": float(model.params[1]),
        "intercept_se": float(model.bse[0]),
        "slope_se": float(model.bse[1]),
    }


def plot_calibration(y_true, prediction_map: Dict[str, np.ndarray], out_path: Path):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for name, prob in prediction_map.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, prob, n_bins=10, strategy="quantile"
        )
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event proportion")
    plt.title("Calibration curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# DECISION CURVE ANALYSIS
# ============================================================

def decision_curve(y_true, y_prob, thresholds: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob)
    n = len(y)
    prevalence = y.mean()

    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))

        odds = pt / (1 - pt)
        nb_model = tp / n - fp / n * odds
        nb_all = prevalence - (1 - prevalence) * odds
        nb_none = 0.0

        rows.append({
            "threshold": pt,
            "net_benefit_model": nb_model,
            "net_benefit_all": nb_all,
            "net_benefit_none": nb_none,
        })
    return pd.DataFrame(rows)


def plot_dca(
    y_true,
    prediction_map: Dict[str, np.ndarray],
    cfg: Config,
    out_path: Path,
):
    thresholds = np.linspace(
        cfg.dca_min_threshold,
        cfg.dca_max_threshold,
        cfg.dca_points
    )
    plt.figure(figsize=(7, 5))

    first = True
    for name, prob in prediction_map.items():
        d = decision_curve(y_true, prob, thresholds)
        plt.plot(d["threshold"], d["net_benefit_model"], label=name)
        if first:
            plt.plot(d["threshold"], d["net_benefit_all"], linestyle="--", label="Treat all")
            plt.plot(d["threshold"], d["net_benefit_none"], linestyle=":", label="Treat none")
            first = False

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# PLOTTING ROC / PR
# ============================================================

def plot_roc(y_true, prediction_map: Dict[str, np.ndarray], out_path: Path):
    plt.figure(figsize=(6, 6))
    for name, prob in prediction_map.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pr(y_true, prediction_map: Dict[str, np.ndarray], out_path: Path):
    plt.figure(figsize=(6, 6))
    prevalence = np.mean(y_true)
    for name, prob in prediction_map.items():
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.axhline(prevalence, linestyle="--", label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# GRID SEARCH / INTERNAL TEST
# ============================================================

def fit_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Config,
    modality: str,
    model_name: str,
    estimator,
    param_grid: Dict,
) -> GridSearchCV:
    cv = StratifiedKFold(
        n_splits=cfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )
    pipe = build_pipeline(cfg, modality, model_name, estimator)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=cfg.scoring,
        cv=cv,
        n_jobs=cfg.n_jobs,
        refit=True,
        return_train_score=False,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def benchmark_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    modality: str = "combined",
) -> Tuple[pd.DataFrame, Dict[str, GridSearchCV], Dict[str, np.ndarray]]:
    y_train = train_df[cfg.target_col]
    y_test = test_df[cfg.target_col]

    models = get_models_and_grids(cfg)
    searches = {}
    predictions = {}
    rows = []

    for model_name, (estimator, grid) in models.items():
        print(f"[{modality}] Fitting {model_name}...")
        search = fit_grid_search(
            train_df, y_train, cfg, modality, model_name, estimator, grid
        )
        prob = search.predict_proba(test_df)[:, 1]
        metrics = metric_dict(y_test, prob, cfg.classification_threshold)

        rows.append({
            "Model": model_name,
            "Modality": modality,
            "CV_best_AUROC": search.best_score_,
            **metrics,
            "Best_params": json.dumps(search.best_params_),
        })
        searches[model_name] = search
        predictions[model_name] = prob

    table = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    return table, searches, predictions


# ============================================================
# FINAL LIGHTGBM BY MODALITY
# ============================================================

def fit_lightgbm_modality(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    modality: str,
) -> Tuple[GridSearchCV, np.ndarray, Dict]:
    estimator, grid = get_models_and_grids(cfg)["LightGBM"]
    search = fit_grid_search(
        train_df,
        train_df[cfg.target_col],
        cfg,
        modality,
        "LightGBM",
        estimator,
        grid,
    )
    prob = search.predict_proba(test_df)[:, 1]
    evaluation = evaluate_predictions(
        test_df[cfg.target_col].values,
        prob,
        f"LightGBM-{modality}",
        cfg,
    )
    return search, prob, evaluation


# ============================================================
# EXTERNAL VALIDATION
# ============================================================

def two_centre_external_validation(
    df: pd.DataFrame,
    cfg: Config,
    modality: str = "combined",
) -> pd.DataFrame:
    centres = list(pd.Series(df[cfg.centre_col].dropna().unique()).astype(str))
    if len(centres) != 2:
        raise ValueError(
            f"Expected exactly two centres; found {len(centres)}: {centres}"
        )

    # Ensure comparison as strings
    temp = df.copy()
    temp["_centre_str"] = temp[cfg.centre_col].astype(str)

    rows = []
    estimator, grid = get_models_and_grids(cfg)["LightGBM"]

    for train_c, test_c in [(centres[0], centres[1]), (centres[1], centres[0])]:
        tr = temp[temp["_centre_str"] == train_c].copy()
        te = temp[temp["_centre_str"] == test_c].copy()

        if tr[cfg.target_col].nunique() < 2 or te[cfg.target_col].nunique() < 2:
            raise ValueError("Both centres must contain positive and negative cases.")

        print(f"External validation: train {train_c} -> test {test_c}")

        search = fit_grid_search(
            tr, tr[cfg.target_col], cfg, modality,
            "LightGBM", estimator, grid
        )
        prob = search.predict_proba(te)[:, 1]
        metrics = metric_dict(te[cfg.target_col], prob, cfg.classification_threshold)
        cis = bootstrap_metric_cis(
            te[cfg.target_col], prob,
            threshold=cfg.classification_threshold,
            n_boot=cfg.bootstrap_iterations,
            seed=cfg.random_state,
        )
        cal = calibration_intercept_slope(te[cfg.target_col], prob)

        row = {
            "Train_centre": train_c,
            "Test_centre": test_c,
            "N_train": len(tr),
            "N_test": len(te),
            "Prevalence_test": te[cfg.target_col].mean(),
            **metrics,
            **cal,
        }
        for metric_name, (lo, hi) in cis.items():
            row[f"{metric_name}_CI_low"] = lo
            row[f"{metric_name}_CI_high"] = hi

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# SHAP
# ============================================================

def shap_analysis(
    fitted_search: GridSearchCV,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    out_dir: Path,
    prefix: str = "combined",
):
    pipe = fitted_search.best_estimator_
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    X_trans = preprocess.transform(X_eval)
    feature_names = preprocess.get_feature_names_out()

    # LightGBM TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_trans)

    # For plotting reliability with sparse inputs, use a manageable dense copy.
    max_plot_n = min(2000, X_trans.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_trans.shape[0], size=max_plot_n, replace=False)
    X_plot = X_trans[idx]
    if hasattr(X_plot, "toarray"):
        X_plot = X_plot.toarray()

    values = shap_values.values
    if values.ndim == 3:
        # Some SHAP versions return class dimension.
        values = values[:, :, -1]
    values_plot = values[idx]

    plt.figure()
    shap.summary_plot(
        values_plot,
        X_plot,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_beeswarm_{prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Mean absolute SHAP table
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(out_dir / f"shap_importance_{prefix}.csv", index=False)

    # Local explanations: one highest-risk and one lowest-risk prediction.
    prob = pipe.predict_proba(X_eval)[:, 1]
    high_i = int(np.argmax(prob))
    low_i = int(np.argmin(prob))

    for label, i in [("high_risk", high_i), ("low_risk", low_i)]:
        x_i = X_trans[i]
        if hasattr(x_i, "toarray"):
            x_i = x_i.toarray().ravel()
        else:
            x_i = np.asarray(x_i).ravel()

        v_i = values[i]
        explanation = shap.Explanation(
            values=v_i,
            base_values=np.asarray(shap_values.base_values)[i]
                if np.asarray(shap_values.base_values).ndim > 0
                else shap_values.base_values,
            data=x_i,
            feature_names=list(feature_names),
        )
        plt.figure()
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_waterfall_{prefix}_{label}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    return importance


# ============================================================
# MANUSCRIPT-FRIENDLY TABLE FORMATTER
# ============================================================

def flatten_evaluation(eval_obj: Dict) -> pd.DataFrame:
    rows = []
    for metric, value in eval_obj["point"].items():
        lo, hi = eval_obj["ci95"].get(metric, (np.nan, np.nan))
        rows.append({
            "Metric": metric,
            "Estimate": value,
            "CI_2.5": lo,
            "CI_97.5": hi,
            "Formatted": f"{value:.3f} ({lo:.3f}-{hi:.3f})"
                if np.isfinite(lo) else f"{value:.3f}",
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(CFG.random_state)
    out = ensure_dir(CFG.output_dir)

    print("Loading:", CFG.data_path)
    raw = pd.read_csv(CFG.data_path)
    df = prepare_dataframe(raw, CFG)

    # Save cohort checks and baseline table
    checks = cohort_checks(df, CFG)
    save_json(checks, out / "cohort_checks.json")

    baseline = make_baseline_table(df, CFG)
    baseline.to_csv(out / "table_baseline_characteristics.csv", index=False)

    # --------------------------------------------------------
    # INTERNAL DEVELOPMENT / LOCKED TEST SPLIT
    # --------------------------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=CFG.test_size,
        stratify=df[CFG.target_col],
        random_state=CFG.random_state,
    )

    split_manifest = pd.DataFrame({
        CFG.patient_id: df[CFG.patient_id],
        "split": np.where(df.index.isin(test_df.index), "test", "train"),
        "centre": df[CFG.centre_col].astype(str),
        "target": df[CFG.target_col],
    })
    split_manifest.to_csv(out / "patient_split_manifest.csv", index=False)

    # 10-model benchmark on combined data, matching the manuscript.
    benchmark, searches, bench_probs = benchmark_models(
        train_df, test_df, CFG, modality="combined"
    )
    benchmark.to_csv(out / "table_10_model_combined_benchmark.csv", index=False)

    # ROC curves for all ten combined-data models.
    plot_roc(
        test_df[CFG.target_col].values,
        bench_probs,
        out / "figure_roc_10_models_combined.png",
    )

    # --------------------------------------------------------
    # LIGHTGBM: COMBINED vs TEXT vs STRUCTURED
    # --------------------------------------------------------
    final_searches = {}
    final_probs = {}
    final_evals = {}

    for modality in ["combined", "text", "structured"]:
        print(f"Final LightGBM modality: {modality}")
        search, prob, ev = fit_lightgbm_modality(
            train_df, test_df, CFG, modality
        )
        final_searches[modality] = search
        final_probs[modality] = prob
        final_evals[modality] = ev

        flatten_evaluation(ev).to_csv(
            out / f"metrics_lightgbm_{modality}_with_95CI.csv",
            index=False,
        )
        joblib.dump(search.best_estimator_, out / f"model_lightgbm_{modality}.joblib")

    # Combined table
    combined_rows = []
    for modality, ev in final_evals.items():
        row = {"Modality": modality}
        for metric, value in ev["point"].items():
            lo, hi = ev["ci95"].get(metric, (np.nan, np.nan))
            row[metric] = value
            row[f"{metric}_CI_low"] = lo
            row[f"{metric}_CI_high"] = hi
        combined_rows.append(row)
    pd.DataFrame(combined_rows).to_csv(
        out / "table_lightgbm_modality_comparison_with_95CI.csv", index=False
    )

    # ROC and PR comparison
    plot_roc(
        test_df[CFG.target_col].values,
        {
            "Combined": final_probs["combined"],
            "Text": final_probs["text"],
            "Structured": final_probs["structured"],
        },
        out / "figure_roc_lightgbm_modalities.png",
    )
    plot_pr(
        test_df[CFG.target_col].values,
        {
            "Combined": final_probs["combined"],
            "Text": final_probs["text"],
            "Structured": final_probs["structured"],
        },
        out / "figure_pr_lightgbm_modalities.png",
    )

    # --------------------------------------------------------
    # PAIRED DELONG TESTS
    # --------------------------------------------------------
    delong_results = []
    pairs = [
        ("combined", "text"),
        ("combined", "structured"),
        ("text", "structured"),
    ]
    y_test = test_df[CFG.target_col].values
    for a, b in pairs:
        res = delong_roc_test(y_test, final_probs[a], final_probs[b])
        res.update({"model_1": a, "model_2": b})
        delong_results.append(res)
    pd.DataFrame(delong_results).to_csv(
        out / "table_delong_auc_comparisons.csv", index=False
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------
    cal_rows = []
    for modality, prob in final_probs.items():
        row = {
            "Modality": modality,
            "Brier": brier_score_loss(y_test, prob),
            **calibration_intercept_slope(y_test, prob),
        }
        cal_rows.append(row)
    pd.DataFrame(cal_rows).to_csv(
        out / "table_calibration_statistics.csv", index=False
    )

    plot_calibration(
        y_test,
        {
            "Combined": final_probs["combined"],
            "Text": final_probs["text"],
            "Structured": final_probs["structured"],
        },
        out / "figure_calibration_lightgbm_modalities.png",
    )

    # --------------------------------------------------------
    # DECISION CURVE ANALYSIS
    # --------------------------------------------------------
    plot_dca(
        y_test,
        {
            "Combined": final_probs["combined"],
            "Text": final_probs["text"],
            "Structured": final_probs["structured"],
        },
        CFG,
        out / "figure_decision_curve_analysis.png",
    )

    thresholds = np.linspace(
        CFG.dca_min_threshold, CFG.dca_max_threshold, CFG.dca_points
    )
    for modality, prob in final_probs.items():
        decision_curve(y_test, prob, thresholds).to_csv(
            out / f"dca_values_{modality}.csv", index=False
        )

    # --------------------------------------------------------
    # SHAP FOR FINAL COMBINED LIGHTGBM
    # --------------------------------------------------------
    print("Running SHAP analysis...")
    shap_analysis(
        final_searches["combined"],
        test_df,
        test_df[CFG.target_col],
        out,
        prefix="combined",
    )

    # Selected feature names
    pp = final_searches["combined"].best_estimator_.named_steps["preprocess"]
    pd.DataFrame({"feature": pp.get_feature_names_out()}).to_csv(
        out / "selected_features_combined.csv", index=False
    )

    # --------------------------------------------------------
    # TWO-CENTRE EXTERNAL VALIDATION
    # --------------------------------------------------------
    if df[CFG.centre_col].nunique(dropna=True) == 2:
        print("Running bidirectional two-centre external validation...")
        external = two_centre_external_validation(df, CFG, modality="combined")
        external.to_csv(
            out / "table_two_centre_external_validation.csv", index=False
        )
    else:
        print(
            "External validation skipped: centre column does not contain exactly "
            "two non-missing centres."
        )

    # Save config
    save_json(CFG.__dict__, out / "analysis_config.json")

    print("\nDONE. Outputs written to:", out.resolve())


if __name__ == "__main__":
    main()
