"""
Sprint 4 — MVP Modeling Pipeline (BRFSS 2023 Diabetes Classification)

Builds on Sprint 3 and adds:
1) Threshold tuning on TRAIN via out-of-fold (OOF) probabilities (optimize F1)
2) Imbalance handling (class weights / scale_pos_weight)
3) Light hyperparameter tuning (small randomized search, fast & reproducible)
4) Stronger outputs: confusion matrices, summary files, subgroup metrics incl. FPR/FNR, PR-AUC

INPUT:
- modeling_dataset.csv  (created in preprocessing step)

OUTPUTS (in modeling_outputs_mvp/):
- cv_oof_summary.csv
- best_params.json
- test_results.csv
- confusion_matrix_test_threshold05.csv
- confusion_matrix_test_tuned.csv
- subgroup_metrics_best_model.csv
- run_summary.txt
- best_model_test_probs.csv  (optional helpful file for plots later)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,  # PR-AUC
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


# -----------------------------
# 0) CONFIG
# -----------------------------
warnings.filterwarnings("ignore")

DATA_FILE = "modeling_dataset.csv"
OUT_DIR = Path("modeling_outputs_mvp")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5

# Threshold tuning grid
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)

TARGET_COL = "TARGET_BIN"
SUBGROUP_COLS = ["SEX", "EDUCATION", "INCOME"]

# RandomizedSearch budget (keep light for MVP)
N_ITER_SEARCH = 12


# -----------------------------
# 1) UTIL FUNCTIONS
# -----------------------------
def pick_threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """Pick threshold that maximizes F1 on provided probs."""
    best_t = 0.50
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Compute threshold-independent + threshold-dependent metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    # Threshold-independent metrics
    roc_auc = np.nan
    pr_auc = np.nan
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

    # Threshold-dependent metrics
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "roc_auc": float(roc_auc) if roc_auc == roc_auc else np.nan,
        "pr_auc": float(pr_auc) if pr_auc == pr_auc else np.nan,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns confusion-matrix-derived rates:
    - FPR = FP/(FP+TN)
    - FNR = FN/(FN+TP)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": float(fpr) if fpr == fpr else np.nan,
        "fnr": float(fnr) if fnr == fnr else np.nan,
    }


def safe_confusion_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])


def subgroup_metrics(
    df_test: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    subgroup_cols: list[str],
    target_col: str
) -> pd.DataFrame:
    """
    Compute subgroup metrics on TEST set at a given threshold.
    Reports: support, prevalence, ROC-AUC, PR-AUC, precision/recall/F1, FPR/FNR, confusion counts.
    """
    rows = []

    # Ensure aligned indexing: assume y_prob corresponds to df_test row order
    y_true_all = df_test[target_col].astype(int).to_numpy()
    y_pred_all = (y_prob >= threshold).astype(int)

    for col in subgroup_cols:
        if col not in df_test.columns:
            continue

        for group_value, sub_df in df_test.groupby(col, dropna=False):
            idx = sub_df.index.to_numpy()
            # Convert index positions into positional selection
            pos_idx = df_test.index.get_indexer(idx)

            y_true = y_true_all[pos_idx]
            sub_prob = y_prob[pos_idx]
            sub_pred = (sub_prob >= threshold).astype(int)

            support = int(len(y_true))
            prevalence = float(np.mean(y_true)) if support > 0 else np.nan

            # Threshold-independent subgroup metrics only if both classes exist
            roc_auc = np.nan
            pr_auc = np.nan
            if support > 0 and len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(y_true, sub_prob)
                pr_auc = average_precision_score(y_true, sub_prob)

            base_metrics = {
                "subgroup": col,
                "group": str(group_value),
                "support": support,
                "prevalence_diabetes": prevalence,
                "roc_auc": float(roc_auc) if roc_auc == roc_auc else np.nan,
                "pr_auc": float(pr_auc) if pr_auc == pr_auc else np.nan,
                "precision": float(precision_score(y_true, sub_pred, zero_division=0)),
                "recall": float(recall_score(y_true, sub_pred, zero_division=0)),
                "f1": float(f1_score(y_true, sub_pred, zero_division=0)),
            }

            rates = confusion_rates(y_true, sub_pred)
            base_metrics.update(rates)

            rows.append(base_metrics)

    return pd.DataFrame(rows)


# -----------------------------
# 2) LOAD DATA
# -----------------------------
print("Loading modeling dataset...")
df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)

if TARGET_COL not in df.columns:
    raise ValueError(f"Expected target column '{TARGET_COL}' not found in {DATA_FILE}.")

df[TARGET_COL] = df[TARGET_COL].astype(int)
print("Target distribution:\n", df[TARGET_COL].value_counts())

# Feature columns: everything except target
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

# Numeric vs categorical: treat AGE as numeric; everything else categorical
numeric_features = [c for c in feature_cols if c.upper() == "AGE"]
categorical_features = [c for c in feature_cols if c not in numeric_features]


# -----------------------------
# 3) TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain size: {X_train.shape}  Test size: {X_test.shape}")
print("Train target distribution:\n", y_train.value_counts(normalize=True).rename("proportion"))
print("Test target distribution:\n", y_test.value_counts(normalize=True).rename("proportion"))

# XGBoost scale_pos_weight from TRAIN only
n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0


# -----------------------------
# 4) PREPROCESSOR
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ],
    remainder="drop"
)


# -----------------------------
# 5) MODELS + SMALL TUNING SPACES
# -----------------------------
models = {
    "LogisticRegression_baseline": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )
}

param_spaces = {
    "LogisticRegression_baseline": {
        "model__C": np.logspace(-2, 1, 10)  # 0.01 to 10
    },
    "RandomForest": {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [None, 6, 10, 14],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0],
    }
}

cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

print("\nSprint 4: tuning + OOF threshold selection on TRAIN set...")
oof_summary_rows: list[dict] = []
best_params: dict = {}
fitted_pipes: dict[str, Pipeline] = {}
chosen_thresholds: dict[str, float] = {}


def fit_oof_and_threshold(model_name: str, base_model) -> tuple[Pipeline, float, dict]:
    """
    Steps:
    1) RandomizedSearchCV on TRAIN only, tuning by ROC-AUC (threshold-independent)
    2) Manual OOF probabilities using CV and CLONED best estimator per fold (clean/defensible)
    3) Choose threshold that maximizes F1 on OOF predictions
    4) Fit best pipeline on full TRAIN for TEST evaluation
    """
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", base_model)
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_spaces[model_name],
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    search.fit(X_train, y_train)

    best_pipe: Pipeline = search.best_estimator_
    best_params[model_name] = search.best_params_

    # Manual OOF prediction using CLONES of best_pipe (important!)
    oof_prob = np.zeros(len(X_train), dtype=float)

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]

        fold_pipe = clone(best_pipe)
        fold_pipe.fit(X_tr, y_tr)
        oof_prob[val_idx] = fold_pipe.predict_proba(X_va)[:, 1]

    oof_y = y_train.to_numpy()

    # Threshold choice on OOF predictions (optimize F1)
    chosen_t, _ = pick_threshold_max_f1(oof_y, oof_prob, THRESHOLDS)

    # OOF metrics at tuned threshold and at 0.5
    tuned_metrics = metrics_from_probs(oof_y, oof_prob, chosen_t)
    default_metrics = metrics_from_probs(oof_y, oof_prob, 0.50)

    # Fit best pipeline on full TRAIN for final test evaluation
    best_pipe.fit(X_train, y_train)

    oof_info = {
        "threshold_tuned": float(chosen_t),

        # tuned threshold metrics
        "oof_roc_auc": tuned_metrics["roc_auc"],
        "oof_pr_auc": tuned_metrics["pr_auc"],
        "oof_precision_tuned": tuned_metrics["precision"],
        "oof_recall_tuned": tuned_metrics["recall"],
        "oof_f1_tuned": tuned_metrics["f1"],

        # baseline 0.5 metrics
        "oof_precision_thr05": default_metrics["precision"],
        "oof_recall_thr05": default_metrics["recall"],
        "oof_f1_thr05": default_metrics["f1"],
        "oof_roc_auc_thr05": default_metrics["roc_auc"],
        "oof_pr_auc_thr05": default_metrics["pr_auc"],
    }

    return best_pipe, chosen_t, oof_info


for name, model in models.items():
    print(f"  - Tuning + OOF threshold selection for: {name}")
    best_pipe, t_star, oof_info = fit_oof_and_threshold(name, model)

    fitted_pipes[name] = best_pipe
    chosen_thresholds[name] = float(t_star)

    row = {"model": name}
    row.update(oof_info)
    oof_summary_rows.append(row)

oof_summary = pd.DataFrame(oof_summary_rows).sort_values("oof_f1_tuned", ascending=False)
oof_summary.to_csv(OUT_DIR / "cv_oof_summary.csv", index=False)

with open(OUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)

print(f"\nSaved OOF summary to: {OUT_DIR / 'cv_oof_summary.csv'}")
print("OOF summary (sorted by tuned OOF F1):")
print(oof_summary[[
    "model", "threshold_tuned",
    "oof_roc_auc", "oof_pr_auc",
    "oof_precision_tuned", "oof_recall_tuned", "oof_f1_tuned"
]])


# -----------------------------
# 6) TEST EVALUATION (thr=0.5 vs tuned thr)
# -----------------------------
test_rows: list[dict] = []

best_model_name = str(oof_summary.iloc[0]["model"])
best_model_threshold = float(oof_summary.iloc[0]["threshold_tuned"])

for name, pipe in fitted_pipes.items():
    prob_test = pipe.predict_proba(X_test)[:, 1]
    y_true_test = y_test.to_numpy()

    # Default threshold 0.5
    m05 = metrics_from_probs(y_true_test, prob_test, 0.50)
    pred05 = (prob_test >= 0.50).astype(int)
    rates05 = confusion_rates(y_true_test, pred05)

    # Tuned threshold (from OOF)
    t_star = chosen_thresholds[name]
    mt = metrics_from_probs(y_true_test, prob_test, t_star)
    predt = (prob_test >= t_star).astype(int)
    ratest = confusion_rates(y_true_test, predt)

    test_rows.append({
        "model": name,
        "threshold_used": 0.50,
        **m05,
        **rates05,
    })
    test_rows.append({
        "model": name,
        "threshold_used": float(t_star),
        **mt,
        **ratest,
    })

test_df = pd.DataFrame(test_rows)
test_df.to_csv(OUT_DIR / "test_results.csv", index=False)

print("\nTest results saved to:", OUT_DIR / "test_results.csv")
print("Best model (by tuned OOF F1):", best_model_name)
print("Best model tuned threshold:", best_model_threshold)

# Confusion matrices for best model (0.5 and tuned threshold)
best_prob_test = fitted_pipes[best_model_name].predict_proba(X_test)[:, 1]
best_pred_05 = (best_prob_test >= 0.50).astype(int)
best_pred_tuned = (best_prob_test >= best_model_threshold).astype(int)

cm05 = safe_confusion_df(y_test.to_numpy(), best_pred_05)
cmt = safe_confusion_df(y_test.to_numpy(), best_pred_tuned)

cm05.to_csv(OUT_DIR / "confusion_matrix_test_threshold05.csv", index=True)
cmt.to_csv(OUT_DIR / "confusion_matrix_test_tuned.csv", index=True)

# Save probabilities for best model (helpful for plots/appendix)
best_probs_out = pd.DataFrame({
    "y_true": y_test.to_numpy(),
    "y_prob": best_prob_test,
    "y_pred_thr05": best_pred_05,
    "y_pred_tuned": best_pred_tuned
})
best_probs_out.to_csv(OUT_DIR / "best_model_test_probs.csv", index=False)

# -----------------------------
# 6b) THRESHOLD SWEEP (best model, TEST)
# -----------------------------
threshold_rows = []
y_true_best = y_test.to_numpy()

for t in THRESHOLDS:
    m = metrics_from_probs(y_true_best, best_prob_test, float(t))
    pred_t = (best_prob_test >= t).astype(int)
    rates_t = confusion_rates(y_true_best, pred_t)

    threshold_rows.append({
        "threshold": float(t),
        **m,
        **rates_t,
    })

threshold_sweep_df = pd.DataFrame(threshold_rows)
threshold_sweep_df.to_csv(OUT_DIR / "threshold_sweep_best_model.csv", index=False)
print("Threshold sweep saved to:", OUT_DIR / "threshold_sweep_best_model.csv")

# -----------------------------
# 7) SUBGROUP METRICS (TEST, best model, tuned threshold)
# -----------------------------

# IMPORTANT:
# Store probabilities and predictions directly in the test dataframe
# before filtering any subgroup rows. This keeps y_true, y_prob,
# and subgroup labels correctly aligned.
df_test_for_subgroup = X_test.copy().reset_index(drop=True)
df_test_for_subgroup[TARGET_COL] = y_test.reset_index(drop=True).astype(int)
df_test_for_subgroup["y_prob"] = best_prob_test
df_test_for_subgroup["y_pred"] = best_pred_tuned


def subgroup_metrics_from_df(
    df_test: pd.DataFrame,
    subgroup_cols: list[str],
    target_col: str = TARGET_COL,
    prob_col: str = "y_prob",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """
    Compute subgroup metrics directly from a dataframe that already contains:
    - true labels
    - predicted probabilities
    - predicted classes

    This avoids alignment errors when filtering subgroup rows.
    """
    rows = []

    for col in subgroup_cols:
        if col not in df_test.columns:
            continue

        for group_value, sub_df in df_test.groupby(col, dropna=False):
            if pd.isna(group_value):
                continue

            y_true = sub_df[target_col].astype(int).to_numpy()
            y_prob = sub_df[prob_col].astype(float).to_numpy()
            y_pred = sub_df[pred_col].astype(int).to_numpy()

            support = int(len(y_true))
            positives = int(y_true.sum())
            prevalence = float(positives / support) if support > 0 else np.nan

            # Threshold-independent subgroup metrics only if both classes exist
            roc_auc = np.nan
            pr_auc = np.nan
            if support > 0 and len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            rates = confusion_rates(y_true, y_pred)

            rows.append({
                "subgroup": col,
                "group": str(group_value),
                "support": support,
                "positive_cases": positives,
                "prevalence_diabetes": prevalence,
                "roc_auc": float(roc_auc) if roc_auc == roc_auc else np.nan,
                "pr_auc": float(pr_auc) if pr_auc == pr_auc else np.nan,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                **rates,
            })

    return pd.DataFrame(rows)


# -------------------------------------------------
# Additional subgroup variables
# -------------------------------------------------

# Age: young vs old
df_test_for_subgroup["AGE_GROUP"] = np.where(
    df_test_for_subgroup["AGE"] < 50,
    "Young",
    "Old"
)

# Education: low vs high
# 1-4 = low education
# 5-6 = high education
# 9 = refused/unknown -> remove from subgroup analysis
df_test_for_subgroup["EDUCATION_GROUP"] = "Unknown"

df_test_for_subgroup.loc[
    df_test_for_subgroup["EDUCATION"].isin([1, 2, 3, 4]),
    "EDUCATION_GROUP"
] = "Low education"

df_test_for_subgroup.loc[
    df_test_for_subgroup["EDUCATION"].isin([5, 6]),
    "EDUCATION_GROUP"
] = "High education"

# Drop refused/unknown education for subgroup analysis
df_test_for_subgroup = df_test_for_subgroup[
    df_test_for_subgroup["EDUCATION_GROUP"].isin(["Low education", "High education"])
].copy()

# Income: low vs high
# 1-5 = low income
# 6+ = high income
df_test_for_subgroup["INCOME_GROUP"] = np.where(
    df_test_for_subgroup["INCOME"] <= 5,
    "Low income",
    "High income"
)

# -------------------------------------------------
# Combination subgroup variables
# -------------------------------------------------

# Age x Education
df_test_for_subgroup["AGE_EDUCATION_GROUP"] = (
    df_test_for_subgroup["AGE_GROUP"].astype(str)
    + " | "
    + df_test_for_subgroup["EDUCATION_GROUP"].astype(str)
)

# Sex x Income
df_test_for_subgroup["SEX_INCOME_GROUP"] = (
    df_test_for_subgroup["SEX"].astype(str)
    + " | "
    + df_test_for_subgroup["INCOME_GROUP"].astype(str)
)

# -------------------------------------------------
# Subgroup list
# -------------------------------------------------
subgroup_cols_extended = [
    "SEX",
    "INCOME_GROUP",
    "AGE_GROUP",
    "EDUCATION_GROUP",
    "AGE_EDUCATION_GROUP",
    "SEX_INCOME_GROUP"
]

sub_df = subgroup_metrics_from_df(
    df_test=df_test_for_subgroup,
    subgroup_cols=subgroup_cols_extended,
    target_col=TARGET_COL,
    prob_col="y_prob",
    pred_col="y_pred",
)

sub_df.to_csv(OUT_DIR / "subgroup_metrics_best_model.csv", index=False)
print("Subgroup metrics saved to:", OUT_DIR / "subgroup_metrics_best_model.csv")

# -----------------------------
# 7b) FEATURE IMPORTANCE (best model)
# -----------------------------
feature_importance_df = None

if best_model_name == "XGBoost":
    # Get fitted preprocessing + fitted xgboost model
    best_pipe = fitted_pipes[best_model_name]
    fitted_preprocessor = best_pipe.named_steps["preprocess"]
    fitted_model = best_pipe.named_steps["model"]

    # Get transformed feature names after preprocessing
    feature_names = fitted_preprocessor.get_feature_names_out()

    # XGBoost gain-based importance
    importances = fitted_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(OUT_DIR / "feature_importance_best_model.csv", index=False)
    print("Feature importance saved to:", OUT_DIR / "feature_importance_best_model.csv")
    
    # Aggregate encoded features back to original variables
    def map_back_to_original(feature_name: str) -> str:
        if feature_name.startswith("num__"):
            return feature_name.replace("num__", "")
        if feature_name.startswith("cat__"):
            stripped = feature_name.replace("cat__", "")
            for col in categorical_features:
                if stripped.startswith(col + "_"):
                    return col
            return stripped
        return feature_name

    feature_importance_df["original_variable"] = feature_importance_df["feature"].apply(map_back_to_original)

    feature_importance_grouped = (
        feature_importance_df
        .groupby("original_variable", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    feature_importance_grouped.to_csv(OUT_DIR / "feature_importance_grouped_best_model.csv", index=False)
    print("Grouped feature importance saved to:", OUT_DIR / "feature_importance_grouped_best_model.csv")

# -----------------------------
# 8) RUN SUMMARY TEXT
# -----------------------------
summary_lines = []
summary_lines.append("SPRINT 4 MVP RUN SUMMARY")
summary_lines.append("")
summary_lines.append(f"Dataset: {DATA_FILE}")
summary_lines.append(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
summary_lines.append(f"CV: StratifiedKFold(n_splits={N_SPLITS_CV}, shuffle=True, random_state={RANDOM_STATE})")
summary_lines.append("")
summary_lines.append("Tuning objective: ROC-AUC (threshold-independent).")
summary_lines.append("Threshold selection: maximize F1 on OOF TRAIN probabilities.")
summary_lines.append("")
summary_lines.append(f"Best model by tuned OOF F1: {best_model_name}")
summary_lines.append(f"Chosen threshold (tuned on TRAIN OOF): {best_model_threshold}")
summary_lines.append("")
summary_lines.append("Outputs:")
summary_lines.append(f"- {OUT_DIR / 'cv_oof_summary.csv'}")
summary_lines.append(f"- {OUT_DIR / 'best_params.json'}")
summary_lines.append(f"- {OUT_DIR / 'test_results.csv'}")
summary_lines.append(f"- {OUT_DIR / 'confusion_matrix_test_threshold05.csv'}")
summary_lines.append(f"- {OUT_DIR / 'confusion_matrix_test_tuned.csv'}")
summary_lines.append(f"- {OUT_DIR / 'subgroup_metrics_best_model.csv'}")
summary_lines.append(f"- {OUT_DIR / 'best_model_test_probs.csv'}")
summary_lines.append(f"- {OUT_DIR / 'feature_importance_best_model.csv'}")
summary_lines.append(f"- {OUT_DIR / 'feature_importance_grouped_best_model.csv'}")
summary_lines.append(f"- {OUT_DIR / 'threshold_sweep_best_model.csv'}")

(OUT_DIR / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

print("\nDone. Sprint 4 MVP outputs in:", OUT_DIR.resolve())