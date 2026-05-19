import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "modeling_dataset.csv"
OUTPUT_DIR = "modeling_outputs_nested_cv"

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
THRESHOLD_FOLDS = 5
N_ITER = 12

# Change this if your target column has a different name.
POSSIBLE_TARGET_COLS = [
    "TARGET_BIN",
    "diabetes",
    "DIABETES",
    "diabetes_binary",
    "DIABETES_BINARY",
    "target",
    "TARGET",
    "DIABETES_STATUS",
    "diabetes_status",
]

# If your age column has a different name, add it here.
POSSIBLE_NUMERIC_COLS = [
    "AGE",
    "age",
    "_AGE80",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_target_column(df: pd.DataFrame) -> str:
    for col in POSSIBLE_TARGET_COLS:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not automatically find target column. "
        f"Available columns are: {list(df.columns)}. "
        "Please add your target column name to POSSIBLE_TARGET_COLS."
    )


def get_feature_columns(df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c != target_col]

    numeric_cols = [c for c in feature_cols if c in POSSIBLE_NUMERIC_COLS]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return feature_cols, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))

    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def choose_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def make_models_and_param_spaces(scale_pos_weight):
    models = {}

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    lr_params = {
        "model__C": np.logspace(-3, 2, 20),
    }
    models["Logistic Regression"] = (lr, lr_params)

    # Random Forest
    rf = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_params = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    models["Random Forest"] = (rf, rf_params)

    # XGBoost
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )
    xgb_params = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    models["XGBoost"] = (xgb, xgb_params)

    return models


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    target_col = find_target_column(df)
    feature_cols, numeric_cols, categorical_cols = get_feature_columns(df, target_col)

    print(f"Target column: {target_col}")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Dataset shape: {df.shape}")

    X = df[feature_cols]
    y = df[target_col].astype(int)

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    threshold_cv = StratifiedKFold(
        n_splits=THRESHOLD_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_results = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print("\n" + "=" * 70)
        print(f"Outer fold {outer_fold}/{OUTER_FOLDS}")
        print("=" * 70)

        X_train_outer = X.iloc[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_train_outer = y.iloc[train_idx]
        y_test_outer = y.iloc[test_idx]

        n_pos = int(y_train_outer.sum())
        n_neg = int(len(y_train_outer) - n_pos)
        scale_pos_weight = n_neg / n_pos

        print(f"Outer train positives: {n_pos}")
        print(f"Outer train negatives: {n_neg}")
        print(f"scale_pos_weight: {scale_pos_weight:.3f}")

        models = make_models_and_param_spaces(scale_pos_weight)

        for model_name, (base_model, param_space) in models.items():
            print(f"\nRunning nested CV for: {model_name}")

            preprocessor = make_preprocessor(numeric_cols, categorical_cols)

            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", base_model),
                ]
            )

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_space,
                n_iter=N_ITER,
                scoring="roc_auc",
                cv=inner_cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            )

            # Inner CV hyperparameter tuning on outer training data
            search.fit(X_train_outer, y_train_outer)

            best_estimator = search.best_estimator_
            print(f"Best inner ROC-AUC: {search.best_score_:.4f}")
            print(f"Best params: {search.best_params_}")

            # OOF threshold selection inside outer training data
            print("Selecting threshold using OOF predictions...")
            oof_probs = cross_val_predict(
                clone(best_estimator),
                X_train_outer,
                y_train_outer,
                cv=threshold_cv,
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]

            best_threshold, best_oof_f1 = choose_best_threshold(
                y_train_outer,
                oof_probs,
            )

            print(f"Selected threshold: {best_threshold:.2f}")
            print(f"OOF F1 at selected threshold: {best_oof_f1:.4f}")

            # Fit final model on full outer training fold
            best_estimator.fit(X_train_outer, y_train_outer)

            # Evaluate on outer test fold
            test_probs = best_estimator.predict_proba(X_test_outer)[:, 1]
            metrics = compute_metrics(y_test_outer, test_probs, best_threshold)

            result = {
                "outer_fold": outer_fold,
                "model": model_name,
                "inner_best_roc_auc": search.best_score_,
                "oof_threshold_f1": best_oof_f1,
                **metrics,
            }

            all_results.append(result)

            print(
                f"Outer test ROC-AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(OUTPUT_DIR, "nested_cv_outer_fold_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\nSaved outer-fold results to:")
    print(results_path)

    # Summary table
    summary = (
        results_df
        .groupby("model")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
        )
        .reset_index()
    )

    summary_path = os.path.join(OUTPUT_DIR, "nested_cv_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nNested CV summary:")
    print(summary.round(4))
    print("\nSaved summary to:")
    print(summary_path)

    # Paired tests: XGBoost versus each alternative model
    paired_rows = []

    for metric in ["roc_auc", "pr_auc", "f1", "precision", "recall"]:
        pivot = results_df.pivot(
            index="outer_fold",
            columns="model",
            values=metric,
        )

        if "XGBoost" not in pivot.columns:
            continue

        for comparison_model in ["Logistic Regression", "Random Forest"]:
            if comparison_model not in pivot.columns:
                continue

            xgb_scores = pivot["XGBoost"]
            other_scores = pivot[comparison_model]
            diff = xgb_scores - other_scores

            # Paired t-test
            t_stat, t_p = ttest_rel(xgb_scores, other_scores)

            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = wilcoxon(xgb_scores, other_scores)
            except ValueError:
                w_stat, w_p = np.nan, np.nan

            paired_rows.append(
                {
                    "metric": metric,
                    "comparison": f"XGBoost - {comparison_model}",
                    "xgboost_mean": xgb_scores.mean(),
                    "other_mean": other_scores.mean(),
                    "mean_difference": diff.mean(),
                    "std_difference": diff.std(),
                    "paired_t_p_value": t_p,
                    "wilcoxon_p_value": w_p,
                }
            )

    paired_df = pd.DataFrame(paired_rows)
    paired_path = os.path.join(OUTPUT_DIR, "nested_cv_pairwise_tests.csv")
    paired_df.to_csv(paired_path, index=False)

    print("\nPairwise tests:")
    print(paired_df.round(4))
    print("\nSaved pairwise tests to:")
    print(paired_path)

    print("\nDone.")


if __name__ == "__main__":
    main()