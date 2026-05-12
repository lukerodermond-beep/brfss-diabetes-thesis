from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
OUT_DIR = Path("modeling_outputs_mvp")
FIG_DIR = OUT_DIR / "thesis_figures"
FIG_DIR.mkdir(exist_ok=True)

TEST_RESULTS_FILE = OUT_DIR / "test_results.csv"
THRESHOLD_SWEEP_FILE = OUT_DIR / "threshold_sweep_best_model.csv"
CM_05_FILE = OUT_DIR / "confusion_matrix_test_threshold05.csv"
CM_TUNED_FILE = OUT_DIR / "confusion_matrix_test_tuned.csv"
FEATURE_IMPORTANCE_FILE = OUT_DIR / "feature_importance_grouped_best_model.csv"
SUBGROUP_FILE = OUT_DIR / "subgroup_metrics_best_model.csv"

BEST_MODEL_NAME = "XGBoost"
SELECTED_THRESHOLD = 0.67


# =========================
# HELPERS
# =========================
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def clean_model_name(name: str) -> str:
    mapping = {
        "LogisticRegression_baseline": "Logistic Regression",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
    }
    return mapping.get(name, name)


def pretty_feature_name(name: str) -> str:
    mapping = {
        "HYPERTENSION": "Hypertension",
        "BMI_CAT": "BMI category",
        "CHOLESTEROL": "Cholesterol",
        "EDUCATION": "Education",
        "PHYS_ACTIVITY": "Physical activity",
        "INCOME": "Income",
        "SEX": "Sex",
        "ALCOHOL_FREQ": "Alcohol consumption",
        "AGE": "Age",
        "SMOKING": "Smoking",
    }
    return mapping.get(name, name)


def save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# FIGURE 1
# Overall model performance
# =========================
def make_model_performance_figure() -> None:
    df = safe_read_csv(TEST_RESULTS_FILE).copy()

    tuned_df = df[df["threshold_used"] != 0.50].copy()
    tuned_df["model"] = tuned_df["model"].apply(clean_model_name)

    plot_df = tuned_df[["model", "roc_auc", "pr_auc", "precision", "recall", "f1"]].rename(
        columns={
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
        }
    )

    ax = plot_df.set_index("model")[["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-score"]].plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title("Overall model performance on the held-out test set")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)

    save_current_figure("figure1_overall_model_performance.png")


# =========================
# FIGURE 2
# Threshold effects for best model
# =========================
def make_threshold_tuning_figure() -> None:
    df = safe_read_csv(THRESHOLD_SWEEP_FILE).sort_values("threshold").copy()

    plt.figure(figsize=(10, 6))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1-score")
    plt.axvline(
        x=SELECTED_THRESHOLD,
        linestyle="--",
        label=f"Selected threshold ({SELECTED_THRESHOLD})"
    )

    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title(f"Effect of classification threshold for {BEST_MODEL_NAME}")
    plt.ylim(0, 1)
    plt.legend()

    save_current_figure("figure2_threshold_effect_xgboost.png")


# =========================
# FIGURE 3
# Classification outcomes at two thresholds
# =========================
def make_classification_outcomes_figure() -> None:
    cm05 = pd.read_csv(CM_05_FILE, index_col=0)
    cmt = pd.read_csv(CM_TUNED_FILE, index_col=0)

    counts = pd.DataFrame({
        "Outcome": [
            "False positives",
            "False negatives",
            "True positives",
            "True negatives",
        ],
        "Threshold 0.50": [
            int(cm05.loc["true_0", "pred_1"]),
            int(cm05.loc["true_1", "pred_0"]),
            int(cm05.loc["true_1", "pred_1"]),
            int(cm05.loc["true_0", "pred_0"]),
        ],
        f"Selected threshold ({SELECTED_THRESHOLD})": [
            int(cmt.loc["true_0", "pred_1"]),
            int(cmt.loc["true_1", "pred_0"]),
            int(cmt.loc["true_1", "pred_1"]),
            int(cmt.loc["true_0", "pred_0"]),
        ],
    })

    ax = counts.set_index("Outcome")[
        ["Threshold 0.50", f"Selected threshold ({SELECTED_THRESHOLD})"]
    ].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(f"Classification outcomes for {BEST_MODEL_NAME} at two thresholds")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")

    save_current_figure("figure3_classification_outcomes_xgboost.png")


# =========================
# FIGURE 4
# Feature importance
# =========================
def make_feature_importance_figure() -> None:
    df = safe_read_csv(FEATURE_IMPORTANCE_FILE).copy()
    df = df.sort_values("importance", ascending=False)
    df["feature_label"] = df["original_variable"].apply(pretty_feature_name)

    ax = df.set_index("feature_label")["importance"].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(f"Gain-based feature importance for {BEST_MODEL_NAME}")
    ax.set_ylabel("Aggregated importance")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")

    save_current_figure("figure4_feature_importance_xgboost.png")


# =========================
# FIGURE 5
# Recall across individual demographic subgroups
# =========================
def make_individual_subgroup_recall_figure() -> None:
    df = safe_read_csv(SUBGROUP_FILE).copy()

    keep_rows = (
        ((df["subgroup"] == "AGE_GROUP") & (df["group"].isin(["Young", "Old"]))) |
        ((df["subgroup"] == "SEX") & (df["group"].astype(str).isin(["Female", "Male", "1", "2", "1.0", "2.0"]))) |
        ((df["subgroup"] == "INCOME_GROUP") & (df["group"].isin(["Low income", "High income"]))) |
        ((df["subgroup"] == "EDUCATION_GROUP") & (df["group"].isin(["Low education", "High education"])))
    )

    plot_df = df[keep_rows].copy()

    sex_label_map = {
        "1": "Male",
        "1.0": "Male",
        "2": "Female",
        "2.0": "Female",
        "Male": "Male",
        "Female": "Female",
    }

    def make_label(row: pd.Series) -> str:
        subgroup = row["subgroup"]
        group = str(row["group"])

        if subgroup == "AGE_GROUP":
            return group
        if subgroup == "SEX":
            return sex_label_map.get(group, group)
        if subgroup == "INCOME_GROUP":
            return group
        if subgroup == "EDUCATION_GROUP":
            return group
        return group

    plot_df["label"] = plot_df.apply(make_label, axis=1)

    order = [
        "Young",
        "Old",
        "Female",
        "Male",
        "High income",
        "Low income",
        "High education",
        "Low education",
    ]

    plot_df["label"] = pd.Categorical(plot_df["label"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("label")

    ax = plot_df.set_index("label")["recall"].plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title("Recall across individual demographic subgroups")
    ax.set_ylabel("Recall")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")

    save_current_figure("figure5_recall_individual_subgroups.png")


# =========================
# FIGURE 6
# Recall across age x education groups
# =========================
def make_recall_age_education_figure() -> None:
    df = safe_read_csv(SUBGROUP_FILE).copy()
    comb_df = df[df["subgroup"] == "AGE_EDUCATION_GROUP"].copy()

    order = [
        "Young | High education",
        "Young | Low education",
        "Old | High education",
        "Old | Low education",
    ]

    comb_df["group"] = pd.Categorical(
        comb_df["group"],
        categories=order,
        ordered=True
    )
    comb_df = comb_df.sort_values("group")

    ax = comb_df.set_index("group")["recall"].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("Recall across age and education groups")
    ax.set_ylabel("Recall")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")

    save_current_figure("figure6_recall_age_education.png")


# =========================
# FIGURE 7
# Recall across sex x income groups
# =========================
def make_recall_sex_income_figure() -> None:
    df = safe_read_csv(SUBGROUP_FILE).copy()
    comb_df = df[df["subgroup"] == "SEX_INCOME_GROUP"].copy()

    comb_df["group"] = (
        comb_df["group"].astype(str)
        .str.replace("1.0 |", "Male |", regex=False)
        .str.replace("2.0 |", "Female |", regex=False)
        .str.replace("1 |", "Male |", regex=False)
        .str.replace("2 |", "Female |", regex=False)
    )

    order = [
        "Female | High income",
        "Female | Low income",
        "Male | High income",
        "Male | Low income",
    ]

    comb_df["group"] = pd.Categorical(
        comb_df["group"],
        categories=order,
        ordered=True
    )
    comb_df = comb_df.sort_values("group")

    ax = comb_df.set_index("group")["recall"].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("Recall across sex and income groups")
    ax.set_ylabel("Recall")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")

    save_current_figure("figure7_recall_sex_income.png")


# =========================
# RUN ALL
# =========================
if __name__ == "__main__":
    make_model_performance_figure()
    make_threshold_tuning_figure()
    make_classification_outcomes_figure()
    make_feature_importance_figure()
    make_individual_subgroup_recall_figure()
    make_recall_age_education_figure()
    make_recall_sex_income_figure()

    print("All final thesis figures saved in:", FIG_DIR.resolve())