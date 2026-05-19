import os
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "modeling_outputs_mvp"

SUBGROUP_FILE = os.path.join(
    OUTPUT_DIR,
    "subgroup_metrics_best_model.csv"
)

FAIRNESS_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "fairness_gap_summary.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_gap(df, subgroup_code, comparison_name, main_issue):
    """Compute recall, FNR, and FPR gaps for one subgroup comparison."""
    temp = df[df["subgroup"] == subgroup_code].copy()

    if temp.empty:
        raise ValueError(f"No rows found for subgroup: {subgroup_code}")

    recall_gap = temp["recall"].max() - temp["recall"].min()
    fnr_gap = temp["fnr"].max() - temp["fnr"].min()
    fpr_gap = temp["fpr"].max() - temp["fpr"].min()

    return {
        "Comparison": comparison_name,
        "Recall Gap": round(recall_gap, 3),
        "FNR Gap": round(fnr_gap, 3),
        "FPR Gap": round(fpr_gap, 3),
        "Main Issue": main_issue,
    }


def main():
    if not os.path.exists(SUBGROUP_FILE):
        raise FileNotFoundError(
            f"Could not find: {SUBGROUP_FILE}\n"
            "Make sure modeling_diabetes_mvp.py has been run first."
        )

    df = pd.read_csv(SUBGROUP_FILE)

    required_columns = {"subgroup", "group", "recall", "fnr", "fpr"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns are: {list(df.columns)}"
        )

    fairness_rows = [
        compute_gap(
            df=df,
            subgroup_code="SEX",
            comparison_name="Sex",
            main_issue="Lower recall for females",
        ),
        compute_gap(
            df=df,
            subgroup_code="INCOME_GROUP",
            comparison_name="Income",
            main_issue="Lower recall for high-income respondents",
        ),
        compute_gap(
            df=df,
            subgroup_code="AGE_GROUP",
            comparison_name="Age",
            main_issue="Younger respondents more often missed",
        ),
        compute_gap(
            df=df,
            subgroup_code="EDUCATION_GROUP",
            comparison_name="Education",
            main_issue="Lower recall for high-education respondents",
        ),
        compute_gap(
            df=df,
            subgroup_code="AGE_EDUCATION_GROUP",
            comparison_name="Age × education",
            main_issue="Young high-education respondents most often missed",
        ),
        compute_gap(
            df=df,
            subgroup_code="SEX_INCOME_GROUP",
            comparison_name="Sex × income",
            main_issue="Female high-income respondents most often missed",
        ),
    ]

    fairness_df = pd.DataFrame(fairness_rows)

    fairness_df.to_csv(FAIRNESS_OUTPUT_FILE, index=False)

    print("\nFairness gap summary:")
    print(fairness_df.to_string(index=False))

    print("\nSaved to:")
    print(FAIRNESS_OUTPUT_FILE)


if __name__ == "__main__":
    main()